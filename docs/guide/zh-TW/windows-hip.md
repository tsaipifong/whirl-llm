[English](../en/windows-hip.md) | **繁體中文**

# Windows 上的 HIP

**對誰有幫助：**任何在 Windows 上撰寫或執行 HIP 程式碼的人——kernel 作者、使用或貢獻 llama.cpp ROCm/HIP 後端的人、移植 Linux ROCm 工具的人，以及 AMD GPU 明明還有大量記憶體卻「記憶體不足」的人。以下大多數內容不在 AMD 的文件裡；每一項都是在我們的機器上量測所得，並註明觀察到的條件。

## <a id="env"></a>0. 這些發現的來源環境

| 元件 | 版本／細節 |
|---|---|
| OS | Windows 11 Home，build 26200（量測期間 2026-09-25 → 2026-10-02） |
| HIP SDK | 7.2（`C:\Program Files\AMD\ROCm\7.2`）：用於裝置端程式碼的 clang、`amdhip64.lib`、標頭檔 |
| HIP runtime | `C:\Windows\System32\amdhip64_7.dll` 10.0.3679.0，由 Adrenalin 驅動程式安裝 |
| 主機端編譯器 | MSVC 19.44（Visual Studio 2022 Build Tools 17.14），C++20 |
| GPU 0 | AMD Radeon 8060S（Ryzen AI Max+ 395），gfx1151，統一記憶體（HIP 回報 99.7 GiB） |
| GPU 1 | AMD Radeon AI PRO R9700，gfx1201，32 GB（HIP 回報 31.9 GiB），**USB4 eGPU** |
| 參考 | llama.cpp b10686 → b11214（ROCm 與 Vulkan 版本），僅用於比較 |

R9700 透過 USB4 連接。標記 **[eGPU]** 的發現取決於這條連結，在直接插 PCIe 插槽的情況下未必能重現。標記 **[8060S]** 的發現是在 APU 上觀察到的。

## <a id="pal"></a>1. 驅動程式堆疊是 PAL，不是 ROCr/KFD

在 Linux 上，HIP 架在 ROCr 與 KFD 核心驅動程式之上。在 Windows 上，HIP 架在 WDDM 底下 AMD 的 **PAL**（Platform Abstraction Library）之上。我們碰到的後果：

- **Linux 工具不存在。**`rocprof`、`rocminfo`、`omniperf`/`rocprof-compute` 都無法使用。沒有可以附加到 HIP 行程的硬體計數器效能分析工具。你得自己建構量測方式（第 3 節）。
- **錯誤訊息來自 PAL。**設定 `AMD_LOG_LEVEL=1` 時，runtime 會把自己的錯誤印到 stderr，內容會提到 `palvirtual.cpp`。沒設定時，應用程式通常只顯示籠統的失敗（`hipErrorLaunchFailure`、llama.cpp 的 `ROCm error: unspecified launch failure`、WHIRL 的 `HipFailed`）。**任何東西失敗時，先設定 `AMD_LOG_LEVEL=1`。**
- **Linux 專屬的建議無法套用。**例子：有人回報在 Linux/ROCm 上，這張卡的 decode（逐 token 生成）迴圈每步會在約 28 與約 36 ms 之間交替，用 `GPU_MAX_HW_QUEUES=1` 修正（KFD 硬體佇列分配問題）。我們在 Windows 上量測，每次執行都用新的行程，有無該變數交錯進行：

  | 執行（未註明者為 R9700） | 不設定 | 設定 `GPU_MAX_HW_QUEUES=1` |
  |---|---|---|
  | 27B 純 decode，8 組 | 全部 16 次執行皆為 28.0 ms/token（35.67–35.74 tok/s，最大–最小 0.2%） | 相同 |
  | Ornith 純 decode，5 組 | 176.2–176.4 tok/s | 176.2–176.9 |
  | 27B MTP，4 組 | 116.11–116.63 | 116.27–116.61 |
  | 每次執行重啟 server，MTP 開啟 | 115.96–116.13 | 115.95–116.36 |
  | 8060S 27B 純 decode，3 組，每次前閒置 60 s | 13.47–13.49 | 13.48–13.49 |

  Windows 上不存在雙峰現象，該變數也沒有可量測的效果。不要從 Linux 指南盲目照抄 ROCm 環境變數。

## <a id="build"></a>2. 建置與載入裝置端程式碼

### 2.1 每個架構一個 code object，內嵌進執行檔

裝置端程式碼與主機端程式碼分開編譯：

```
clang -x hip --offload-arch=gfx1201 --cuda-device-only --no-gpu-bundle-output -O3 kernels.hip -o kernels_gfx1201.co
```

（使用 HIP SDK 的 `clang.exe`；每個架構呼叫一次。）`.co` 檔被轉成位元組陣列並連結進執行檔；執行時由 `hipModuleLoadData` 載入對應所選裝置的那一份。在 Windows 上的好處：不需要 fat-binary/bundle 工具，exe 旁邊不用安裝任何東西，且主機端程式碼可以用 MSVC 建置，裝置端程式碼則用 SDK 的 clang。

### 2.2 選用 kernel 與黏著的 last error

當主機端探測一個在已載入 code object 中不存在的 kernel（例如在 gfx1151 上探測僅限 RDNA 4 的 kernel）時，`hipModuleGetFunction` 會失敗，**而且留下一個錯誤，之後的 `hipGetLastError` 會回傳它**。在某個不相干的呼叫之後檢查 last error 的程式碼，就會回報假的失敗。WHIRL 的 `getFunctionOpt` 對缺少的 kernel 回傳 null 並清除 last error；必要的 kernel 使用 `getFunction`，失敗時會明確報錯。

### 2.3 `hipModuleLaunchKernel` 不檢查引數

Module launch 傳入的是指向引數值的指標陣列。沒有任何東西檢查數量或型別。某個呼叫點少傳了一個引數（後來才加到 kernel 簽章的 KV 基底指標），導致 kernel 讀到垃圾指標，並在兩張 GPU 上都以 `0xC0000005`（存取違規）讓行程崩潰——而不是 HIP 錯誤。規則：由一份 kernel 簽章與主機端共用的定義產生 launch 引數包，並在變更簽章後測試每一條 launch 路徑（包括只有基準測試才走的路徑）。

### 2.4 資源使用報告必須使用完全相同的建置旗標

`-Rpass-analysis=kernel-resource-usage` 會回報每個 kernel 的 VGPR、SGPR、scratch 與溢出。請用與實際建置**完全相同**的旗標執行。旗標缺漏時，我們看到假的「Dynamic Stack」、符號化的 VGPR 數量，以及實際建置中並不存在的溢出。另一個可靠來源是 code object 的 metadata（每個 kernel 的 VGPR/SGPR/scratch），我們從建好的 `.co` 讀取它，以確認重構沒有改變程式碼產生結果。

### 2.5 Kernel 引數大小

Kernel 引數上限為 4 KB。我們分段的 DeltaNet kernel 以傳值方式傳遞每段的狀態指標；16 段時該 struct 約 2.2 KB。以傳值方式傳遞小型表格（token id、分段描述子、KV 頁表引數）是刻意的設計選擇——原因見下一節。

## <a id="timing"></a>3. 沒有效能分析工具時的計時與 profile

- **在 null stream 上的 `hipEvent` 計時偏低。**我們最早的 kernel 計時來自記錄在 null stream 上的 event，結果一直偏小。請在同步過的區段前後使用主機端牆鐘時間，或在明確指定的 stream 上使用 event，且兩個 event 之間要有足夠的工作量。
- **每個 op 一個 event 會扭曲被量測的對象。**每個運算一個 event 的 profiler 模式讓 decode 從 34 掉到 14 tok/s，並讓 server 的一個 decode 週期從 44 ms 拉長到 74 ms；它還把 MTP 草擬時間算到 embedding op 頭上。每週期三個 event（MTP 開始、驗證開始、驗證結束）的粗略模式，在 MoE 模型上仍有約 2.5% 的成本。profile 只用來找出*比例*，絕不用來回報速度，做 A/B 執行時要關掉。
- **剔除法（knockout）。**對 decode 而言，event 扭曲太大，因此我們建構跳過某一類 kernel 的變體並量測差值（例如，一次 41.65 ms 的 16 列驗證，拿掉 DeltaNet step + snapshot 少 4.6 ms，拿掉 attention 少 1.2 ms，拿掉 RMSNorm+量化少 1.0 ms）。
- **獨立探測程式。**Kernel 實驗在一個小型主機程式中執行：它載入 code object，使用從 GGUF 抽出的真實權重，將各變體交錯執行 6–12 輪，回報最小/中位數/最大值，並把每個變體的輸出與正式版 kernel 逐位元組比對。
- **在探測程式中避開快取。**R9700 有 64 MB 的 Infinity Cache（8060S 則有 32 MB 的 MALL）。重複讀取同一個權重矩陣的探測程式，量到的是快取頻寬。請輪流使用總量超過 512 MB 的多份副本（我們在 R9700 上用 ≥ 512 MB，在 8060S 上用 > 2 GB）。
- **工作管理員的「Compute」圖不會顯示 HIP 工作。**GPU 滿載時它仍是平的。基準測試期間 VRAM 圖的鋸齒狀，是每個 prompt 都重新載入模型造成的，不是洩漏。請用 AMD Software 的使用率與功耗讀數，或看 tok/s。

## <a id="memory"></a>4. 記憶體

### <a id="big-alloc"></a>4.1 即使記憶體充足，單一巨大配置仍被拒絕 **[8060S]**

在 8060S（HIP SDK 7.2）上，PAL 拒絕了**一次**約 56 GB 的配置，而當時回報的可用記憶體有 110 GB。總容量從來不是問題：六個行程中的六個 15.3 GB 模型合計達到 90.46 GB。特徵如下：

```
Failed PAL memory allocation!
PAL failed to submit CMD! result:-5
ggml-cuda.cu:107: ROCm error
```

在 llama.cpp 中有兩件事會掩蓋原因：它的 logger 是非同步佇列，因此 `GGML_ABORT` 可能丟掉指出失敗呼叫的那幾行（`AMD_LOG_LEVEL=1` 會把 HIP 的錯誤直接印到 stderr，abort 後依然留存）；而且 HIP 後端不回報最大緩衝區大小，因此配置器會把整個模型塞進一個緩衝區。**修正：**限制單一緩衝區大小，讓載入器把它們切開（我們在本地用了每個緩衝區 8192 MiB 的上限）。這台機器上先前的四個診斷（BIOS 保留記憶體、記憶體故障、僅限 ROCm 的 bug、KV 量化）都是錯的，因為它們都假設是容量問題。WHIRL 以 tensor 群組為單位配置權重，從不要求一整塊巨大的記憶體。

### <a id="mmap"></a>4.2 從記憶體映射檔案載入會失敗

llama.cpp 的 ROCm 版本預設以 mmap 載入；在這個堆疊上，載入期間會以 `ROCm error: unspecified launch failure` 崩潰，而 `AMD_LOG_LEVEL=1` 顯示 PAL `result: -28`。因應方式：舊版本（b10686）用 `--no-mmap`；從 b11214 起該旗標被移除，改為 `--load-mode none`（`-lm none`），傳入舊旗標會讓 `llama-server` 無法啟動。WHIRL 讀取檔案後以明確的複製上傳，因此從未遇到這個問題。（WHIRL 仍在**主機端**對 GGUF 做記憶體映射以便解析——這沒問題；失敗出在把映射的頁面當作 GPU 上傳來源。）

### <a id="vmm"></a>4.3 HIP 虛擬記憶體管理在 WDDM 上無法使用

我們想要每個序列都有連續的虛擬 KV 範圍，並視需要映射實體頁面（`hipMemAddressReserve` / `hipMemCreate` / `hipMemMap`），這樣 attention kernel 就不需要頁表。在兩張 GPU 上的探測結果：

- runtime 回報支援 VMM，粒度 64 KiB；
- 對映射範圍做 `hipMemcpy` 正常；
- **kernel 只看得到映射進該範圍的第一個實體配置。**寫入由第二個實體 handle 支撐之頁面的資料會遺失，部分執行中 GPU 還會發生 fault。

我們的解讀：WDDM 的駐留機制只讓 kernel 引數所參照的那個配置常駐。在 R9700 上執行這個探測時，同一張卡上正在計時的基準測試行程也跟著 fault——絕對不要在正被量測的 GPU 上執行探測程式。我們改為實作真正的頁表（256 token 一頁；[kv-and-caching.md](kv-and-caching.md)）。查表成本為 decode 時間的 0.3–0.5%。

### <a id="wddm-demote"></a>4.4 同一張 GPU 上兩個大型行程：WDDM 把兩者都降級到共享記憶體

有兩次，在 R9700 上已有一個模型行程執行時，又啟動了第二個模型行程（各約 15+ GiB）。WDDM 沒有讓任何一個配置失敗；它把**兩個**行程的部分記憶體都移到共享系統記憶體（其中一個顯示 11.6 GiB 中有 7.2 GiB 為共享），兩者都慢到幾乎停擺約 20 分鐘，直到被終止。那段期間的每一筆量測都無效。我們現在採用的分層防護：

1. **執行檔中的具名 mutex。**選定裝置後，WHIRL 會取得 `Local\whirl-gpu-<device index>`。行程結束或崩潰時 Windows 會釋放它。第二個實例會等待（每 30 s 印一則訊息）直到逾時（預設 1800 s），然後失敗。有明確的覆寫選項供刻意共用時使用；量測時不要用它。
2. **每個啟動腳本中的檔案鎖。**mutex 只存在於新的執行檔中，而第二次事件牽涉的是舊執行檔。所有腳本都以共享模式 0（`FileShare.None`）開啟一個鎖定檔，並在整個生命週期內持有；第二個啟動器會等待。取得鎖之後，啟動器還會等到沒有任何引擎行程存活（殘留行程），最多 10 分鐘。子行程會繼承一個環境標記，避免已上鎖的腳本樹自己卡死自己。PowerShell 與 Python 啟動器彼此互鎖（雙向都測過）。
3. **啟動下一個背景工作前，確認上一個已經結束。**兩次事件都源自只讀了部分日誌就假設背景工作已完成。
4. **只終止你自己記錄下來的 PID。**曾經依行程名稱終止等待中的行程，結果把另一位使用者不相干的等待工作一併砍掉。

### <a id="shared-usage"></a>4.5 Pinned host 記憶體會被回報為「Shared Usage」

Windows 提供每個行程、每個介面卡的 GPU 記憶體計數器（Dedicated Usage、Shared Usage）。我們每 2 s 對每個引擎行程取樣一次，並拒絕任何共享使用量超過 256 MiB 的量測（閒置基準約 89 MiB），因為共享使用量通常代表 VRAM 溢到系統記憶體，會讓速度數字失效。

但 **`hipHostMalloc`（pinned）記憶體會被算進 Shared Usage**，即使並沒有任何溢出。把 token embedding（0.67 GiB）或前綴檢查點（1.56 GiB）放進 pinned host 記憶體（鎖頁主機記憶體）的實驗，讓共享使用量剛好增加那些量，而專用使用量維持在 31.28–31.31 GiB。我們的 RAM KV 層（預設 pinned 9 GiB）也一樣。現在我們讓 server 在啟動時把它的 pinned 配置大小寫入一個宣告檔，監控程式在套用 256 MiB 規則前先扣掉宣告的大小。觀察結果：共享 = 宣告的層大小（8192 或 9216 MiB）+ 35–55 MiB。

同樣的效應也出現在 llama.cpp：沒設定 `GGML_CUDA_NO_PINNED=1` 時，它在 CPU 端的 token embedding／主機運算緩衝區是 pinned 的，會增加約 1 GB 的共享使用量。

### 4.6 Pinned 配置很慢；只配置一次

配置 16 × 512 MiB 的 pinned 記憶體花了 1.7–1.8 s。CPU 讀寫 pinned 記憶體的速度為 32.7 GB/s。WHIRL 在 server 啟動時一次配置好 pinned 區域（啟動時間約 +1.6 s），並以 2 MiB 區塊管理；請求路徑上不會做任何 pin 或 unpin。

### <a id="zero-copy"></a>4.7 Kernel 對 pinned 記憶體的零複製存取無效 **[eGPU]**

直接寫入 `hipHostMalloc` 記憶體（零複製）的 kernel，在這台機器上資料從未真正落到主機記憶體中。因此 WHIRL 所有主機↔裝置的傳輸都在非阻塞 stream 上使用 `hipMemcpyAsync`。我們尚未在直接 PCIe 連接的 R9700 上驗證這點；在那種環境下應視為未驗證，而不是壞掉。

### 4.8 `hipMemGetInfo` 看不到 WDDM 計入的所有東西

- 建立一個非阻塞 stream 與 pinned 區域，讓 `hipMemGetInfo` 的可用記憶體減少 12.8 MiB，但 WDDM 的專用計數器增加了約 60 MiB。請在建立所有 stream 與輔助緩衝區*之後*才決定 VRAM 池的大小，並保留餘裕。
- MoE 模型在 KV 池已被設成「所有剩餘記憶體」之後，又延遲配置了約 0.5 GiB，把行程推進共享記憶體（368–399 MiB 共享）。我們現在為每個模型保留餘裕：dense 模型 768 MiB，MoE 模型 1.5 GiB。（延遲配置的來源尚未查明。）
- Code object 也會計入：code object 大 1.8 MB，就少掉一個 256 token 的 KV 頁。

### 4.9 監控中的 GPU 識別碼在重新開機後會改變

監控程式以介面卡 LUID 作為 GPU 的鍵值。**LUID 在重新開機後會改變。**我們的報告腳本仍以舊 LUID 過濾，結果找到零筆樣本，卻印出「all runs in dedicated VRAM」。現在它在沒看到樣本時會失敗，並印出它實際看到的 LUID。規則：監控檢查在輸入為空時必須失敗。

## <a id="streams"></a>5. Stream、複製與同步

- **同步的 `hipMemcpy` 會清空佇列。**在每個 MTP 草稿步驟前用 `hipMemcpy` 上傳幾個 token id，會等待所有已排入佇列的 GPU 工作，使 GPU 每個週期閒置四次。改把 id 當作 kernel 引數傳入（一個小型傳值 struct）而不複製，端到端提升 +5%。規則：每步的小資料放進 kernel 引數；裝置端自行寫入下一步需要的結果（argmax 把下一個 token 與位置寫進裝置記憶體）。
- **`hipDeviceSynchronize` 會等待每一個 stream，包括背景複製。**非阻塞 stream 上的複製不會阻塞 `hipStreamSynchronize(main)`（0.00 ms），也不會阻塞小型的 null-stream `hipMemcpy`（排入 6000 個複製時為 0.57 ms），但 `hipDeviceSynchronize` 會等它們全部完成。WHIRL 有了背景層 stream 之後，server 中每一次全裝置同步都變成停頓，因此全部改為 stream 同步。
- **HIP graph 在 Windows 上毫無助益。**擷取 decode 步驟：早期為 28.87 vs 28.81 ms/token，後來為 27.81 vs 27.79 ms。主機端排入佇列（每步約 600–740 次 launch，耗時 0.4–0.7 ms）本來就被 GPU 執行時間遮蓋。graph 路徑保留為選項。
- **Kernel 啟動（launch）開銷確實存在但很小。**每次 launch 有約 2.8 µs 的頭尾開銷（純讀取 kernel 在 50 MB 矩陣上達到 605 GB/s，但在 1 GB 的 output head 上達到 626 GB/s）。以每 token 約 740 次 launch 計，值得把相同輸入的矩陣融合成一次 launch（[kernels.md](kernels.md#grouped)），但不值得用 graph。

## <a id="limits"></a>6. Launch 限制

- **Grid 的 y 與 z 上限為 65,536 個 block。**把 248,320 列 output head 放在 y 軸的 kernel 回傳了 `HipFailed`。請把大的維度放在 x。
- **每個 workgroup 的 LDS 超過約 41 KB 時，佔用率（occupancy）從每個 SIMD 16 個 wave 降到 6 個**，這發生在一個 multi-query attention 變體上，而且它變得更慢。gfx12 上每個 workgroup 的 LDS 為 64 KB，但全部用掉會犧牲並行度。

## <a id="devices"></a>7. 在雙 GPU 機器上選對 GPU

| Runtime | 在這台機器上的順序 |
|---|---|
| HIP / ROCm（llama.cpp ROCm） | `ROCm0` = 8060S，`ROCm1` = R9700 |
| HIP 搭配 `HIP_VISIBLE_DEVICES=1` | R9700 變成 `ROCm0` |
| Vulkan（llama.cpp Vulkan） | `Vulkan0` = 8060S，`Vulkan1` = R9700 |

每個工具都必須明確選擇 GPU；預設會落在內顯上。WHIRL 依名稱／架構選擇（每個版本都有預設裝置；`WHIRL_DEVICE` 可覆寫）。注意 Vulkan 對 RDNA 回報的 warp size 是 64，而 RDNA 上的 HIP wave 是 32 個 lane；WHIRL 所有的 reduction 與 shuffle 都是為 wave32 撰寫的。

**[8060S] 兩張 GPU 並非彼此獨立。**8060S 與 CPU 共用電源與散熱。在 R9700 上的建置或基準測試（會加重 CPU 負載）在某次執行中讓 8060S 的 decode 慢了 11–18%。在 8060S 上計時時，絕對不要同時執行其他任何東西。

## <a id="egpu"></a>8. eGPU（USB4）的影響 **[eGPU]**

這些是開發機的環境限制。我們量測它們，是為了避免有人誤以為那是引擎本身的行為，而我們並未針對它們做最佳化。

| 量測項目 | 結果 |
|---|---|
| 主機連結，`hipMemcpyAsync`，任何 16 KiB–256 MiB 的區塊大小 | D2H 3.78–3.81 GB/s，H2D 3.84–3.86 GB/s |
| 6018 個分散的 KV 頁區塊 | 3.8 GB/s，每次複製排入佇列 0.45 µs |
| H2D 暫存到 VRAM + D2D 分散寫入 vs 直接逐塊複製 | 3.51 vs 3.69–3.83 GB/s（暫存較慢） |
| H2D 佔滿連結時的 kernel 派送 | 1500 個小 kernel：15 → 64 ms；20 個大 kernel：+4%；小型同步複製 0.11 → 0.25 ms |
| `GPU_BLIT_ENGINE_TYPE` | 無效果 |
| `PAL_DISABLE_SDMA=1` | 成本轉移到別處；複製變慢 |
| 對服務的影響 | KV 還原執行的 0.5–2.0 s 期間，其他 slot 的 decode 週期 43.0 → 50.4 ms（+17%） |

派送變慢的程度與傳輸的位元組數成正比；調節複製節奏或改變區塊大小都無法降低它。在直接 PCIe x16 連結上，我們預期既不會有低頻寬也不會有派送干擾，但尚未量測。

## <a id="storage"></a>9. SSD 層的儲存 I/O

| NVMe 存取 | 吞吐量 |
|---|---|
| 有緩衝寫入 | 1.8 GB/s |
| 無緩衝寫入／讀取 | 5.1–5.3 / 5.1 GB/s |
| 無緩衝讀取，2 MiB，佇列深度 1 | 4.1 GB/s |
| 無緩衝讀取，8 MiB，QD1 | 5.06 GB/s |
| 無緩衝讀取，2 MiB，QD4（4 個重疊讀取，依序完成） | 7.0 GB/s |

使用 `FILE_FLAG_NO_BUFFERING`（對齊的緩衝區），並讓多個重疊讀取同時在進行中。

## <a id="shell"></a>10. Windows 上的 shell 與工具陷阱

| 陷阱 | 症狀 | 規則 |
|---|---|---|
| Windows PowerShell 5.1 把沒有 BOM 的 UTF-8 檔案當成 ANSI 字碼頁讀取 | 含中文文字的 `.ps1` 解析失敗 | 用 PowerShell 7（`pwsh`）執行腳本 |
| 主控台字碼頁 cp950（zh-TW） | Python 測試框架在印出 ☔ 或 é 時以 `UnicodeEncodeError` 終止 | 在每個啟動器中設定 `PYTHONIOENCODING=utf-8` |
| Bash heredoc 與內嵌字串 | 反斜線被吃掉；`\x..` 與 `\u....` 被改寫；產生的原始碼無法編譯 | 把修補／編輯腳本寫成檔案再執行 |
| 程式碼產生器中非 raw 的 Python 三引號字串 | `\n` 在產生的字串常值中變成真正的換行；raw 字串則讓反斜線加倍 | 從檔案產生原始碼，或測試產生器的輸出 |
| 對 CRLF 檔案使用 Git Bash 的 `sed -i` | 整個檔案被改寫成 LF 換行 | 用會保留換行格式的工具編輯 CRLF 檔案 |
| 從 Python 呼叫 `bash` | 執行的是 WSL 的 `bash.exe`，不是 Git Bash | 絕不從 Python 呼叫 `bash`；改用純 Python 或明確路徑 |
| Git Bash 的 `$!` | 是 MSYS PID；`taskkill` 無法使用 | 用 `Start-Process -PassThru` 啟動背景行程以取得 Windows PID |
| `Invoke-WebRequest` 的錯誤不會終止腳本 | 下載失敗後，腳本接著執行「清空目標資料夾」步驟，把可用的安裝清空 | 動到目的地之前先驗證壓縮檔 |
| GitHub release 下載 | 130–176 KB/s 且連線中斷 | 使用 `curl -C -` 並重試 |
| 過時的輸出檔案 | 執行檔啟動失敗後，檢查腳本比對了上一次執行留下的輸出，並印出「ok」 | 每次執行前刪除預期的輸出檔；檢查結束碼 |
| 在共用機器上長時間建置 | 干擾互動工作與 GPU 計時 | 以低優先權建置（`start /low`、idle 優先權類別），且計時期間絕不建置 |

## 11. 檢查清單

1. 除錯任何東西之前先設定 `AMD_LOG_LEVEL=1`。
2. 明確選擇 GPU；啟動時印出其名稱與架構。
3. 每張 GPU 只跑一個模型行程——同時用具名 mutex *和*檔案鎖強制執行。
4. 監看每個行程的 Dedicated/Shared Usage；扣除宣告的 pinned 記憶體；資料為空時要失敗。
5. 不用 mmap 支撐的 GPU 上傳；不做單一巨大配置。
6. 熱路徑上不做同步複製或全裝置同步；小資料放進 kernel 引數。
7. 在同步過的區段前後用牆鐘計時；探測程式中輪流使用 > 512 MB 的資料。
8. 把 eGPU 主機連結的影響視為環境限制；針對它們最佳化之前先在 PCIe 上確認。

所有 Windows/HIP 的坑，連同症狀 → 原因 → 修正 → gate，也都列在 [pitfalls.md](pitfalls.md#hip)。
