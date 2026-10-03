[English](../en/pitfalls.md) | **繁體中文**

# 踩坑紀錄：所有咬過我們的坑，以及現在怎麼抓

**對誰有幫助：** Windows 上的 HIP 開發者；RDNA 3.5 / RDNA 4 kernel 作者；llama.cpp / ggml
貢獻者（tokenizer、ROCm/HIP 後端、MTP、量化）；為混合注意力 + 遞迴模型打造推論伺服器（server）的人；
任何在筆電或 eGPU 上對 AMD GPU 做基準測試的人。

這些陷阱有很多在別處都沒有文件記載，因為很少人會在 Windows 上原生執行從零寫起的 HIP
引擎。每一條都會列出我們遇到它的**條件**（GPU、HIP SDK 7.2、Windows 11 build 26200，必要時註明
eGPU 或直連 PCIe）、附上確切錯誤字串的**症狀**、**根本原因**與證明它的證據、以通用規則表述的
**修正**，以及**關卡（gate）**——也就是現在會自動抓到它的檢查。

慣例：R9700 = Radeon AI PRO R9700（gfx1201，經 USB4 連接）。8060S = Radeon 8060S
（gfx1151，ASUS ROG Flow Z13 裡的 Ryzen AI Max+ 395）。數字來自研究原型。
分類：[Windows 上的 HIP](#hip) · [工具鏈與 shell](#tool) · [Kernel 與數值](#kern) ·
[推測解碼的精確性](#mtp) · [Server 與快取](#srv) · [Tokenizer 與
樣板](#tok) · [模型檔案與量化](#quant) · [視覺](#vis) · [量測](#meas) ·
[eGPU](#egpu)。

## <a id="lessons"></a>可通用的教訓

1. **在 Windows/PAL 上，「記憶體不足」可能意味著「單一配置太大」。** 形成任何理論之前，先設定
   `AMD_LOG_LEVEL=1`；在正確診斷之前，我們先做了四個錯誤診斷（[HIP-2](#hip-2)）。
2. **逐位元相同是設計性質，不是測試結果。** 批次與非批次路徑必須共用相同的算術（相同的逐單位
   運算式、相同的歸約樹、明確的 fma）。只要這點成立，「推測解碼輸出 == 一般輸出」就變成一行就能
   寫完的 gate，能抓到整類的 bug（[MTP-1](#mtp-1)、[KERN-1](#kern-1)）。
3. **測試預設路徑。** 每個 gate 都固定了某個環境變數，因此一個只在*什麼都沒設定*時才會出現的
   bug 曾經被發布出去（[SRV-6](#srv-defaultenv)）。保留一個零設定的測試。
4. **檢查遇到空輸入必須失敗。** 一個沒找到任何樣本的記憶體監控回報了成功；一個比對過期檔案的
   腳本回報了「ok」（[HIP-21](#hip-21)、[TOOL-11](#tool-11)）。
5. **最佳化之前先量上限。** 純讀取或只用暫存器的探測顯示，大多數格式早已達到頻寬的 97–100%，
   並指出了沒達到的那兩個（[KERN-13](#kern-13)）；一個記憶體存取模式的探測則顯示，某個「記憶體
   問題」其實是指令排程問題。
6. **在 RDNA 上，暫存器決定速度。** 每次修改 kernel 都要讀 VGPR/scratch/溢出；我們許多最大的
   收益，都是移除編譯器因為與數學無關的原因而引入的溢出（[KERN-4](#kern-4)…[KERN-12](#kern-12)）。
7. **絕不要在不同輸出之間比較推測解碼速度。** 內容會改變接受率；要比較每個循環的 ms、交錯 A/B、
   回報最小–最大值，並保留小但可重現的收益（[MEAS-2](#meas-2)、[MEAS-7](#meas-7)）。
8. **遞迴狀態會改變快取方式。** DeltaNet/Mamba 式的狀態無法像 KV 那樣截斷；重用需要檢查點，
   而且不論冷啟動或續跑，檢查點邊界都必須是 prefill（提示詞預填）排程的一部分
   （[SRV-1](#srv-1)、[SRV-12](#srv-12)）。
9. **Tokenizer bug 就是效能 bug。** 一個空白預切分的 bug 讓提示詞變得超出分布、弄壞了 n-gram
   草擬、降低了前綴快取命中——而每個輸出看起來都還是正常的（[TOK-1](#tok-1)）。
10. **任何活得比請求還久的東西，都必須在請求邊界重設、重新初始化或歸還：** 計時表
    （[MTP-4](#mtp-4)）、帶填充的借用 scratch（[VIS-1](#vis-1)）、KV 頁（[SRV-5](#srv-5)）、
    層項目連結（[SRV-9](#srv-9)）。
11. **WDDM 不會讓第二個大型行程失敗——它會把兩個都降級。** 用具名 mutex *加上* launcher 檔案鎖
    強制每張 GPU 只跑一個模型行程（[HIP-5](#hip-5)）。
12. **標註環境限制；不要把它們最佳化掉。** eGPU 連結效應與筆電散熱都經過量測並記錄下來，
    而不是設法繞過（[EGPU-1](#egpu-1)、[MEAS-3](#meas-3)）。
13. **Linux ROCm 的建議無法套用到 Windows**（[HIP-17](#hip-17)），而且參考引擎也可能是壞的——
    比較之前先檢查參考（[MEAS-10](#meas-10)）。

---

## <a id="hip"></a>1. Windows 上的 HIP

### <a id="hip-1"></a>HIP-1 · Linux 的工具與錯誤訊息都不存在
- **條件：** Windows 上的任何 HIP 程式（HIP SDK 7.2、Adrenalin `amdhip64_7.dll` 10.0.3679.0）。
- **症狀：** `rocprof`、`rocminfo`、`omniperf` 都不存在；失敗只會以泛用代碼呈現
  （`hipErrorLaunchFailure`、llama.cpp 的 `ROCm error: unspecified launch failure`、我們的 `HipFailed`）。
- **根本原因：** Windows 上的 HIP 是在 WDDM 之下透過 AMD 的 PAL 執行，而不是 ROCr/KFD。
- **修正：** 以 `AMD_LOG_LEVEL=1` 執行；runtime 就會把自己的錯誤（會引用
  `palvirtual.cpp`）印到 stderr。自己建立計時機制（[windows-hip.md](windows-hip.md#timing)）。
- **現在怎麼抓：** 每個 launcher 都能設定 `AMD_LOG_LEVEL=1`；無法解釋的失敗一律先用它重跑，
  再做其他事。

### <a id="hip-2"></a>HIP-2 · 記憶體明明有空，單一巨大配置卻被拒絕
- **條件：** 8060S、統一記憶體、HIP SDK 7.2，llama.cpp ROCm 載入約 56 GB 的模型，
  回報可用 110 GB。
- **症狀：** `Failed PAL memory allocation!`，接著 `PAL failed to submit CMD! result:-5`，再接著
  `ggml-cuda.cu:107: ROCm error`，沒有任何原因說明。
- **根本原因：** PAL 拒絕該大小的單一配置；總容量沒問題（六個行程各載入一個 15.3 GB 模型，
  達到 90.46 GB）。llama.cpp 的 HIP 後端沒有回報最大緩衝區大小，因此整個模型被放進同一個緩衝區；
  而當 `GGML_ABORT` 觸發時，它的非同步記錄器可能會丟掉指出失敗呼叫的那幾行。
- **修正：** 限制單一緩衝區大小，讓載入器自行切分（8192 MiB 可行）。在自己的程式碼中，
  依張量群組分別配置。先前四個理論（BIOS 保留區、RAM 故障、只在 ROCm 出現的 bug、KV
  量化）都錯了，因為它們都假設這是容量問題。
- **現在怎麼抓：** 載入失敗時使用 `AMD_LOG_LEVEL=1`；WHIRL 從不要求單一巨大區塊。

### <a id="hip-3"></a>HIP-3 · 從記憶體映射檔案載入會當掉（llama.cpp）
- **條件：** llama.cpp ROCm b10686 → b11214、R9700。
- **症狀：** 載入時當掉，訊息為 `ROCm error: unspecified launch failure`；`AMD_LOG_LEVEL=1` 顯示
  PAL `result: -28`。
- **根本原因：** 在這套軟體堆疊上，用 mmap 支撐的頁面作為上傳來源會失敗。
- **修正：** `--no-mmap`（b10686）；在 b11214 中這個旗標被**移除**，改由 `--load-mode none`
  （`-lm none`）取代——舊旗標會讓 `llama-server` 拒絕啟動。在自己的引擎中，讀取檔案並以明確的
  複製上傳（在主機端用 mmap 做解析沒問題）。
- **現在怎麼抓：** 所有 llama.cpp 對照腳本都傳入 `--load-mode none`。

### <a id="hip-4"></a>HIP-4 · HIP 虛擬記憶體：kernel 只看得到第一個實體配置
- **條件：** 兩張 GPU、WDDM、`hipMemAddressReserve` / `hipMemCreate` / `hipMemMap`。
- **症狀：** VMM 回報為支援（粒度 64 KiB），`hipMemcpy` 寫入映射範圍也正常，
  但 kernel 寫入由第二個實體 handle 支撐的頁面時資料會遺失；有些執行還讓 GPU 發生 fault。
- **根本原因（我們的判讀）：** WDDM 的駐留機制只會讓 kernel 參數所引用的那個配置駐留。
- **修正：** 在 Windows 上不要依賴 HIP VMM；改用明確的頁表（WHIRL：256-token 的 KV 頁、
  每個 key tile 查一次表，decode（逐 token 生成）成本 0.3–0.5%）。
- **現在怎麼抓：** 分頁路徑經驗證與連續快取逐位元相同。

### <a id="hip-5"></a>HIP-5 · 一張 GPU 上跑兩個大型行程：WDDM 把兩個都降級到共享記憶體
- **條件：** R9700 32 GB，兩個各 15+ GiB 的模型行程（發生過兩次：一次是在 needle 測試期間啟動了
  測試 server，一次是在先前的基準測試還在跑時啟動了檢查）。
- **症狀：** 沒有配置失敗；兩個行程都變得極慢，持續約 20 分鐘；逐行程計數器顯示大量的
  共享使用量（Shared Usage）（其中一個：11.6 GiB 中有 7.2 GiB 是共享）。那段期間的所有數字都
  無效。
- **根本原因：** WDDM 超額分配 VRAM，並把兩個行程的部分內容分頁到系統記憶體。
- **修正：** (1) 在執行檔中使用具名 mutex `Local\whirl-gpu-<device>`，結束或當掉時由 Windows
  釋放；第二個實例會等待，每 30 s 印一次訊息，逾時後放棄。(2) 在每個 launcher 中使用檔案鎖
  （share mode 0），因為舊的執行檔沒有該 mutex；取得鎖之後，還要等到沒有任何引擎行程殘留。
  (3) 啟動下一個背景工作之前，確認前一個已經結束。(4) 只砍你自己記錄下來的 PID。
- **現在怎麼抓：** 記憶體監控會標記同一個 adapter 上任何兩個重疊的引擎行程。

### <a id="hip-6"></a>HIP-6 · pinned host 記憶體會被算進「Shared Usage」
- **條件：** Windows 的逐行程 GPU 記憶體計數器；`hipHostMalloc`。
- **症狀：** 對於把 token embedding 或檢查點放在 pinned host 記憶體（鎖頁主機記憶體）的實驗，
  我們的 VRAM 溢位警報觸發了（共享 1,916 / 771 / 1,234 MiB）；專用 VRAM 使用量沒有變化
  （31.28–31.31 GiB）。
- **根本原因：** pinned 配置會被計入 GPU 的共享區段。
- **修正：** 宣告 pinned 大小（server 會把它們寫進檔案），並在套用 256 MiB 規則之前先扣掉。
  在 llama.cpp 中設定 `GGML_CUDA_NO_PINNED=1`，讓約 1 GB 的主機緩衝區不計入共享使用量。
- **現在怎麼抓：** 對於啟用分層的 server，監控回報 共享 = 宣告的層大小 + 35–55 MiB。

### <a id="hip-7"></a>HIP-7 · 在 null stream 上用 `hipEvent` 計時會偏低
- **條件：** 早期在 null stream 上用 event 做 kernel 計時。
- **症狀：** kernel 時間一直偏小。
- **修正：** 在同步過的區段外圍用主機牆鐘時間，或在明確的 stream 上使用 event，且兩個 event
  之間要有足夠的工作量。
- **現在怎麼抓：** 速度宣稱只來自牆鐘時間的端到端執行。

### <a id="hip-8"></a>HIP-8 · 每個 op 一個 event 的 profile 會破壞它要量的東西
- **症狀：** 開啟效能分析工具時 decode 從 34 → 14 tok/s；一個 server decode 循環從 44 → 74 ms；
  MTP 草擬時間被歸到 embedding op 上。即使每個循環只用 3 個 event，在 MoE 模型上也要付出約 2.5%。
- **修正：** profile 只用來看 prefill 中的比例；decode 改用剔除式建置（knockout build）；
  A/B 期間絕不 profile。
- **現在怎麼抓：** A/B 腳本會拒絕 profile 旗標。

### <a id="hip-9"></a>HIP-9 · 工作管理員看不到任何運算活動
- **症狀：** GPU 已經滿載，「Compute」圖卻是平的；VRAM 的鋸齒狀變化看起來像記憶體洩漏。
- **根本原因：** HIP 工作不會顯示在那裡；鋸齒是模型在每個提示詞都重新載入造成的。
- **修正：** 讀 AMD Software 的使用率/功耗，或看 tok/s。

### <a id="hip-10"></a>HIP-10 · 一次同步 `hipMemcpy` 會清空整個佇列
- **條件：** 在每個 MTP 草擬步驟之前上傳幾個 token id。
- **症狀：** 每個循環 GPU 閒置四次。
- **根本原因：** 同步的 H2D 複製會等待所有已排入佇列的工作。
- **修正：** 小量的逐步驟資料以 kernel 參數傳入（以值傳遞的 struct）；讓 kernel 自己寫出下一步
  需要的東西（argmax 在裝置上寫入下一個 token/位置）。端到端 +5%。
- **現在怎麼抓：** decode 循環恰好只有一次主機同步。

### <a id="hip-11"></a>HIP-11 · `hipDeviceSynchronize` 會等待背景複製
- **條件：** 非阻塞的層 stream 把 KV 複製到主機，同時主 stream 在做 decode。
- **實測：** 背景複製不會阻塞 `hipStreamSynchronize(main)`（0.00 ms）或小量的 null-stream
  `hipMemcpy`（佇列中有 6000 個複製時為 0.57 ms）；`hipDeviceSynchronize` 則會等待所有複製。
- **修正：** 有背景傳輸的 server 中不要做整個裝置的同步；改為同步 stream。

### <a id="hip-12"></a>HIP-12 · 對 pinned 記憶體的零複製寫入永遠不會落地 **[eGPU]**
- **條件：** 經 USB4 連接的 R9700。
- **症狀：** 直接寫入 `hipHostMalloc` 記憶體的 kernel，在主機端沒有產生任何資料。
- **修正：** 在非阻塞 stream 上使用 `hipMemcpyAsync`。未在直連 PCIe 上驗證。

### <a id="hip-13"></a>HIP-13 · Grid 的 y/z 上限 → `HipFailed`
- **症狀：** 一個針對 248,320 列輸出頭的重新量化 kernel 回傳了 `HipFailed`。
- **根本原因：** grid 的 y 與 z 上限為 65,536 個 block；那些列被放在 y 上。
- **修正：** 把大的維度放在 x 上。

### <a id="hip-14"></a>HIP-14 · 探測選用 kernel 會留下黏著的錯誤
- **症狀：** 當程式碼檢查 `hipGetLastError` 時，之後某個不相關的呼叫「失敗」了。
- **根本原因：** 對已載入 code object 中不存在的 kernel 呼叫 `hipModuleGetFunction`，會設定
  last error。
- **修正：** 選用查詢回傳 null 並清除 last error；必要查詢則大聲失敗。
- **現在怎麼抓：** 冒煙測試會以不存在的名稱呼叫選用查詢，並檢查錯誤狀態。

### <a id="hip-15"></a>HIP-15 · Module 啟動不會檢查參數
- **症狀：** 某個基準測試模式讓兩張 GPU 的行程都以 `0xC0000005` 當掉；某個自我檢查印出
  `-nan`。
- **根本原因：** 某個 kernel 多了一個 KV-base 參數後，一個呼叫端少傳了一個參數（kernel 把垃圾
  當成指標讀取）；另外，某個自我檢查為一個 2048 欄的矩陣配置了 MoE 專家寬度（512）的輸入。
- **修正：** 從單一定義產生 launch 參數包；依最大矩陣維度決定測試緩衝區大小；簽章變更後要
  實際跑過只有基準測試會用到的路徑。

### <a id="hip-16"></a>HIP-16 · 除非旗標與建置一致，否則資源使用報告會說謊
- **症狀：** `-Rpass-analysis=kernel-resource-usage` 顯示了「Dynamic Stack」、符號化的 VGPR 數量
  以及溢出，但實際的建置並沒有這些。
- **修正：** 用與正式建置完全相同的旗標執行它，或從建好的 code object 的 metadata 讀取
  VGPR/SGPR/scratch。

### <a id="hip-17"></a>HIP-17 · `GPU_MAX_HW_QUEUES=1` 在 Windows 上毫無作用
- **條件：** 來自 Linux/ROCm 的建議；在 Linux/ROCm 上，這張卡的 decode 步驟會在約 28 ↔ 36 ms
  之間交替。
- **實測：** 每次執行用新的行程、交錯執行、8 組：不論有沒有設定，16 次執行全都是 28.0 ms/token
  （離散 0.2%）；MTP、server 重啟與 8060S 的執行也同樣沒有變化。
- **根本原因：** 這種雙峰現象是 KFD 硬體佇列的問題；Windows 是透過 PAL/WDDM 排程。
- **修正：** 不要未經量測就移植 Linux ROCm 的調校變數。

### <a id="hip-18"></a>HIP-18 · HIP graph 在這裡沒有任何幫助
- **實測：** 28.87 vs 28.81 ms/token；後來 27.81 vs 27.79 ms。主機端排入 600–740 次啟動
  （0.4–0.7 ms）早已被 GPU 執行時間掩蓋。
- **修正：** 保留 graph 路徑為選用；改以融合相同輸入的 GEMV 來減少啟動次數（−1.2…−2.0%）。

### <a id="hip-19"></a>HIP-19 · 雙 GPU 機器中的裝置編號
- **事實：** ROCm/HIP：`ROCm0` = 8060S、`ROCm1` = R9700；設定 `HIP_VISIBLE_DEVICES=1` 時 R9700 會變成
  `ROCm0`。Vulkan：`Vulkan0` = 8060S、`Vulkan1` = R9700。Vulkan 回報 warp size 為 64；RDNA 上的 HIP
  wave 是 32。
- **修正：** 依名稱/架構明確選擇，並在啟動時印出所選的裝置。

### <a id="hip-20"></a>HIP-20 · `hipMemGetInfo` 不等於 WDDM 的記帳；晚到的配置會溢位
- **症狀：** 把 KV 池大小設為「所有剩餘記憶體」後，MoE server 行程顯示 368–399 MiB 的共享；
  建立一個 stream + pinned arena 在 `hipMemGetInfo` 中花了 12.8 MiB，但在 WDDM 的專用計數器中
  約 60 MiB。
- **根本原因：** 池大小決定之後，約 0.5 GiB 被延遲配置（來源未查明）；驅動程式端的配置在
  `hipMemGetInfo` 中看不到。
- **修正：** 在決定池大小之前建立所有 stream 與輔助緩衝區；保留每個模型的預留量（dense
  768 MiB、MoE 1.5 GiB）。
- **現在怎麼抓：** 在記憶體監控下執行最壞情況的 VRAM 測試（4 個並行的長請求填滿池的 95%）。

### <a id="hip-21"></a>HIP-21 · GPU 的 LUID 在重開機後會改變
- **症狀：** 記憶體報告印出「所有執行都在專用 VRAM 中」——但樣本數是零。
- **根本原因：** 它篩選的是舊的 adapter LUID。
- **修正：** 沒有任何樣本符合時就失敗，並印出看到的 LUID。

### <a id="hip-22"></a>HIP-22 · PID 重用造成記憶體假警報
- **症狀：** 一個 server 被標記為共享 9,256 MiB，雖然它宣告了 9,216 MiB 的 pinned 層。
- **根本原因：** 以 PID 命名的宣告檔被之後一個相同 PID 的行程覆寫。
- **修正：** 以 PID + 啟動時間作為鍵；當某個 PID 消失超過 20 s 時，切分監控樣本。

### <a id="hip-23"></a>HIP-23 · pinned 配置很慢
- **實測：** 16 × 512 MiB 的 pinned 配置花了 1.7–1.8 s。
- **修正：** 啟動時配置一個 arena（server 啟動 +1.6 s）並管理 2 MiB 的區塊；絕不在請求路徑上
  做 pin。

### <a id="hip-24"></a><a id="hip-256k"></a>HIP-24 · 256k prefill 期間發生 `HipFailed`，伴隨顯示器重新列舉（未解決）
- **條件：** R9700、27B、q8h KV、262,400-token 的池，一次約 262k token 的 prefill。
- **症狀：** prefill 中途出現 `HipFailed`；同一時間 Windows 記錄了一次 Win32k 顯示器重新列舉；
  下一次載入也失敗了；後來同一個提示詞成功了（prefill 398 tok/s，找到 needle）。
- **根本原因：** 不明（懷疑是 GPU 重設）。
- **緩解：** 超過 128k context 的 prefill 注意力會依 head 範圍切成多次啟動（成本約 2%，
  結果完全相同；≤ 128k 時從不套用）。若再次發生：以 `AMD_LOG_LEVEL=1` 重跑。

### <a id="hip-25"></a>HIP-25 · 在正被量測的 GPU 上跑探測
- **症狀：** 一次基準測試執行沒有產生 decode 那一行；同一時間，同一張卡上的 VMM 探測讓 GPU
  發生了 fault。
- **修正：** GPU 在計時期間不跑任何其他東西——包括「很小的」探測。

### <a id="hip-26"></a>HIP-26 · 兩張 GPU 並不獨立（APU） **[8060S]**
- **症狀：** 某次基準測試期間，8060S 的 27B 一般 decode 下降 11–18%，prefill 下降約 20%。
- **根本原因：** R9700 上的長 context 工作讓 CPU 負載升高，而 CPU 與 8060S 共用電源與散熱。
- **修正：** 只在機器其他部分閒置時對 8060S 計時。經乾淨的交錯 A/B 確認
  （+0.7%，不是退步）。

---

## <a id="tool"></a>2. 工具鏈與 shell

### <a id="tool-1"></a>TOOL-1 · Windows PowerShell 5.1 會誤讀 UTF-8 腳本
- **症狀：** 含有中文字的 `.ps1` 解析失敗。
- **根本原因：** 沒有 BOM 時，5.1 會以 ANSI 字碼頁讀取檔案。
- **修正：** 用 PowerShell 7（`pwsh`）執行腳本，或存檔時加上 BOM。

### <a id="tool-2"></a>TOOL-2 · cp950 主控台會弄死 Python 測試框架
- **症狀：** 測試印出 ☔ 或 é 時出現 `UnicodeEncodeError`；整輪 gate 執行中止。
- **修正：** 在每個 launcher 中設定 `PYTHONIOENCODING=utf-8`。

### <a id="tool-3"></a>TOOL-3 · Shell 的 heredoc 與內嵌字串會改寫跳脫字元
- **症狀：** 含 `\x..`、`\u....` 或 `\\n` 的產生原始碼無法編譯；反斜線消失；
  經過 Bash 傳遞的路徑會失去分隔符（`C:\a\b` → `C:ab`）。
- **修正：** 把修補與產生器腳本寫成檔案再執行；大量使用 Windows 路徑的流程從
  PowerShell 執行。

### <a id="tool-4"></a>TOOL-4 · 使用非 raw Python 字串的程式碼產生器
- **症狀：** 產生的字串常值中的 `\n` 變成真正的換行；在 raw 字串中 `\\` 變成
  兩個反斜線。
- **修正：** 從樣板檔產生；對產生器的輸出做編譯測試。

### <a id="tool-5"></a>TOOL-5 · Git Bash 的 `sed -i` 會把 CRLF 檔案轉成 LF
- **症狀：** 單行編輯改寫了一個 CRLF 原始檔的每一個行尾。
- **修正：** 用會保留行尾的工具編輯（以 `newline=''` 讀寫）。

### <a id="tool-6"></a>TOOL-6 · 從 Python 呼叫的 `bash` 是 WSL 的 bash
- **症狀：** 一個呼叫 `bash` 的 Python 框架，在不同的環境中執行了 WSL 的 `bash.exe`。
- **修正：** 在 Windows 上絕不要從 Python 呼叫 `bash`；把框架用 Python 寫。

### <a id="tool-7"></a>TOOL-7 · Git Bash 中的 `$!` 不是 Windows PID
- **修正：** 用 `Start-Process -PassThru` 啟動背景行程，並記錄 Windows PID。

### <a id="tool-8"></a>TOOL-8 · 非終止性的下載錯誤刪掉了一份安裝
- **症狀：** 一個可正常運作的 llama.cpp 安裝目錄被清空。
- **根本原因：** `Invoke-WebRequest` 失敗了卻沒有讓腳本停止；下一步清空了目標位置。
- **修正：** 動到目的地之前，先驗證壓縮檔（大小/雜湊/解壓測試）。

### <a id="tool-9"></a>TOOL-9 · 緩慢又會斷線的 release 下載
- **實測：** 一個 257 MB 的套件，速度 130–176 KB/s 且會斷線。
- **修正：** 使用帶重試的 `curl -C -`。

### <a id="tool-10"></a>TOOL-10 · 只套用了一半的修補
- **症狀：** kernel 改了，主機端程式碼卻沒改（修補腳本中途失敗）。
- **修正：** 修補腳本在寫入任何東西之前檢查每一個錨點字串，且可以重複執行。

### <a id="tool-11"></a>TOOL-11 · 過期的輸出被當成成功
- **症狀：** 一個數值檢查印出「ok」，但執行檔根本沒有跑（路徑錯誤）。
- **根本原因：** 該檢查讀到的是上一次執行留下的輸出檔。
- **修正：** 每次執行前刪除預期的輸出檔；檢查結束代碼。

### <a id="tool-12"></a>TOOL-12 · 建置失敗卻沒有停止流程
- **症狀：** 發生編譯錯誤（一個變數遮蔽錯誤）之後，一整套背景測試全都跑在先前的執行檔上；
  每個數字都作廢。
- **修正：** 建置失敗就中止；在每個結果檔中記錄並驗證執行檔的雜湊。

### <a id="tool-13"></a>TOOL-13 · 改名後的設定會被舊執行檔默默忽略
- **症狀：** 對較舊候選執行檔的 A/B 執行忽略了新的環境變數名稱。
- **修正：** 每當設定名稱變更，A/B 的基準版本就必須重新建置（或設定別名）。

### <a id="tool-14"></a>TOOL-14 · 原始碼常值中的原始控制位元組
- **症狀：** 一個含有原始 NUL 位元組的字串常值無法編譯。
- **修正：** 在產生的表格中輸出跳脫序列（`\x00`）。

### <a id="tool-15"></a>TOOL-15 · device 編譯途中改了原始檔，留下過期的 code object
- **條件：** Ninja、一次約 2 分鐘的 HIP device 編譯，編譯途中修改 kernel 原始檔。
- **症狀：** 加進 gfx1151 kernel 集的標記 kernel 不在內嵌的 code object 裡：能力探測回報不存在，kernel 測試有數十項
  以荒謬數值失敗（CPU 參考用錯了 scale word 格式）。
- **根本原因：** 編譯器讀到的是舊原始檔；它寫出的輸出比修改時間還新，所以 Ninja 認為已是最新，不會重編。
- **修正：** 建置進行中不要改它會讀的原始檔；改了就再 touch 一次。新 kernel「找不到」時先看 code object 的符號表
  （`llvm-objdump -t`）。
- **現在怎麼抓：** `whirl-kernel-test` 一開始就印出探測到的能力旗標。

---

## <a id="kern"></a>3. Kernel 與數值

### <a id="kern-1"></a>KERN-1 · 重構改變了 fma 縮併，也改變了 logits
- **條件：**融合兩個 decode kernel；gfx1201/gfx1151 用的 clang。
- **症狀：**把 `h0*w0 + h1*w1 + h2*w2 + x*w3` 搬進 inline 函式後，最後一個 token 的 logits
  差了最多 0.17（不是逐位元相同）。
- **根本原因：**編譯器選了不同的 fma 縮併順序。
- **修正：**把標準順序（原本編譯出來的順序）寫成明確的 `__builtin_fmaf` 鏈，或使用
  `#pragma clang fp contract(off)`。
- **現在怎麼抓：**在兩張 GPU 上逐位元比對融合與未融合的輸出（批次、step-1、step-2、1,107 token 的
  prefill）；跨 build 比對端到端 logits。

### <a id="kern-2"></a>KERN-2 · `x*x` 被縮併進歸約；乘法和轉換被合併
- **症狀：**融合後的 RMSNorm／gated-norm 產生器出現 1-ulp 差異。
- **根本原因：**clang 把平方縮併進第一個 butterfly 加法（fma），並把一個乘法和 f16 轉換合併成
  `v_fma_mix`。
- **修正：**在平方與最終乘積之後加 `asm volatile("" : "+v"(v))`；重現參考歸約的 lane 排列。
- **現在怎麼抓：**用探針把融合 kernel 與原版逐位元組比對；在同一個 binary 中比對開／關時的 logits。

### <a id="kern-3"></a>KERN-3 · 遞迴步驟改寫中出現同樣的縮併
- **症狀：**改寫後的 DeltaNet step kernel 不是逐位元相同。
- **根本原因：**`y*y` 被縮併進 shuffle-add 的 fma。
- **修正／偵測：**asm 屏障；逐位元比對 16 列 teacher-forced 的 logits 傾印。

### <a id="kern-4"></a>KERN-4 · 更快的反量化反而讓 prefill 慢很多
- **症狀：**最常用的 tile 配置每個 lane 溢出 552 bytes 到 scratch。
- **修正：**把每個候選配置都編譯一遍，解析資源用量，只保留沒有溢出的（剩下 24 個）。
- **現在怎麼抓：**只要 kernel 模板有變動就檢查資源用量。

### <a id="kern-5"></a>KERN-5 · 捕捉陣列的 device lambda → scratch
- **修正：**fetch 輔助函式改用巨集而不是 lambda。

### <a id="kern-6"></a>KERN-6 · HIP 的 `int4` 陣列放在 scratch
- **症狀：**一個 int8 GEMM 探針只有 52.5 TOPS；它的預取緩衝區在 scratch 裡。
- **根本原因：**HIP 的 `int4` 是 struct；它的陣列沒有被提升到暫存器。
- **修正：**改用 `ext_vector_type` 向量 → 95.0 TOPS。

### <a id="kern-7"></a>KERN-7 · 被提出迴圈的不變位址：700 多個溢出
- **條件：**f16-WMMA 的 DeltaNet scan。
- **根本原因：**編譯器把 64 個 64-bit 位址（32 個讀、32 個寫）提到迴圈外。
- **修正：**讓 lane 偏移在每次迭代都不透明（`asm volatile("" : "+v"(off))`），把位址拆成 uniform
  基底 + 32-bit 偏移 → 20 個溢出。

### <a id="kern-8"></a>KERN-8 · 排程器預先載入了所有 V fragment
- **條件：**gfx1151 的 prefill attention。
- **症狀：**256 個 VGPR + 13 個溢出，56 B scratch；兩次重新組織結構後仍溢出 11–18 個。
- **修正：**在每個 WMMA 之後釘住它的累加器 → 253 個 VGPR、0 溢出，prefill +1.1%，逐位元相同。

### <a id="kern-9"></a>KERN-9 · 交錯的 WMMA 鏈撐爆暫存器檔
- **條件：**int8 WMMA 中批次 GEMV。
- **症狀：**256 個 VGPR 外加數百 bytes 的溢出。
- **修正：**wave-uniform 索引用 `readfirstlane`、無分支的 16-byte header 解碼、每個單元累加器用
  `asm volatile` 釘住。

### <a id="kern-10"></a>KERN-10 · 計算出來的權重指標改變了 codegen
- **症狀：**加入分組 launch 的 ABI 後，多 token GEMV 的 VGPR 增加 30–90%，部分變體出現溢出
  （q6_k v7 T=8：0.92 → 1.97 ms）。
- **根本原因：**只要權重指標不是直接來自 kernel 參數（`select`，或指標 + 偏移）就會發生。
  `readfirstlane`、`__builtin_assume`、整數偏移都沒用。
- **修正：**分組 launch 用另外的「雙胞胎」入口；單一 launch 保留原本的入口。
- **現在怎麼抓：**GEMV 變動後，比對每個入口的 code-object metadata（VGPR/SGPR/scratch）。

### <a id="kern-11"></a>KERN-11 · 多了第二個呼叫者後不再 inline
- **症狀：**248 個 VGPR + scratch；verify n=3 從 30 → 32 ms。
- **修正：**共用實作加上 `__forceinline__`。

### <a id="kern-12"></a>KERN-12 · 管線化迴圈中的 `break` 讓預取失效
- **症狀：**預取深度 2–3 沒有效果；ISA 顯示 wait counter 每一步都被歸零。
- **修正：**沒有 `break` 的主迴圈加上獨立的尾段。

### <a id="kern-13"></a>KERN-13 · 發散的 scale 分支讓記憶體延遲序列化
- **條件：**Q3_K/IQ3_S 單 token GEMV（491/466 與 502/482 GB/s，上限為 593–598）；中批次 GEMV 的
  `j < 2` scale 分支。
- **根本原因：**`kk < 8 ? sb8[kk] & 0xF : sb8[kk-8] >> 4` 變成 wave 發散的分支：load、分支、load、
  等待、load——每個 super-block 有兩段序列化的延遲。
- **修正：**無分支的位元組索引與位移；計算前先為 4 個 super-block 發出 load → +16…+25%。
- **現在怎麼抓：**基準測試自我回報中列出各型別 GB/s 對純讀取上限的比較。

### <a id="kern-14"></a>KERN-14 · 編譯器不肯選 `v_mad_i32_i24`
- **症狀：**int8 epilogue 中出現 `mul_lo_u32` 或 `mul24 + add3`。
- **修正：**inline asm（與 header 預先解碼一起，從 95.0 提升到 118.2 TOPS）。

### <a id="kern-15"></a>KERN-15 · `v_perm` 查找表中有個常數錯了
- **症狀：**IQ4 對某些 code 解碼錯誤；表中一個常數是 `0xCD`，應為 `0x98`。
- **偵測：**發布前與 numpy 參考比對時抓到。

### <a id="kern-16"></a>KERN-16 · 快取的 int8 activation 沒有失效
- **症狀：**來源緩衝區被重寫後仍讀到過期資料。
- **修正：**在每個寫入點重設快取鍵。

### <a id="kern-17"></a>KERN-17 · 32-bit 候選遮罩在 48 個候選時溢位
- **根本原因：**對 32-bit 遮罩位移 ≥ 32 是未定義行為。
- **修正：**改用 64-bit 遮罩。

### <a id="kern-18"></a>KERN-18 · 自動調校量到的是 launch 開銷
- **症狀：**調校器偏好融合路徑。
- **根本原因：**每次重複後都 sync。
- **修正：**對一批重複只在最後 sync 一次來計時。冷快取調校沒有幫助。

### <a id="kern-19"></a>KERN-19 · 批次 bucket 在錯誤的大小上調校
- **症狀：**85 token 的提示詞被補齊到 256 token 的 tile（FFN-up 0.791 ms，用較小 tile 只要
  0.271 ms）；server 的 1024 列 chunk 用的是在 512 上調校的配置。
- **修正：**更多 bucket（11 個），每個都在自己的大小上調校；85–89 token 的 prefill −29%；光是 1024
  bucket 就讓 server 8k 冷 prefill +9.3%。
- **現在怎麼抓：**不變性檢查涵蓋每個選項（tile 選擇絕不改變位元）。

### <a id="kern-20"></a>KERN-20 · 每個 workgroup 的 LDS 超過約 41 KB 會壓低佔用率
- **實測：**每個 SIMD 從 16 → 6 個 wave，比 LDS 較小的版本慢。

### <a id="kern-21"></a>KERN-21 · 「所有 query 只讀一次 KV」反而更慢
- **實測：**24k MTP 39.5 → 35.6 tok/s。
- **根本原因：**瓶頸在歸約，不在 KV 讀取；L2/Infinity Cache 已經承接了重複的讀取。

### <a id="kern-22"></a>KERN-22 · gfx11 WMMA 在一種情況下不精確
- **條件：**gfx1151 分組 verify attention。
- **症狀：**分組後的 query 欄與單欄結果不是逐位元相同。
- **根本原因：**P = 0 的項乘上另一個 query 的真實 V 列，沒有產生精確為零的貢獻。
- **修正：**在 gfx1151 上每組只放一個 query。

### <a id="kern-23"></a>KERN-23 · gfx11 WMMA 會複製 B，所以我們解碼了兩次
- **修正：**每個 half-wave 解碼自己那一半，再用 `v_permlanex16` 交換 → IQ4_XS T=3 −8…−9%，
  逐位元相同。

### <a id="kern-24"></a>KERN-24 · 在 gfx1201 上，精確的 int8 prefill 贏不了 f16
- **實測：**W8A8 精確 96.5 TOPS、有損 108.8，精確 W4A8 最佳 126.1，對比 f16 108–116 TFLOPS；FFN-up
  只有 +6.6…+10.4%，down-projection 更差；GEMM 誤差大 35 倍。
- **根本原因：**每個 sub-block 的整數 scale（每個元素一個 `v_mad_i32_i24`）把 iu8 WMMA 限制在約
  205 TOPS；每個 WMMA 只有 4–5 個 VALU 能免費重疊。
- **規則：**int8 GEMM 只有在沒有 sub-block scale 時才划算（或像 MXFP4 那樣把 2 的冪次 scale 摺進
  fp8）。llama.cpp 的 Vulkan int8 路徑在這張卡上得到相同結論。

### <a id="kern-25"></a>KERN-25 · 把反量化融合進 GEMM 反而更慢
- **實測：**逐位元相同，但 Q4_K_M prefill −10…−13%：解碼用的 VALU 落在 GEMM 的關鍵路徑上。
- **規則：**大批次時先反量化再 GEMM 勝出；融合前先量測。

### <a id="kern-26"></a>KERN-26 · 雙緩衝 LDS 與權重直接進暫存器都輸了
- **實測：**fp8 GEMM：直接 A fragment 183–189 TOPS、雙緩衝 158–173，對比單緩衝 + 暫存器預取
  194–210（在 fragment-tiled 改寫之前）。int8 也是同樣結果（v4 110 對 v3 126）。

### <a id="kern-27"></a>KERN-27 · Prefill attention 對程式碼佈局很敏感
- **實測：**在非對角 tile 上略過 causal mask：30k 時 −4%，120k 時 +1.3%；只略過 mask：+8%（更差）；
  釘住 PV 迴圈讓 scratch 從 252 → 28 B/lane，但慢了 5%。
- **規則：**每個變更都要在多個上下文長度下量測。

### <a id="kern-28"></a>KERN-28 · KV 位元組減半，decode 反而變慢
- **實測：**q8h decode 在 16k／64k／128k 時每 token 毫秒數 +1.0%／+3.6%／+5.2%；prefill attention
  +49%。
- **根本原因：**在 split-K kernel 中反量化 K 的成本高於省下的頻寬。
- **修正：**q8v（K 用 f16、V 用 int8）的 decode 和 f16 一樣快（[kv-and-caching.md](kv-and-caching.md#formats)）。

### <a id="kern-29"></a>KERN-29 · MoE 路由的近似平手讓 KL 門檻在沒實測前毫無意義
- **症狀：**一項長上下文 MoE 檢查在 8060S 上失敗（KL 3.84e-3，門檻 1e-3）。
- **證據：**按（token, layer）傾印專家 id：任兩條有效路徑之間，從第 0–4 層起就有 17–28% 的配對選了
  不同的專家組合；45 組路徑配對的 KL 為 1.2e-4 … 3.4e-2，top-1 全部相同。Dense 27B：KL 約 1e-7。
- **修正：**MoE 長上下文門檻定為 1e-2，並要求 top-1 相同，且「兩邊 top-10 中每個 p ≥ 1e-3 的 token
  都要出現」；理由寫在腳本裡。真正的 bug 會表現為 top-1 改變或 KL ≫ 0.1。

### <a id="kern-30"></a>KERN-30 · 依批次 bucket 決定輸出精度，破壞了 solo == batched
- **條件：**選用的 f16 輸出路徑。
- **根本原因：**GEMM 是否寫出 f16 取決於 n 所屬的調校 bucket；單獨 chunk 與分段 forward 落在
  不同的 bucket。
- **修正：**只有在每個 bucket 的配置都支援時才啟用 f16 輸出。
- **現在怎麼抓：**在寬鬆模式下也跑分段 prefill gate。

### <a id="kern-31"></a>KERN-31 · 兩套分支的 kernel 在一個參數上走岔了 **[8060S]**
- **條件：** gfx1201 與 gfx1151 的 kernel 各自維護原始碼、kernel 名稱相同；host 共用同一條啟動路徑。
- **症狀：** 短測試看不出來。kernel 測試的「分 head 區段啟動 == 一次啟動」在 gfx1151 上失敗（一半的 head 錯）。
- **根本原因：** gfx1201 的 `attn_prefill_wmma*` 為了把長 prefill 分成多個 head 區段，加了 `h0`（第一個 head）參數；
  gfx1151 的版本沒有。host 傳了 `h0`、kernel 沒用，第一段之後的每一段都重算 head 0 起的內容 —— 只在提示長到需要分段時
  （n × context > 4096 × 128k）才會發生。
- **修正：** gfx1151 kernel 補上 `h0`。現在會從 AMDGPU metadata（`.args`：offset、size、kind）比對兩個 code object 中
  共同 kernel 的參數版面。
- **現在怎麼抓：** ABI 比對（0 個不同）與 `whirl-kernel-test` 的分 head 區段不變性檢查（每張 GPU 都跑）。

### <a id="kern-32"></a>KERN-32 · int8 scale word 的格式依架構而不同 **[8060S]**
- **症狀：** 在 gfx1151 上所有 int8 activation 的檢查都失敗：`quantize_q8 xd` 每個 word 都不同，GEMV 參考差了 1e37。
- **根本原因：** gfx1151 kernel 把每 32 值區塊的 scale 存成一個 32 位元 word：(f32 scale 捨入到 11 位元尾數，區塊總和
  + 4096)，內積 kernel 因此免費拿到區塊總和；CPU 參考卻假設是單純的 f32 scale。
- **修正：** code object 以標記 kernel 宣告格式（`kernels::Caps::xd_sum`）；參考實作依此打包 / 解包
  （`ref::setXdSum`、`ref::xdScale`）。
- **現在怎麼抓：** 兩張 GPU 上的 `quantize_q8` / 融合量化 exact 檢查。

---

## <a id="mtp"></a>4. 推測解碼的精確性

### <a id="mtp-1"></a>MTP-1 · 多 token GEMV 差一個 ulp，就破壞了 MTP == greedy
- **症狀：**多 token 內積改寫後，MTP 輸出不再與單純 greedy 輸出一致。
- **根本原因：**verify 的 logits 與 decode 的 logits 在最後幾個位元不同；近似平手的 token 被翻轉。
- **修正：**單 token kernel 與所有多列 kernel 共用標準的逐單元運算式與相同的 xor 歸約樹。
- **現在怎麼抓：**`checkGemvBitwise`（所有型別 × 2–16 列 × 所有變體 × 輸出 head）；在 CLI、server
  與並行測試中，草稿數 1、2、5、10 以及強制 n-gram 時都要求 MTP == 單純 greedy。

### <a id="mtp-2"></a>MTP-2 · F32 小矩陣在 n = 1 與 n ≥ 2 時走不同 kernel
- **條件：**GGUF 把 `ssm_alpha`／`ssm_beta` 存成 F32。
- **症狀：**三個量化變體上都是 MTP ≠ 單純 greedy；自動草稿數掉到 1。
- **根本原因：**F32 權重：單列用 f32-activation GEMV，verify 用 f16 GEMM。
- **修正：**這些張量改用 Q8_0（融合的精確路徑）；引擎修正已排入佇列。
- **現在怎麼抓：**每個新模型檔都跑 MTP 冒煙測試（plain == MTP == MTP + n-gram == 強制 n-gram）。

### <a id="mtp-3"></a>MTP-3 · 序列結束後仍接受草稿
- **症狀：**下一輪快取未命中（`cache now 42`，下一輪的共同前綴為 41）。
- **根本原因：**`<|im_end|>` 之後的草稿被吃進狀態與快取的 token 中。
- **修正：**在 EOS 之前截斷已接受的草稿；`max_tokens` 也一樣。輸出不變。
- **現在怎麼抓：**多輪快取命中測試要求每一輪快取的量 ≥ 前一輪提示詞 − 1。

### <a id="mtp-4"></a>MTP-4 · 計時表跨請求沿用
- **症狀：**同一使用者的第二個及之後的請求草擬太多（在 8060S 上每輪約 4.5，應為約 3.3）；在 R9700
  上 server 比 CLI 慢 3.6%。
- **修正：**請求在閒置引擎上開始時，重設成本模型的計時表（請求 2–4 +4.1%；server 對 CLI −0.5%）。
- **現在怎麼抓：**server 對 CLI 的速度檢查，附修正前／後的 CLI 參考值。

### <a id="mtp-5"></a>MTP-5 · 依 MTP 草稿上限配置的緩衝區被 n-gram 草稿撐爆
- **根本原因：**陣列按 10 個草稿配置；n-gram 允許更多。
- **修正：**一律按最大 verify 批次（16 列）配置。

### <a id="mtp-6"></a>MTP-6 · 天真的 n-gram 草擬器只會讓速度變慢
- **實測：**編輯任務 128.4 → 111.5 tok/s（−13%）。
- **根本原因：**只要有匹配就使用；只取 3-gram 最近的一次出現（常見程式碼的 3-gram 會指向錯誤的
  位置）；上限 8 個。
- **修正：**3 與 12 token 的鍵、最長向後匹配、反事實評分、由成本模型決定、15 個草稿 → 編輯
  +48…+99%。

### <a id="mtp-7"></a>MTP-7 · 一次性的首次使用成本污染了計時模型
- **根本原因：**第一次 16 列 verify 會多花約 50 ms，只發生一次。
- **修正：**忽略每種列數的第一個樣本。

### <a id="mtp-8"></a>MTP-8 · 截斷草稿詞彙表毀掉了中文接受率
- **實測：**82% → 28–33%（中文 token 的 id 很大）；之後在 150k/100k 時 2.58 → 2.43 token/輪。
- **規則：**草稿 head 可以量化（2-bit 可行），但不能截斷。

### <a id="mtp-9"></a>MTP-9 · 草稿越多，MoE 模型越吃虧
- **實測：**各位置接受率 92／31／4／0%；24k 時 3 個草稿比不用草稿還慢。
- **修正：**MoE 預設 1 個草稿。

### <a id="mtp-10"></a>MTP-10 · 每個草稿都做一次 host sync 的 p-min 花掉 3%
- **修正：**改在 GPU 上決定（由後續草稿 kernel 讀取停止旗標）；每輪只 sync 一次。

---

## <a id="srv"></a>5. 伺服器與快取

### <a id="srv-1"></a>SRV-1 · `<think>\n` 與 `<think>\n\n</think>`：每一輪都重新 prefill
- **條件：**思考模式；不回傳 `reasoning_content` 的客戶端（大多數 agent 客戶端）。
- **症狀：**多輪快取命中率 0.000；每一輪都把整段對話重新 prefill。
- **根本原因：**提示詞以 `<think>\n` 結尾；重新渲染的歷史則含 `\n\n`，是單一個不同的 token——`prompt-end` 檢查點差一個 token 而沒命中。
- **修正：**這類提示詞在 N−1 處切開，並在 `<think>` 之後存一個 `think-open` 檢查點（CLI 也照樣切，維持 server == CLI）。命中率 0.000 → 0.928（agent）／0.515（聊天）。
- **現在怎麼抓：**多輪快取測試，涵蓋五種 session 類型、兩個模型。

### <a id="srv-2"></a>SRV-2 · 生成的 token 不一定是標準 tokenization
- **症狀：**偶爾在一輪中途出現快取分歧（例如程式碼中 `"""'` 附近）。
- **決定：**接受（損失一次回覆的 prefill）；另一種做法會讓有快取與無快取的輸入不同。

### <a id="srv-3"></a>SRV-3 · 有快取與無快取並非逐位元相同
- **根本原因：**重用的歷史 KV 來自 decode kernel，重新 prefill 的歷史則來自 prefill GEMM。
- **測試中的修正：**文字不同時，由一個驗證 server 重新 prefill 每個有快取的輪次，要求 KL ≤ 1e-2 且 top-1 相同。

### <a id="srv-4"></a><a id="srv-ngram-pages"></a>SRV-4 · 驗證列寫進了未對應的 KV 頁
- **條件：**n-gram 草稿 > 8，且有額外的快照容量。
- **症狀：**多輪對話的第二輪結果不同。
- **根本原因：**slot 只保證頁面到 `pos + n_draft + 2`；多出來的驗證列打到頁表第 0 項 → 池的第 0 頁，也就是另一個序列的資料。
- **修正：**以 slot 已對應的頁數限制草稿數。
- **現在怎麼抓：**agent 基準測試（LF 與 CRLF），在 off／on／on 加更多快照三種設定下，六輪全部相同。

### <a id="srv-5"></a>SRV-5 · 執行中的 slot 霸佔已失效 session 的頁面（無法還原）
- **症狀：**其他 slot 在 decode 時，一次 126k-token 的還原始終沒發生；請求退回 117k-token 的 prefill（約 75 s；其他請求掉到 9–15 tok/s），最後以 `KvPoolFull`（HTTP 500）失敗。
- **根本原因：**某個 slot 接到一個什麼都沒重用的短請求，卻保留了舊的 459 頁；執行中的 slot 不可逐出；空閒頁只有 310 頁。
- **修正：**工作開始時歸還超過 `N + drafts + 2` 的頁面（考慮隔離區／參考計數）。
- **現在怎麼抓：**`restore_conc_gate`（在 3 個 decode 中的 slot 旁還原 == 從未被逐出）。

### <a id="srv-6"></a><a id="srv-defaultenv"></a>SRV-6 · 只有在預設設定下才出現亂碼輸出
- **症狀：**某個候選版本通過了所有 gate，但用 server 預設設定跑的基準測試卻產生 61 筆亂碼輸出。
- **根本原因：**在自動 KV 格式下，server 載入後重新載入 kernel 表；這次重設把函式指標換掉（fp8 producer 變回 row-major），但一個「tiled」旗標仍是開的——producer 寫一種排列，GEMM 讀另一種。每個 gate 都明確指定了 KV 格式，所以測試中從未發生這次重新載入。
- **修正：**絕不為載入時的決定去改動 kernel 表；每次啟動時依狀態選擇。壞掉的候選版本已立即解除安裝。
- **現在怎麼抓：**常設的「預設環境」階段：不帶任何變數與選項的 server，必須選出與 CLI 參考相同的 KV 格式，且 greedy 與固定種子的輸出都要完全相同。

### <a id="srv-7"></a>SRV-7 · MXFP4 並行 ≠ 單獨（路徑條件問題）
- **症狀：**MXFP4 在 2–4 個並行請求時：greedy ≠ 單獨執行（其中一個請求在第 4 個字元就開始分歧——在 prefill 之內）。
- **二分搜尋：**停用分段 prefill 時消失；停用兩項速度模式功能時消失；只停用其中一項時仍存在。
- **根本原因：**一條 f16 輸出的 DeltaNet 路徑以「沒有分段」為條件，所以分段 prefill 走 f32 路徑，單獨執行走 f16 路徑。
- **修正：**分段 prefill 每段都走同一條路徑。
- **現在怎麼抓：**MXFP4 的分段 prefill gate＋C = 4 並行 == 單獨的檢查。

### <a id="srv-8"></a>SRV-8 · 等待背景寫出時卡住了 decode
- **症狀：**新請求打到正在寫出中的 slot 時，主串流卡住約 150 ms（寫出 692 MiB ≈ 180 ms）。
- **修正：**邊界頁採 copy-on-write＋舊頁面隔離到複製 fence 完成為止；等待次數 0。

### <a id="srv-9"></a>SRV-9 · slot 一直連結到錯誤的層條目
- **症狀：**一個不相干的 session 就地覆寫了另一個 session 的 RAM 條目。
- **修正：**新請求什麼都沒重用時解除連結。
- **現在怎麼抓：**`tier_gate`。

### <a id="srv-10"></a>SRV-10 · 跨串流順序：檢查點尚未寫入就被寫出
- **症狀：**共享檢查點的 RAM／SSD 副本存的是複製前的內容。
- **修正：**層串流等待一個在主串流複製之後記錄的 event。
- **一併修正的競態：**逐出後跳過了寫出完成檢查，導致後續的檢查點儲存覆寫了寫出仍在讀取的緩衝區。

### <a id="srv-11"></a>SRV-11 · 延後的 SSD 寫入從未被排程
- **症狀：**重新啟動後，還原到較舊的條目（28,356 而非 28,414 個 token）。
- **根本原因：**條目正寫入 SSD 時被延後的寫出，讓閒置迴圈一直處於睡眠狀態。
- **修正：**延後之後回報「忙碌」。

### <a id="srv-12"></a>SRV-12 · 系統提示詞中的字面 `<|im_end|>` 移動了快取邊界
- **症狀：**一個 30k-token 的系統提示詞沒有被重用；server 退回較粗的 `prefix` 檢查點。
- **根本原因：**系統文字（tokenizer 文件）含有 `<|im_end|>`，被解析成特殊 token；邊界搜尋用的是第一個 `<|im_end|>`。
- **修正：**改用三個 token 的模式 `<|im_end|>\n<|im_start|>`；gate 的系統提示詞現在含有字面 `<|im_end|>`。

### <a id="srv-13"></a>SRV-13 · prefill chunk 變大反而讓後續請求變慢
- **症狀：**chunk 2048：冷 prefill +18…+21%，但跨 slot 的後續請求重用的是 27,418 而非 28,442 處的檢查點（30k + 1000 token：1548 → 2230 ms）。
- **修正：**以 1024-token chunk 排程，執行時合併成最多 2048 列的 forward。

### <a id="srv-14"></a>SRV-14 · 檢查點放在 pinned host 記憶體：96 次小複製
- **實測：**每次儲存／載入約 40 ms，約 3.75 GB/s；agent 輪次 −25%。
- **規則：**在 host 複製前先收集到連續緩衝區；熱檢查點留在 VRAM。

### <a id="srv-15"></a>SRV-15 · 被逐出的 session ≠ 單獨——是政策，不是 bug
- **症狀：**共享系統檢查點上線後，一個多 session 測試的「tier == solo」檢查失敗。
- **根本原因：**比系統檢查點多不到 512 個 token 的條目不會被還原；約 60 個由 decode 產生的 token 會被重新 prefill（數值上等價）。
- **證據：**強制還原 28/28 == 單獨；位元組驗證模式發現全部 115 次寫出／還原都相同。

### <a id="srv-16"></a>SRV-16 · 測試用 server 共用同一個 SSD 快取目錄
- **症狀：**後面的測試還原了前面測試寫入的條目，而不是做 prefill。
- **修正：**每個測試 server 用全新的 SSD 目錄（預設環境測試保留真正的預設值）。

### <a id="srv-17"></a>SRV-17 · 參考 server 選了不同的 KV 格式
- **症狀：**某個 gate 中的每一項比較都「失敗」，連快取 token 數為 0 的請求也一樣。
- **根本原因：**沒有快取的參考 server 空閒 VRAM 較多，自動規則選了 f16；受測 server 選了 q8v。在系統提示詞 gate 發生過，在視覺 gate 又發生一次。
- **修正：**在比較 server 的 gate 中固定 KV 格式。

### <a id="srv-18"></a>SRV-18 · 隨機的工具呼叫 id 破壞了相等性檢查
- **修正：**以名稱與參數比較工具呼叫。

### <a id="srv-19"></a>SRV-19 · 一波請求中的第一個單獨 prefill，然後卡住
- **症狀：**一波 4 個請求中，第一個請求開始 decode，接著在其他請求一起 prefill 時停住 2.4 s。
- **修正：**對仍在接收或 tokenize 中的請求，做 30 ms 的批次聚集。

### <a id="srv-20"></a>SRV-20 · 平行的子 agent 重複計算同一個系統提示詞
- **修正：**排隊中的請求等待另一個請求正在產生的系統檢查點。

---

## <a id="tok"></a>6. Tokenizer 與模板

### <a id="tok-1"></a>TOK-1 · 沒有實作 Qwen 的空白規則 `\s+(?!\S)`
- **條件：**我們自己為 `qwen35` pre-tokenizer 類型寫的 BPE pre-tokenizer（MoE 模型也用它）。
- **症狀：**看起來一切正常——輸出很流暢。是在調查為何複製檔案的 n-gram 草稿接受率只有 25% 時才浮現：提示詞的 token 數與參考不同（一個寫程式的提示詞 152 → 156 個 token；一個 4k-token 的原始碼提示詞 4068 → 4128）。
- **根本原因：**切分 regex 是 `…|\s*[\r\n]+|\s+(?!\S)|\s+`。負向前瞻的意思是：一串空白後面接單字時，*最後一個*空白要分給單字：`"    return"` 必須切成 `"   " + " return"`，而不是 `"    " + "return"`。此外 `\s*[\r\n]+` 必須延伸到一串換行中的最後一個換行（`"\n  \n"` 原本切錯），而且 `\v` 也算空白。每個有縮排的程式碼提示詞都被 tokenize 成模型訓練時從沒看過的序列。模型自己寫出的是標準 token——所以 MTP 草稿（來自模型）複製得很好，但 n-gram 草稿（來自提示詞）對不上。
- **修正：**實作前瞻（在非空白字元前退回一個空白）以及換行串規則。行為改變：有縮排的提示詞現在會產生不同（正確）的輸出。agent session 的快取命中率上升（dense 0.972 → 0.978，MoE 0.963 → 0.975），因為重新 tokenize 的歷史現在與生成的 token 一致。
- **驗證：**以 GGUF 詞彙表建構、在 tiktoken 的 fancy-regex 上套用 llama.cpp `qwen35` 模式的參考 tokenizer，在原始碼檔案、大量中文的文件（33k token）、一個 44k-token 的原始碼檔案、所有基準測試提示詞與空白邊界字串上逐 token 一致。
- **現在怎麼抓：**C++ tokenizer 在 440 種檔案／模型／模式組合（約 1.5 M token）上與 `llama-tokenize` 比對，語料中含縮排邊界案例（空白、tab、混用、空行）。

### <a id="tok-2"></a>TOK-2 · 提示詞中的 CRLF 檔案 vs LF 輸出
- **症狀：**n-gram 比對在 Windows 檔案上失敗。
- **根本原因：**`"\r\n"`（id 317）永遠對不上模型的 `"\n"`（198）。
- **修正：**為 n-gram 索引建立 token 層級的 CRLF→LF 正規化表（5 個檔案中 653 個 CR token，全部 1:1；正規化後的序列 == LF 檔案的 token）；除非模型自己寫出 CR，否則草稿輸出 LF。

### <a id="tok-3"></a>TOK-3 · Unicode 15.1 vs 16.0
- **症狀：**有 5,185 個碼位的分類與 llama.cpp 不同。
- **根本原因：**llama.cpp 的表是 Unicode 15.1；Python 3.14 的 `unicodedata` 是 16.0。這 5,185 個全是 16.0 新增的字元（在 llama.cpp 中為 UNDEFINED）。
- **修正：**預設以 15.1 版本產生表（可用選項改為 16.0）。
- **現在怎麼抓：**針對全部 0x110000 個碼位，與 llama.cpp 的 `unicode-data.cpp` 驗證表格；語料中加入一個只有 16.0 字元的案例。

### <a id="tok-4"></a>TOK-4 · 前導位元組 F5–F7 解碼後超過 U+10FFFF
- **症狀：**遇到這類輸入時，`llama-tokenize` 因未處理的 C++ 例外而中止。
- **修正（WHIRL）：**丟出 tokenizer 錯誤；一致性測試把兩者都算作相同的錯誤結果。

### <a id="tok-5"></a>TOK-5 · 不同引擎的 Jinja `tojson` 浮點數輸出不同
- **事實：**llama.cpp 的 Jinja 把 `2.0` 印成 `2`，精度 6 位數；Python `json.dumps` 印成 `2.0`。WHIRL 依循 Python；已記錄為已知分歧。

### <a id="tok-6"></a>TOK-6 · `|trim` 在 Python Jinja 與 C++ 引擎間不同
- **事實：**Python 的 `str.strip()` 還會移除 U+3000、U+00A0、U+2028、U+0085、U+001C–U+001F；llama.cpp 與 WHIRL 只修剪 ASCII 空白。只影響以這類字元開頭／結尾的內容。

---

## <a id="quant"></a>7. 模型檔案與量化

### <a id="quant-1"></a>QUANT-1 · MXFP4 與 Q8_0 會忽略 imatrix
- **事實：**主線 `llama-quantize` 對 MXFP4（`GGML_UNUSED(quant_weights)`）或 Q8_0 不使用量化權重；只有 K-quant 張量（這裡是輸出頭）會用到。

### <a id="quant-2"></a>QUANT-2 · llama.cpp b11214 沒有 dense MXFP4 檔案類型
- **修正：**ftype `MXFP4_MOE`＋`--tensor-type-file`（第一個符合者優先）＋`--output-tensor-type`。以傾印張量類型驗證結果。

### <a id="quant-3"></a>QUANT-3 · 輸出頭類型對 MTP 速度的影響大於一般速度
- **實測：**Q8_0 頭：接受率最高（73.2%），但 MTP 98.0 vs 117.0 tok/s（Q6_K 頭）。Q4_K 頭：不用 MTP 時 +1.9%，用 MTP 時 −4.5%（沒有 Q6_K 專用的草稿／驗證路徑）。
- **規則：**WHIRL 優先用 Q6_K 頭。

### <a id="quant-4"></a>QUANT-4 · fp8 activation 會放大其他每一種擾動
- **實測：**同一個 DeltaNet 修改：f16 prefill 下 KL 4.5e-8 … 1.5e-4，fp8 下 1.2e-4 … 7.0e-2。
- **規則：**對精度敏感的修改要在精確模式下評估；速度模式的 KL 只記錄，不設 gate。

（另見 [MTP-2](#mtp-2)：F32 `ssm_alpha`／`ssm_beta`。）

---

## <a id="vis"></a>8. 視覺

### <a id="vis-1"></a>VIS-1 · 未寫入的填充 lane 讓 embedding 取決於 server 歷史
- **症狀：**server 中的影像 embedding 與獨立編碼器不同，且會隨先前的流量而變。
- **根本原因：**Q/K 準備 kernel 沒有寫入填充的 head 維度 lane 76–79；在 server 裡，activation 借用 prefill 暫存區，裡面留有舊資料。獨立工具的全新 arena 掩蓋了這個問題。
- **修正：**寫入填充部分。
- **現在怎麼抓：**經過任意歷史後，server embedding == 獨立版，逐位元相同。通用規則：用髒緩衝區測試。

---

## <a id="meas"></a>9. 量測

### <a id="meas-1"></a>MEAS-1 · 快取讓微基準測試數字虛高
- **修正：**輪替 > 512 MB（R9700，64 MB Infinity Cache）或 > 2 GB（8060S，32 MB MALL）的真實權重。

### <a id="meas-2"></a>MEAS-2 · 機器漂移看起來像進步與退步
- **實測：**沒改程式碼，基準測試 session 之間就差 2–6%；MoE 模型在不同時段之間出現 ±3.5% 的雙峰偏移。
- **修正：**在同一個 session 中交錯 A/B，2 輪以上，看 min–max。

### <a id="meas-3"></a>MEAS-3 · 筆電散熱：boost vs 持續 **[8060S]**
- **實測（ROG Flow Z13，持續約 80 W）：**約 5–7 s 後 prefill −15%；連續執行的行程：一般 decode 在 3–5 分鐘後 13.4 → 5.4–5.8 tok/s（−58%），新舊執行檔表現一樣。
- **修正：**每次執行前閒置 60–90 s、交錯執行、分別回報 boost 與持續數字；迷你 PC 的數字不可比較。

### <a id="meas-4"></a>MEAS-4 · 速度檢查拿冷的 CLI 比熱的 server
- **症狀：**在 8060S 上 server 比 CLI「慢 −7…−14%」。
- **修正：**在 server 前後各量一次 CLI 參考（各取 3 次中位數），容忍度 5%＋漂移，server 請求之間間隔 15 s。改寫之後反而揭露了一個真正的 server 問題（[MTP-4](#mtp-4)）。

### <a id="meas-5"></a>MEAS-5 · 過嚴的 top-k 一致性門檻
- **症狀：**int8 雙 token 路徑偶爾會調換排名第 10 的 token。
- **修正：**top-10 重疊 ≥ 9（KL 與 top-1 不變）；MoE 則用機率下限規則（[KERN-29](#kern-29)）。

### <a id="meas-6"></a>MEAS-6 · 不夠敏感的 QA 基準測試
- **症狀：**只給答案的 QA：兩個模型都只有 5–7% 正確——毫無偵測能力。
- **修正：**在 `ANSWER:` 之前先推理，350 題，對不一致的配對做 McNemar 檢定。

### <a id="meas-7"></a>MEAS-7 · 不同輸出之間的推測解碼速度
- **例子：**一個數值修改讓一段英文摘要變成 138 而非 105 個 token；接受率從 2.76 → 2.42 tokens/cycle，而每個 cycle 變快（40.6 → 39.9 ms）。
- **修正：**分開比較 ms/cycle 與接受率；只在輸出完全相同或多提示詞平均時比較 tok/s。

### <a id="meas-8"></a>MEAS-8 · 會違反 VRAM 規則的工具
- **事實：**`llama-perplexity` 共享使用量峰值 306 MiB（速度無效，數值沒問題）；llama.cpp 沒設 `GGML_CUDA_NO_PINNED=1` 會多出約 1 GB 共享使用量。

### <a id="meas-9"></a>MEAS-9 · 緩衝 I/O 與佇列深度會掩蓋 SSD 速度
- **實測：**緩衝寫入 1.8 GB/s vs 無緩衝 5.1–5.3；2 MiB 讀取 QD1 4.1 vs QD4 7.0 GB/s。

### <a id="meas-10"></a>MEAS-10 · 參考引擎也可能壞掉
- **事實：**llama.cpp b10686 Vulkan 在 R9700 上 decode 只有 5.6 tok/s（prefill 正常）；b11214 已修正。b11214 Vulkan 搭配 MTP 在 14k/24k 上下文時，prefill 從約 800 掉到約 97 tok/s。ROCm 版需要 `--load-mode none`（[HIP-3](#hip-3)）。
- **規則：**發表比較之前，先檢查參考是否正常。

### <a id="meas-11"></a>MEAS-11 · agent 基準測試需要有上限的工具迴圈
- **症狀：**在一個文字已分歧的設定中，模型不斷呼叫 `read_file` 直到 128k 上限，基準測試因此當掉。
- **修正：**在測試框架中限制每題的工具呼叫次數。

---

## <a id="egpu"></a>10. eGPU（USB4）——環境限制

### <a id="egpu-1"></a>EGPU-1 · 約 3.8 GB/s 的主機連結
- **實測：**不論分塊大小，D2H 3.78–3.81、H2D 3.84–3.86 GB/s；KV 還原每個 token 約 14–16 µs；128k 還原約 1.9–2.0 s。
- **狀態：**環境限制；直接接 PCIe 預期會快很多（未量測）。

### <a id="egpu-2"></a>EGPU-2 · H2D 流量會拖慢 kernel 派發
- **實測：**H2D 佔滿連結時，1500 個小 kernel 15 → 64 ms（20 個大 kernel +4%）；還原期間其他 slot 的 decode cycle +17%。`GPU_BLIT_ENGINE_TYPE` 無效；`PAL_DISABLE_SDMA=1` 只是把成本挪到別處；VRAM 中轉＋D2D scatter 更慢（3.51 vs 3.69–3.83 GB/s）。
- **狀態：**成本與位元組數成正比；調節節奏沒有幫助；未最佳化。

### <a id="egpu-3"></a>EGPU-3 · 視覺權重每張影像都要串流一次
- **實測：**編碼器權重未常駐時（27B 預設設定），每張影像約 260 ms 的 H2D。
- **狀態：**取決於環境；VRAM 足夠時用常駐模式可避免。

---

條目數：HIP 26 · TOOL 14 · KERN 30 · MTP 10 · SRV 20 · TOK 6 · QUANT 4 · VIS 1 · MEAS 11 ·
EGPU 3 — **共 125 條**。
