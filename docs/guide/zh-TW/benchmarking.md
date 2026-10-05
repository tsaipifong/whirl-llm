[English](../en/benchmarking.md) | **繁體中文**

# 基準測試方法與目前數據

> **狀態。** 第 1～9 節說明量測方法。第 10 節的數字是在我們的機器上（R9700 以 USB4 eGPU 連接，Windows 11
> build 26200，HIP SDK 7.2），以 C++ 引擎之前的研究建置於 2026-09-25 → 2026-10-02 量測；C++ 引擎的輸出與它相同，
> 速度差距在 ±1% 內。WHIRL 0.1.0 對 llama.cpp 的發行版量測見 [benchmarks.md](benchmarks.md)。

**對誰有幫助**：在 AMD GPU 上比較推論引擎的人，或想量測小幅（1–5%）kernel 改進而不自欺的人；以及數據會隨溫度漂移的筆電／APU 使用者。

## 1. 原則

1. **以實測上限為基準**，而非規格表（[kernels.md](kernels.md#ceilings)）。
2. **每張 GPU 只跑一個模型行程**，由二進位檔中的具名 mutex 與每個啟動腳本中的檔案鎖強制執行
   （[windows-hip.md](windows-hip.md#wddm-demote)）。曾發生兩次同一張卡上跑了兩個 15 GiB 行程的事故，
   讓好幾個小時的數據作廢。
3. **只用專用 VRAM。** 每次執行都受監控；行程的共享使用量（Shared Usage）超過 256 MiB（基準約
   89 MiB；已扣除宣告過的 pinned host 緩衝區）的執行一律作廢並重跑。
4. **A 與 B 交錯執行，回報 min–max。** 絕不拿新 build 和舊數字比較。
5. **回報百分比，且保留小幅改進。** 核心變快之後，其他地方 2–3% 的增益（或損失）會被放大；被當成
   雜訊否決的點子日後會重新量測。（Q4_K MTP 區塊曾在 −1.6% 時被否決，在其他提速完成後重新量測，
   單人 +0.8%、四人並發 +2.8%，因而被採用。）
6. **正確性關卡（gate）先於速度。** 任何未通過 gate 的較快 build 都不再繼續量測。

## 2. 交錯 A/B

- 舊與新二進位檔（或同一個二進位檔切換某個設定）逐次輪流執行。第 0 輪依序執行提示詞，第 1 輪反序。
  通常 2 輪；範圍重疊時跑 3 輪以上。
- 每個結果都以**各輪的 min–max**（或平均值加 min–max）回報，並註明輪數。
- 記錄每個輸出的雜湊。在 greedy 解碼下，相同引擎／模式／提示詞在各輪產生相同輸出（某次統一執行中
  190/190），因此 min–max 是真正的逐次執行變異，而非內容差異。
- **如果 A 與 B 的輸出不同，不要直接比較 MTP tok/s。** 推測解碼的速度取決於內容。例如：某次 GEMV
  修改讓一段英文摘要的文字從 105 變成 138 個 token；新文字較難預測（每個循環 2.76 → 2.42 個 token），
  該提示詞因而「變慢」，但每個循環其實變快了（40.6 → 39.9 ms）。應分別比較每循環 ms、多提示詞平均值
  與接受率。
- **我們量到的雜訊下限**：R9700 上純 decode（逐 token 生成）的每 token ms 在不同行程間穩定在約 0.2%
  以內；MoE 模型在不同時段之間出現 ±3.5% 的雙峰偏移（同一二進位檔交錯執行為 198.0 對 198.1，但換個
  時段就不同）；自動草稿策略每次執行會增加 ±2–4%。不同 session（工作階段）之間的機器漂移可能看起來
  像 2–6% 的增益或損失；只有交錯執行才算數。
- A/B 執行時要關閉 profile（逐運算事件會讓 decode 速度減半；即使每循環只有三個事件，在 MoE 模型上
  也會損失約 2.5%）。

## <a id="unified"></a>3. 統一協定（WHIRL vs llama.cpp）

兩個引擎都透過各自的 OpenAI 相容伺服器（server），由**同一支客戶端腳本**量測，GPU 上一次只有一個
模型行程，全程持有 GPU 鎖。

| 設定 | 值 |
|---|---|
| llama.cpp | b11214 ROCm `llama-server`：`-ngl 99 -fa on -c 131072 -np 1 --jinja --load-mode none`，`HIP_VISIBLE_DEVICES=1`，`GGML_CUDA_NO_PINNED=1`（否則約 1 GB 的 pinned host 緩衝區會顯示為共享 GPU 記憶體） |
| llama.cpp MTP | `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0.3`（無自動 n-gram） |
| WHIRL | `whirl serve MODEL --port 1234`，預設設定（4 個 slot，自動決定 context 與 KV 格式） |
| 提示詞 | 7 個中英混合程式提示詞（思考關閉與開啟，`max_tokens` 800）+ 5 個檔案編輯提示詞（思考關閉，`max_tokens` 2500，全部以 `stop` 結束）= 每種模式 19 個 |
| 取樣 | 等同 greedy 的 `top_k = 1`（temperature 1.0、top_p 1.0、min_p 0、seed 42 —— **不是** temperature 0），`cache_prompt = false` |
| 模式 | plain（無推測）、MTP、MTP + n-gram（僅 WHIRL） |
| 掃描（plain） | 固定輸入檔案：prefill（提示詞預填）2,022 / 8,178 / 32,751 個 token（`max_tokens` 1）；在 16,354 token 的 context 之後 decode 128 個 token（`ignore_eos`）。token 數以 llama-server 的 `/tokenize` 校準一次後固定 |
| 記錄項目 | 提示詞 token 數、`timings.prompt_per_second`、`timings.predicted_per_second`、MTP 接受率、輸出 SHA-1 |
| 輪數 | 2，交錯（第 0 輪依序，第 1 輪反序） |

使用 `top_k = 1` 而非 temperature 0，可讓兩個引擎走相同的取樣器路徑。llama.cpp 的 MXFP4 kernel 沒有
折疊縮放（folded-scale）的 fp8 路徑，因此它的 MXFP4 數據反映的是其現有 kernel，而非上限。

## <a id="probes"></a>4. 微基準測試：Infinity Cache 與其他陷阱

- R9700 有 64 MB Infinity Cache，8060S 有 32 MB MALL。探測程式輪流使用真實權重的多份副本，總量
  > 512 MB（R9700）或 > 2 GB（8060S），並先暖機。
- decode kernel 要用從 GGUF 抽出的真實權重，而非隨機資料（分支與 LUT 行為取決於數值）。
- 各變體交錯執行 6–12 輪；回報 min/median/max。
- 在同一個探測程式中，將每個變體的輸出與正式版 kernel 逐位元組比對。
- 自動調校：對一批重複執行計時，最後只同步一次；每次重複都同步會量到啟動（launch）開銷，使選擇產生偏差。

## <a id="thermal"></a>5. 8060S（ROG Flow Z13）的散熱規則

Z13 是平板式筆電：持續功耗上限約 80 W（搭載同款 Ryzen AI Max+ 395 的迷你主機跑 120–160 W），效能
模式只能維持全功率約 20–30 秒。迷你主機的數據（包括已發表的 llama.cpp 與 gufo 結果）不可直接比較，
對受運算限制的 prefill 尤其如此。

| 觀察 | 數字 |
|---|---|
| 27B prefill 4.9k token、512-token 分塊、冷機 | 前 4 塊（約 5.4 s）372–394 tok/s，之後 345 → 327（−15%） |
| 同上，緊接在長時間 decode 之後 | 從一開始就是 310–332 |
| 27B 純 decode 1,800 個 token，閒置 2 分鐘後 | 139 s 內 13.34 → 12.80 tok/s（−4%，部分來自 context 增長）；無斷崖 |
| 不降溫、連續執行多個行程 | 27B plain 在 3–5 分鐘後 13.4 → 5.4–5.8 tok/s（−58%），兩個二進位檔皆同；prefill 也下降 |

規則：短於加速時間窗的測試量到的是加速速度——要註明量的是哪一種，或兩者都回報。比較版本時使用交錯
執行，每次執行前閒置 60–90 秒。40 分鐘基準測試套件的後段不能當作效能退步的證據。為 8060S 計時時，
另一張 GPU 上不得執行任何東西——R9700 那邊的 CPU 負載也會拖慢 APU（某次執行損失 11–18%）。某次
server 對 CLI 的速度檢查在一開始以冷機量 CLI、在數分鐘負載後量 server，回報了錯誤的 −7…−14%
「退步」；該檢查現在會在 server 之前*與*之後都量 CLI（每側取 3 次中位數），容許 5% + CLI 本身的
漂移，漂移超過 8% 即判定失敗，且 server 請求之間等待 15 秒。

## <a id="vram"></a>6. 只用 VRAM 規則與其監控

- 背景取樣器每 2 秒記錄每個 GPU 介面卡上每個引擎行程的專用與共享使用量；報告會標記任何共享超過
  256 MiB（扣除宣告的 pinned 記憶體後）的行程，以及同一介面卡上任何兩個引擎行程重疊的情況。
- 介面卡以 LUID 識別，**而 LUID 在重新開機後會改變**；某次報告以過時的 LUID 篩選，結果找不到任何樣本，
  卻印出「all runs in dedicated VRAM」。現在資料為空時會判定失敗。
- 以 PID 為鍵的宣告檔在 PID 被重複使用時產生誤報；應以 PID + 啟動時間為鍵。
- llama.cpp 的 `llama-perplexity` 在兩次執行中共享使用量峰值達 306 MiB，依此規則這兩次執行的速度
  數據無效（數值本身不受影響）。

## <a id="egpu"></a>7. eGPU 注意事項

開發用的 R9700 接在 USB4 後面：主機連結每個方向約 3.8 GB/s，且 H2D 複製佔滿連結時 kernel 派送會變慢
（[windows-hip.md](windows-hip.md#egpu)）。受影響的有：模型載入時間、RAM/SSD KV 還原時間（每 token
約 14–16 µs）、還原期間其他 slot 的 decode（+17%）、視覺權重串流（權重未常駐時每張圖約 260 ms）。
不受影響的有：所有留在 VRAM 內的工作——prefill、decode、MTP、批次處理。我們沒有針對 eGPU 做最佳化；
直接 PCIe 的使用者應會看到更快的還原。

## <a id="quality"></a>8. 量測品質

- **KL 對照路徑雜訊。** 對有損的修改，將最後一個 token 的分佈與參考比較，並先量出兩條同樣正確的參考
  路徑之間的 KL（分塊 vs 循序 DeltaNet、naive vs WMMA attention）。修改以該雜訊為基準來評判（dense
  27B 的路徑雜訊約 1e-7；MoE 模型因路由接近平手而高達 1e-2）。
- **配對 QA**（350 題：150 題以執行驗證的 Python 輸出預測、100 題多步整數算術、100 題在 17k–70k
  token 的 C++ 原始碼中含誘餌鍵的長 context 鍵／值檢索），greedy，在 `ANSWER:` 前先思考，對不一致的
  配對做 McNemar 精確檢定。只要求答案的版本毫無用處（正確率 5–7%）。
- **Needle**：在 10/50/90% 深度，最長到 256k。
- **Perplexity**（llama.cpp `llama-perplexity`）：使用 219 KB 的中英混合程式碼語料，51 塊 × 2048
  token，逐塊配對。

## <a id="gates"></a>9. 正確性 gate（每次修改都跑）

| Gate | 必須成立的條件 |
|---|---|
| Dense 對照 numpy f32 參考 | batched、step-2、step-q8：KL < 1e-3、top-1 相同、top-10 重疊 ≥ 9；step-f32（精確路徑）：KL < 1e-5；long-attn（WMMA vs naive）、long-gdn（分塊 vs 循序 DeltaNet） |
| MoE 檢查 | 對照 MoE numpy 參考 KL < 5e-3；長 context 路徑比較 KL < 1e-2、top-1 相同、每個 p ≥ 1e-3 的 token 都在雙方的 top-10 中 |
| MTP 精確性 | MTP greedy == plain greedy，兩個模型皆然；草稿數 1、2、5、10；每循環強制 n-gram == plain（MXFP4 冒煙測試：plain == MTP == MTP + n-gram == 強制 n-gram） |
| Kernel 自我檢查 | 多列 GEMV == 單列（所有型別、2–16 列、所有變體、輸出頭）；分組驗證 attention == 逐查詢；每個 GEMM 選項與 MoE tile 皆列不變 |
| Logits 對照前一個 build | 4–5 個提示詞 × 兩種量化，最後一個 token 的 logits 逐位元相同（當修改預期為精確時） |
| Server 套件（dense、MoE） | 各 50 項檢查：端點、server == CLI、並行 == 循序、多輪、**預設環境** |
| 分段 prefill | 8 個同時請求（160–2,489 token）各自 == 單獨執行；27B、MoE 與 MXFP4 × MTP 開／關；MXFP4 C=4 並行 == 單獨 |
| 池／快取 | 逐出與重算、多輪命中率、層還原（RAM、重啟後的 SSD）、系統提示詞檢查點、負載下的還原 |
| 視覺 | 編碼器對照 f32 numpy 參考；載入 mmproj 時 VRAM 不變；圖片提示詞 == 冷啟動參考 |

完整流程需 55–60 分鐘。用慘痛教訓換來的守則：每次執行前刪除預期的輸出檔案（某次二進位檔失敗後留下
舊檔案，被檢查讀成「ok」）；設定 `PYTHONIOENCODING=utf-8`；每個測試 server 都給自己的 SSD 快取目錄；
在比較 server 的 gate 中固定 KV 格式（某次可用 VRAM 較多的參考 server 自動選了 f16，而受測的選了
q8v，結果每項比較都「失敗」）；工具呼叫以名稱 + 參數比較（id 是隨機的）。

## <a id="current-numbers"></a>10. 目前數據（已安裝的原型 build，2026-10-02）

Qwen3.8-27B、R9700、greedy。除非另註，單位為 tok/s。

| | llama.cpp Q4_K_M | **WHIRL Q4_K_M** | llama.cpp MXFP4 | **WHIRL MXFP4** |
|---|---|---|---|---|
| Prefill 2k（2,022 token） | 1,108.9 | 1,721.8–1,724.8 | 1,207.7 | 3,762–3,777 |
| Prefill 8k（8,178） | 1,177.2 | 1,746.5–1,750.9 | 1,279.8 | 3,548–3,557 |
| Prefill 32k（32,751） | 1,074.3 | 1,527.0–1,537.8 | 1,157.4 | 2,782–2,784 |
| Prefill 128k（126,818） | TODO（未量測） | 1,051.5–1,052.3（TTFT 120.6 s） | TODO | 1,531.2–1,531.4（82.8 s） |
| Decode，MTP + n-gram（WHIRL 預設） | N/A | **145.4** | N/A | **159.1** |
| Decode，MTP | 56.9 | 110.4 | 61.6 | 120.3 |
| Decode，**無 MTP**（上限約 38） | 30.9 | 34.3 | 33.0 | 37.6 |
| 16k context 之後的 decode，無 MTP | 29.4 | 32.5 | 31.3 | 35.3 |
| 四位並行使用者，實際時間總和 | 64.1（較舊的執行，MTP 開啟） | 138.7–139.6 | TODO | 190.3–190.7 |
| 四位使用者，decode 迴圈中的穩態 | — | 252.9 | — | 289.8 |
| 新 session 重用 13.1k / 30.5k 系統提示詞，TTFT | TODO | 0.153–0.185 / 0.567–0.608 s（冷啟動 10.30 / 25.6 s） | TODO | 0.085–0.110 / 0.361–0.379 s（冷啟動 4.4 / 12.0 s） |
| 被逐出的 session 還原，TTFT（RAM / 重啟後的 SSD） | TODO | 28.3k：0.590 / 0.654 s；125.1k：2.178 / 2.207 s | TODO | 28.3k：0.62 / 0.71 s |

如何解讀：

- **Prefill（WHIRL**）：CLI、KV f16、MTP 關閉、2 輪交錯。WHIRL *server* 的冷 prefill（預設設定、
  分塊合併）在 Q4_K_M 為 1,560（8k）/ 1,413（30k），MXFP4 為 3,235 / 2,637。llama.cpp 數據來自統一
  server 協定。
- **Decode**：統一 server 協定，19 個提示詞的平均。僅看編輯提示詞時，WHIRL 在 CLI 中以 MTP + n-gram
  可達約 300（Q4_K_M）/ 330（MXFP4）tok/s。
- **無 MTP decode** 受記憶體頻寬限制：每個 token 需讀 14.33 GB 權重，而這張卡串流速度為
  604–626 GB/s，最多約 38 tok/s；WHIRL 的單 token GEMV 達其 97–100%，一個 decode 步驟耗時
  27.74 ms（Q4_K_M）/ 25.15 ms（MXFP4）。要更快的 decode，就必須每次讀過權重時驗證多個 token。
- **並行**：約 1.1k token 的提示詞，生成 256 個 token，`--ctx-per-slot 4096`；實際時間總和包含
  prefill。llama.cpp 的 64.1 是較舊的 `-np 4` 量測，使用相似但不完全相同的腳本。
- **系統提示詞／還原**：預設 server；還原受 eGPU 連結限制（§7）。
- 這些執行中 WHIRL 的每個輸出都通過 MTP/n-gram == plain greedy 與並行 == 單獨的檢查。

### 10.1 長 context（27B Q4_K_M）

| 深度 | Prefill（TTFT） | 純 decode | MTP decode，程式碼續寫 |
|---|---|---|---|
| 16k | 1,395（11.7 s） | 33.3 | 82.8–84.2 |
| 32k | 1,270（27.3 s） | 31.2 | 71.0–72.3 |
| 64k | 1,099（62.8 s） | 28.0 | 61.5–62.4 |
| 128k | 849（149 s） | 23.9 | 47.5–48.9 |

早期量測（f16 KV、單一 slot），在後來的 prefill 工作之前：128k prefill 現在為 1,051–1,052（Q4_K_M）
與 1,531（MXFP4）。每個深度都找到 needle。Server 預設（q8v KV）、MTP 開啟：64k decode 46.79，128k
35.94–36.01。MXFP4 在 256k（q8v、單一 slot）：prefill 863.6 tok/s（TTFT 303.2 s），純 decode 18.86。

### 10.2 MoE 模型（Ornith-1.5-35B-A3B Q4_K_M）

| | WHIRL | llama.cpp ROCm（早期比較） |
|---|---|---|
| CLI prefill 2,022 / 8,178 / 32,751 token（最新 build） | 5,915 / 6,131 / 5,187 | — |
| Prefill，1.1k token 英文摘要（早期） | 5,043 | 3,503 |
| Decode 無 MTP / MTP，短（早期） | 154.6 / 184.9 | 99.6 / 94.8 |
| Decode 無 MTP / MTP，24k（早期） | 124.3 / 130.9 | 89.7 / 84.1 |
| 程式基準測試，MTP + n-gram（CLI 平均） | 約 216 | — |

### 10.3 Radeon 8060S

C++ 引擎的 gfx1151 bring-up（預覽、尚未調校）量測在 [benchmarks.md §14](benchmarks.md#14-radeon-8060s預覽bring-up尚未調校)：
≥ 2k token 的 prefill 是 llama.cpp b11214 的 1.12–1.35×、無 MTP decode 1.06–1.29×、短提示 0.72–0.95×。下表是較舊研究
程式碼樹的數據（降溫後的交錯執行）。

| | 27B 無 MTP | 27B MTP | MoE 無 MTP | MoE MTP |
|---|---|---|---|---|
| WHIRL（降溫後 A/B） | 13.45 | 33.83–34.71 | 79.59–81.75 | 92.48–96.75 |

與 8060S 上的 llama.cpp b11214 相比（早期）：WHIRL 在 prefill 與短 context decode 領先（27B 英文、
MTP：22.4 對 ROCm 16.6 / Vulkan 21.3），但在 WMMA decode-attention 工作之前，llama.cpp Vulkan 在長
context 領先（27B 14k MTP 17.0 對 19.4）；之後 MoE 模型在 24k 達到 59.9，Vulkan 為 55。

## 11. 基準測試集

| 測試集 | 內容 |
|---|---|
| 中英混合程式（主要） | 7 個提示詞——Python LRU cache + pytest、Spring Boot 訂單 API、React hook + 元件、PostgreSQL schema + 報表、修正含中文識別字的 Python、Node.js 重構、閱讀 4k token 的原始碼——各 800 token，思考關閉與開啟 |
| 檔案編輯 | 5 個提示詞，搭配 1.6–2.2k token 的原始碼檔案（重新命名、重構、翻譯註解；其中一個是 CRLF 檔案），輸出整個檔案，`max_tokens` 2500 |
| 長 context | 固定的 2k / 8k / 32k / 128k / 256k 檔案；以真實原始碼建構的 16k–128k needle 與程式碼續寫提示詞 |
| 並行 | 16 個互不重疊、約 1.1k token 的提示詞，各 256 token，C = 1/2/4（過去也測過 8/16） |
| Agent 情境 | 多輪工具使用（`read_file`、`write_file`）、共用的 12–30k token 系統提示詞、逐出壓力、重新啟動 |
| 品質 | 350 題配對 QA、KL 提示詞集（arch 1k、zh-short、4k、12k、24k、code 16k/32k/64k、long 64k/128k）、PPL 語料 |

主要基準測試反映本專案的主要用途：以中文提問，回答含中文說明與註解及英文程式碼。混合文字比純英文
程式碼更難預測（早期 MTP 版本中，純英文程式約 105 tok/s，混合約 86），因此是較保守的選擇。
