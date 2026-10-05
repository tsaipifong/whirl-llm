[English](../../benchmarks.md) | **繁體中文**

# WHIRL 發行版效能評測（v0.1.3）— Radeon AI PRO R9700 上 WHIRL 對 llama.cpp

本頁所有 WHIRL 數字都用 v0.1.3 發行包（`whirl.exe`／`whirl-server.exe`，候選版 1，其執行檔與正式版逐位元相同；SHA-256 見 §1）量測。版本對照表中的 v0.1.0（代理工作階段那幾列另有 v0.1.2）在同一天、同一套量測腳本下用各自的發行包重量，所以版本之間是同條件比較。llama.cpp 為 **b11214**（ROCm，commit `2ebd9ae62`），與 v0.1.0 評測相同。兩邊用同樣的提示、同樣的 context 長度、greedy 解碼，GPU 上一次只有一個模型程序。

**WHIRL 領先不多的地方（請先看）：**

- **不用推測解碼的純 decode** 兩邊都受記憶體頻寬限制；dense 模型上 WHIRL 只快 1.13–1.14×。WHIRL 大部分的 decode 領先來自 MTP + n-gram 推測解碼。
- **Qwen3.8 Q4_K_M 的 prefill** 領先比 MXFP4 檔少：96k 為 1.51×、256k token 為 1.44×，Swift-1.5 MXFP4 是 2.28× 與 1.74×。Q4_K_M 冷的 26k token 系統提示只快 1.34×（§8）。
- **dense 模型在 16k context 之後的 decode** 只有 llama.cpp 最快模式的 1.17×（Swift-1.5）與 1.33×（Qwen3.8 Q4_K_M）（§6）。
- **推測解碼的最差情況**（128k token 文件之後寫全新內容，沒有可照抄的內容）只比純 decode 快 1.56×（§5）。

## 1. 測試環境

| | |
|---|---|
| GPU | AMD Radeon AI PRO R9700（gfx1201，RDNA 4，32 GB GDDR6），**以 USB4 外接（eGPU）** |
| 主機 | ASUS ROG Flow Z13（GZ302EA）：AMD Ryzen AI MAX+ 395（16 核／32 緒），128 GB LPDDR5X-8000，其中 Windows 可用 63.6 GB |
| 作業系統／GPU 驅動 | Windows 11 Home 26H2（build 26300.9457）；AMD Software: Adrenalin Edition 26.8.1（驅動 32.0.31041.1004） |
| WHIRL | `whirl-0.1.3-windows-x64` 候選版 1（執行檔與正式版逐位元相同）；`whirl.exe` SHA-256 `af7201bcbfab8257c53837b2cecb3d67e191802c1d0b4bf7716ab04844f907af`、`whirl-server.exe` `8c2feba05929496fa81d30c6526e637930ac6f49aa5de7ae5293dad7b7c9e519` |
| 舊版 WHIRL | 0.1.0 與 0.1.2 發行包（同一台機器、同一套腳本、同一天） |
| llama.cpp | b11214 ROCm Windows 版（`llama-server`、`llama-bench`），`HIP_VISIBLE_DEVICES=1`、`GGML_CUDA_NO_PINNED=1` |
| 量測日期 | 2026-10-05 |

**eGPU 說明：** R9700 接在 USB4 後面（主機連線每方向約 3.8 GB/s）。這會影響模型載入時間，以及主機 RAM／SSD 的 KV 還原；prefill 與 decode 都在 VRAM 內，不受影響。直接插 PCIe 的系統，§8 的還原時間應該會更快。

### 模型

| 簡稱 | 檔案 | 類型 | 量化 |
|---|---|---|---|
| Swift MXFP4-A（dense） | `Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`（14.7 GiB） | dense 27B，含 MTP head | MXFP4，輸出 head Q6_K（我們發佈並推薦的量化） |
| Ornith MXFP4（MoE） | `Ornith-1.5-35B-A3B-MXFP4.gguf`（18.4 GiB） | MoE，總參數 35B，每 token 約 3B active，含 MTP head | MXFP4 專家（WHIRL 自己發佈的量化） |
| Qwen3.8-27B Q4_K_M（dense，參考） | `Qwen3.8-27B-UD-Q4_K_M.gguf`（15.3 GiB，unsloth） | dense 27B，含 MTP head | 標準 Q4_K_M（unsloth UD） |

主要比較對象是 Swift 與 Ornith；Qwen3.8 Q4_K_M 列出來供參考。

## 2. 方法

### 2.1 規則

- **一次只跑一個量測項目。** 每個項目先確認 R9700 狀態為 OK、取得機器的 R9700 鎖，執行 `whirl bench` 或啟動一個 server 並等 `/health` 回應，量測後正常關閉 server（Ctrl+C，讓 KV 層寫完），釋放鎖、再確認一次 GPU 狀態，最後把結果附加到同一個結果檔。
- **兩邊都是 greedy 解碼。** 請求送 `temperature 0`；llama-server 另加 `--top-k 1`。兩個引擎收到的請求內容完全相同；llama-server 以 `--reasoning off` 執行。7 題寫程式提示、系統提示與還原測試會送 `chat_template_kwargs: {enable_thinking: false}`。WHIRL 的 server 在請求沒指定時預設開 thinking，所以 16k context 之後的 decode（§6）與 4 人並發（§7）對每個 WHIRL 版本都改送同樣的 `enable_thinking: false` 重量；llama.cpp 那邊原本就以 `--reasoning off` 執行。
- **採用哪一次：** 每個 case、模型、執行檔只採用**最後一次**（一次量測的所有列共用同一個時間戳）；只有 4 人並發測試（§7）是刻意重複量，採**所有次數的平均**（每個版本、每個模型 3 次，llama.cpp 1 次）。量測時標為錯誤或警告的列不採用。
- **每次都用新的 SSD 層：** WHIRL server 每次執行都有自己的 `--kv-ssd-dir`，不會還原到前一次的提示（只有還原測試依設計重用同一個目錄）。
- **輸出相同：** 各版 WHIRL 的 greedy 輸出用 hash 比對（`whirl bench` 用它自己印的 hash；server 用生成文字的 SHA-1，tool call 也算在內，但不含每次隨機的 `id`）。本頁所有版本對照的輸出都相同。
- **量測不含 WHIRL 第一次執行時的 GEMM 自動調校**；server 的計時從啟動完成之後開始。
- **倍數 = WHIRL / llama.cpp**，llama.cpp 取該列**最快**的模式（plain、MTP、MTP + n-gram），並寫出是哪一個模式。時間（TTFT）的倍數是 llama.cpp 時間 / WHIRL 時間。
- 「**no MTP**」表示完全不用推測解碼的純 decode。「N/A」= llama.cpp 沒有對應功能。

### 2.2 每項測試量什麼

| 測試 | 工具 | 說明 |
|---|---|---|
| Prefill（CLI） | `whirl bench --prefill N --decode 8 --modes plain` | 8,192／32,768／65,536／98,304／131,072 token，KV f16。8k、32k：暖機後取 2 次中最好；64k：暖機後 1 次；96k、128k：計時 1 次 |
| Prefill（server，長文件） | `whirl serve`，一個客戶端 | q8h KV，`-np 1 --ctx-per-slot 262144`；不重複的文件 61.4k／123.5k／180.8k／249k token（192k 檔是 256k 檔在文件邊界截斷），`max_tokens 32`；tok/s = `timings.prompt_per_second` |
| 主要 decode 情境 | 兩個 server，一個客戶端 | v0.1.0 的 7 題 `bench_zhcode` 提示（中文提問，中文說明加英文程式碼），thinking 關閉，`max_tokens 800`；tok/s = `timings.predicted_per_second`，每輪取 7 題平均，3 輪取中位數 |
| 推測解碼的各種情況 | `whirl bench`／`whirl serve` | 最適用：回答會重複輸入內容的檔案編輯提示；主要：代理工作階段重播（91 個請求）；長文件問答與最差：123.5k token 文件之後，分別問一個大量引用文件的問題（512 token）、寫一篇全新的短篇小說（沒有可照抄的內容，1,024 token）。每項都和同一個 case 不開推測解碼比 |
| 16k 之後的 decode | 兩個 server，一個客戶端 | v0.1.0 的 16,354 token 程式碼提示之後生成 128 token（`ignore_eos`） |
| 4 人並發 | 兩個 server，一個客戶端 | 4 個 slot × 4,096 context；4 個不同的約 1.1k token 提示同時送出，各 256 token（`ignore_eos`）；總吞吐 = 生成的 token 數 / 牆鐘時間（含 prefill） |
| 系統提示 TTFT | 兩個 server，一個客戶端 | TTFT = `max_tokens 1` 請求的牆鐘時間；26.4k token 系統提示冷啟動，再用同一個系統提示開新對話（warm）。WHIRL `--kv-ram-mb 16384`，llama-server `--cache-ram 16384` |
| SSD 還原 | WHIRL server | 25.7k token 的 session，正常重啟並沿用同一個 `--kv-ssd-dir`，量下一輪的 TTFT |
| 長文問答 | WHIRL server，q8h | 128k／256k 文件之後回答 1,024 token；粗略檢查：回答有提到被問的函式（不是正式 needle 測試） |
| 代理工作階段 | WHIRL server | 錄下來的 coding agent 工作階段逐個請求重播（OpenAI `tools`／`tool_calls`）：decode tok/s、TTFT、prefix cache 命中 |
| 多人 | WHIRL server，`-np 4` | 啟動 log 的 KV 池容量；4 個短提示同時送出；3 個子代理（每個約 17k token 的提示）prefill 期間的 decode 保底 |

### 2.3 llama.cpp 設定

- `llama-server --device ROCm0 -ngl 99 -fa on -b 2048 -ub 1024`（MoE `-ub 2048`，8k–32k 最快）`--no-webui --load-mode none --jinja --reasoning off --top-k 1`，KV f16。
- case 的 WHIRL server 參數對應：`-np N` → `-np N`；`--ctx-per-slot N` → `-c N×np`；沒有指定時 `-np 1 -c 131072`；`--kv-ram-mb N` → `--cache-ram N`。
- 模式：plain；MTP = `--spec-type draft-mtp`；MTP + n-gram = `--spec-type draft-mtp,ngram-mod`；兩者都加 `--spec-draft-n-min 0 --spec-draft-p-min 0.3`，`--spec-draft-n-max` 為 3（dense）／MTP 2、MTP + n-gram 1（Ornith），取自 v0.1.0 探測的最佳值。
- llama.cpp 欄標「v0.1.0」的列沿用 v0.1.0 評測的 b11214 數字（同一個 build、同樣的旗標，prefill 用 `llama-bench`）；16k 之後的 decode、4 人並發與系統提示這幾列，是 v0.1.3 用和 WHIRL 完全相同的請求重量的（§6–§8）。

WHIRL 用預設值執行：`whirl serve MODEL`（4 個 slot，自動決定 KV 格式與池大小），加上上面各 case 的參數；no MTP = `WHIRL_MTP=0 WHIRL_NGRAM=0`。

## 3. Prefill

v0.1.0 頁面的圖沒有為 v0.1.3 重畫，以表格為準。

### 3.1 各自的 bench 工具（KV f16）

| 提示 token | Swift MXFP4-A WHIRL | llama.cpp | 倍數 | Ornith MXFP4 WHIRL | llama.cpp | 倍數 | Qwen3.8 Q4_K_M WHIRL | llama.cpp | 倍數 |
|---|---|---|---|---|---|---|---|---|---|
| 8,192 | 3,469 | 1,338 | **2.59×** | 11,258 | 4,637 | **2.43×** | 1,731 | 1,223 | **1.42×** |
| 32,768 | 2,907 | 1,174 | **2.48×** | 8,633 | 3,778 | **2.29×** | 1,570 | 1,086 | **1.45×** |
| 65,536 | 2,388 | 1,002 | **2.38×** | 6,507 | 3,069 | **2.12×** | 1,398 | 939.8 | **1.49×** |
| 98,304 | 2,021 | 886.7 | **2.28×** | 5,184 | —¹ | — | 1,266 | 837.3 | **1.51×** |
| 131,072 | 1,757 | 791.3 | **2.22×** | 4,311 | 2,239 | **1.93×** | 1,158 | 752.0 | **1.54×** |

llama.cpp：`llama-bench` b11214，每個長度取最快的 micro-batch（v0.1.0 評測，見該頁 §2.3）；96k 是 v0.1.3 README 採用的 b11214 數字。¹ Ornith 在 96k 與 256k token 沒有量 llama.cpp。

### 3.2 長文件經 WHIRL server（q8h KV）

| 提示 | Swift MXFP4-A | Ornith MXFP4 | Qwen3.8 Q4_K_M |
|---|---|---|---|
| 61.4k token | 2,162 | 5,671 | 1,290 |
| 123.5k token | 1,533 | 3,674 | 1,032 |
| 180.8k token | 1,212 | 2,771 | 876.6 |
| 249k token | 969.8 | 2,135 | 742.2 |
| 256k，llama.cpp（llama-bench b11214） | 556.1，KV f16（**1.74×**） | —¹ | 516.4，KV q8_0（**1.44×**） |

llama.cpp 跑 Swift MXFP4 256k 用的是 f16 KV（放得下，31.3 GB）；Qwen3.8 Q4_K_M 用 f16 會溢到共用記憶體，所以改用 q8_0 KV。

### 3.3 v0.1.0 → v0.1.3（Swift MXFP4-A）

| Prefill | v0.1.0 | v0.1.3 | 變化 |
|---|---|---|---|
| 8k，f16 KV（bench） | 3,273 | 3,469 | +6.0% |
| 32k，f16 KV（bench） | 2,605 | 2,907 | +11.6% |
| 96k，f16 KV（bench） | 1,653 | 2,021 | +22.3% |
| 128k，f16 KV（bench） | 1,407 | 1,757 | +24.8% |
| 128k，q8h KV（server） | 1,353 | 1,533 | +13.3% |
| 192k，q8h KV（server） | 1,054 | 1,212 | +15.1% |
| 256k，q8h KV（server） | 832.5 | 969.8 | +16.5% |

提示越長進步越多，因為 v0.1.3 的 prefill attention kernel（`attn_kg`，[kernels.md](kernels.md#flash)）讓同一組的 query head 共用每一次 key／value 讀取；輸出與舊 kernel 逐位元相同。

## 4. Decode — 主要情境：中文提問、中文說明、英文程式碼

| 模型 | WHIRL MTP+n-gram（預設） | WHIRL **no MTP** | llama.cpp **plain（no MTP）** | llama.cpp 最快（模式） | WHIRL 預設 / llama.cpp 最快 | WHIRL no MTP / llama.cpp plain |
|---|---|---|---|---|---|---|
| Swift MXFP4-A（dense） | **112.4** | 38.1 | 33.4 | 60.8（MTP） | **1.85×** | **1.14×** |
| Ornith MXFP4（MoE） | **257.6** | 177.4 | 118.8 | 118.8（plain） | **2.17×** | **1.49×** |
| Qwen3.8-27B Q4_K_M（dense） | **97.3** | 34.8 | 30.9 | 56.0（MTP + n-gram） | **1.74×** | **1.13×** |

7 題 × 800 token 的平均，3 輪取中位數。llama.cpp 為 v0.1.0 的數字（b11214）。MoE 模型上 llama.cpp 的 MTP 比它的純 decode 慢，所以對照的是 plain。與 v0.1.0 比（同一天、同一套腳本）：Swift 109.2 → 112.4（+2.9%），Qwen3.8 99.7 → 97.3（−2.4%）；兩者輸出都相同。

## 5. 推測解碼：最適用、主要與最差情況（Swift MXFP4-A）

| | 最適用：檔案編輯（內容重複） | 主要：coding agent 工作階段 | 長文件問答（大量引用，128k 之後） | 最差：128k 之後寫全新內容（沒有可照抄的內容） |
|---|---|---|---|---|
| Decode tok/s，MTP + n-gram | 174.4 | 92.9 | 87.6 | 39.4 |
| Decode tok/s，不開推測解碼 | 39.1 | 32.8 | 25.3 | 25.2 |
| 加速倍數 | **4.47×** | **2.83×** | **3.47×** | **1.56×** |
| 每個 cycle 的 token 數 | 6.19 | — | 6.45 | 1.76 |

開與不開推測解碼的輸出逐位元相同。主要 = 代理工作階段重播（91 個請求，12k → 65k token）；重播不記錄每個 cycle 的 token 數。中文寫程式短提示（`whirl bench`，512 token）：120.7 對 39.6 tok/s（3.05×，每個 cycle 4.00 token）。

| v0.1.0 → v0.1.3，MTP + n-gram | v0.1.0 | v0.1.3 | 變化 |
|---|---|---|---|
| 檔案編輯 | 157.3 | 174.4 | +10.9% |
| 中文寫程式短提示 | 110.7 | 120.7 | +9.1% |
| 代理工作階段 1（91 個請求） | 81.5 | 92.9 | +13.9% |
| 128k 之後的長文件問答 | 63.7 | 87.6 | +37.5% |

## 6. 16k token context 之後的 decode

| 模型 | WHIRL MTP+n-gram（預設） | llama.cpp plain | llama.cpp MTP | llama.cpp MTP + n-gram | WHIRL / llama.cpp 最快 |
|---|---|---|---|---|---|
| Swift MXFP4-A（dense） | **71.4** | 31.2 | 60.8 | 60.9 | **1.17×**（MTP + n-gram） |
| Ornith MXFP4（MoE） | **316.4** | 106.9 | 104.7 | 89.7 | **2.96×**（plain） |
| Qwen3.8-27B Q4_K_M（dense） | **80.7** | 29.4 | 60.6 | 60.7 | **1.33×**（MTP + n-gram） |

兩邊都關閉 thinking（§2.1）。同一批請求中 16,354 token 提示的 prefill：Swift 3,156 對 1,285 tok/s（2.46×），Ornith 9,770 對 4,184（2.33×），Qwen3.8 1,596 對 1,179（1.35×），都是對 llama.cpp plain。v0.1.0 → v0.1.3（同一天）：Swift 70.7 → 71.4（+0.9%），Ornith 313.5 → 316.4（+0.9%），Qwen3.8 76.5 → 80.7（+5.5%），輸出相同。v0.1.0 頁面的 16k 數字（Ornith 301.5、Swift 69.5）是用另一套量測腳本量的，不能直接比較。

## 7. Server，4 人並發

| 模型 | WHIRL MTP+n-gram（預設） | llama.cpp plain | llama.cpp MTP | llama.cpp MTP + n-gram | WHIRL / llama.cpp 最快 |
|---|---|---|---|---|---|
| Swift MXFP4-A（dense） | **182.1** | 64.0 | 54.9 | 56.4 | **2.84×**（plain） |
| Ornith MXFP4（MoE） | **381.1** | 176.8 | 107.7 | 129.0 | **2.16×**（plain） |
| Qwen3.8-27B Q4_K_M（dense） | **122.4** | 57.1 | 53.3 | 50.1 | **2.14×**（plain） |

總吞吐 tok/s，含 prefill，兩邊都關閉 thinking（§2.1）。WHIRL：每個模型 3 次的平均；同一天的 v0.1.0（各 3 次）：Swift 173.9（v0.1.3 +4.7%）、Ornith 377.5（+1.0%），輸出相同。llama.cpp：每個模式 1 次。b11214 的推測解碼模式不會隨並發使用者數擴展；WHIRL 每個 slot 都維持 MTP + n-gram（批次 verify）。

## 8. 首 token 時間：系統提示與 SSD 還原

| 模型 | 請求 | WHIRL TTFT（秒） | llama-server TTFT（秒） | llama.cpp / WHIRL |
|---|---|---|---|---|
| Swift MXFP4-A | 26.4k 系統提示，冷 | 9.276 | 21.768 | **2.35×** |
| Swift MXFP4-A | 新對話，同一個系統提示（warm） | 0.108 | 0.384 | **3.56×** |
| Swift MXFP4-A | 伺服器重啟後的 25.7k session（從 SSD 還原） | 0.613 | N/A | N/A |
| Ornith MXFP4 | 26.4k 系統提示，冷 | 3.082 | 6.667 | **2.16×** |
| Ornith MXFP4 | 新對話，同一個系統提示（warm） | 0.098 | 0.182 | **1.86×** |
| Ornith MXFP4 | 伺服器重啟後的 25.7k session（從 SSD 還原） | 0.318 | N/A | N/A |
| Qwen3.8 Q4_K_M | 26.4k 系統提示，冷 | 17.515 | 23.555 | **1.34×** |
| Qwen3.8 Q4_K_M | 新對話，同一個系統提示（warm） | 0.121 | 0.369 | **3.05×** |
| Qwen3.8 Q4_K_M | 伺服器重啟後的 25.7k session（從 SSD 還原） | 0.672 | N/A | N/A |

- *Warm*：WHIRL 重用它的系統提示檢查點，llama-server 重用快取的前綴（兩邊都快取了 26,403 token）。
- *SSD 還原*：server 正常關閉（Ctrl+C）後沿用同一個 `--kv-ssd-dir` 重新啟動，session 的下一輪從 WHIRL 的 SSD 層還原。llama.cpp 沒有自動的持久化 KV 快取，所以這一列是 N/A。
- 這裡 llama.cpp 只跑 plain：回答只有 1 個 token，推測解碼用不上。
- WHIRL 的冷 TTFT 包含對提示跑一次 MTP 區塊。

## 9. 長上下文（q8h KV，WHIRL server，一個使用者）

| | Swift MXFP4-A | Ornith MXFP4 | Qwen3.8 Q4_K_M |
|---|---|---|---|
| 128k 之後的 decode tok/s（回答 1,024 token） | 62.5 | 214.0 | 58.3 |
| 256k 之後的 decode tok/s（回答 1,024 token） | 35.1 | 115.6 | 33.5 |
| llama.cpp 256k 之後，plain（llama-bench `tg64@d262144`） | 16.8，KV f16（**2.09×**） | —¹ | 10.5，KV q8_0（**3.19×**） |
| 長文問答粗略檢查（128k／256k） | ✓／✓ | ✓／✓ | ✓／✓ |

`--ctx-per-slot 262144`，q8h KV（int8 的 key 與 value；query 與 key 先做 Hadamard 旋轉）。粗略檢查是問文件中的某個函式，✓ = 回答有提到它（不是正式的 needle 測試）。llama.cpp 那一列是在 262,144 token 的 context 之後純 decode 64 token，沒有推測解碼。

## 10. Coding agent 工作階段（Swift MXFP4-A，WHIRL server）

| 重播 | v0.1.0 | v0.1.2 | v0.1.3 | 變化（v0.1.3 / v0.1.0） |
|---|---|---|---|---|
| 工作階段 1（scrapy，91 個請求，12k → 65k token），平均 decode tok/s | 81.5 | — | 92.9 | +13.9% |
| 工作階段 4（aiohttp，102 個請求，14k → 102k），平均 decode tok/s | 79.2 | — | 95.2 | +20.1% |
| 工作階段 3 到第 143 個請求（共 144 個），平均 decode tok/s | 64.9 | 65.0 | 81.7 | +25.9% |
| 工作階段 3 第 143 個請求本身（127.9k token） | 53.7 | 53.5 | 75.2 | +40.0% |

錄下來的 coding agent 工作階段，連同 tools 與 tool call 逐個請求重播；每一版的 prefix cache 命中都是提示 token 的 98–99%，每一版的輸出都相同。

## 11. 一個 server 多個使用者（Swift MXFP4-A）

| | v0.1.0 | v0.1.3 | 變化 |
|---|---|---|---|
| KV 池容量，預設 4 個 slot（token） | 210,688 | 226,304 | +7.4% |
| 4 個短提示同時送出：整批完成時間（秒） | 9.57 | 8.62 | −9.9% |
| ……總吞吐 tok/s | 213.9 | 237.6 | +11.1% |
| 3 個子代理 prefill 期間主串流的 tok/s | 1.08 | 9.63 | ×8.9 |
| ……最差 1 秒視窗的 token 數 | 1 | 5 | |
| 3 個子代理先到、主請求 0.5 秒後才到：主請求 TTFT（秒） | 17.256 | 16.205 | −6.1% |

子代理：3 個各約 17k token 的提示。兩版輸出相同。

## 12. 實際指令

```bat
:: WHIRL CLI
whirl bench MODEL.gguf --prefill 8192 --decode 8 --modes plain
whirl bench MODEL.gguf --prefill 512 --decode 512 --prompt @zh_code.txt

:: WHIRL server（每次執行都用自己的 SSD 層目錄）
whirl serve MODEL.gguf --port 8099 --log-file LOG --kv-ssd-dir RUN\kvcache [-np 1 --ctx-per-slot 262144] [--kv-ram-mb 16384]
    （q8h KV：設 WHIRL_KV=q8h；no MTP：設 WHIRL_MTP=0 與 WHIRL_NGRAM=0）

:: llama.cpp server
set HIP_VISIBLE_DEVICES=1& set GGML_CUDA_NO_PINNED=1
llama-server -m MODEL.gguf --port 8099 --device ROCm0 -ngl 99 -fa on -b 2048 -ub 1024 --no-webui --load-mode none ^
    --jinja --reasoning off --top-k 1 -np 1 -c 131072 [--cache-ram 16384]
    [--spec-type draft-mtp --spec-draft-n-max N --spec-draft-n-min 0 --spec-draft-p-min 0.3]
    [--spec-type draft-mtp,ngram-mod --spec-draft-n-max N --spec-draft-n-min 0 --spec-draft-p-min 0.3]
```

量測腳本（PowerShell 的工作執行器，加上只用標準函式庫的 Python 客戶端）不在本 repository；它記錄每個請求的計時，每次執行的 server log 與完整命令列都保存在量測機器上。

## 13. v0.1.3 沒有重量的項目

VRAM 用量、vision 編碼、88 與 2,048 token 的 prefill、同一份文字經 server 的 prefill，以及檔案編輯情境與 llama.cpp 的對照，這次都沒有重量；它們的 v0.1.0 數字在 [v0.1.0 頁面](https://github.com/tsaipifong/whirl-llm/blob/v0.1.0/docs/guide/zh-TW/benchmarks.md)。v0.1.3 把 token embedding 放到 pinned 主記憶體，VRAM 用量應該比 v0.1.0 低。
