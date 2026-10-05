[English](../../usage.md) | **繁體中文**

# 使用參考：`whirl` 與 `whirl-server`

本頁列出 WHIRL 0.1.1 的每一個指令、選項、環境變數與伺服器端點。內容依發行版執行檔的內建說明
（`whirl --help`、`whirl <指令> --help`、`whirl help env`、`whirl-server --help`）整理；若有出入，以你手上
執行檔的說明文字為準。

第一次使用？請先看[快速上手](../../quickstart_zh-TW.md)。伺服器的設計背景見 [server.md](server.md)，快取見
[kv-and-caching.md](kv-and-caching.md)。

- [1. 兩支程式](#programs)
- [2. `whirl chat`](#chat)
- [3. `whirl bench`](#bench)
- [4. `whirl serve` / `whirl-server`](#serve)——選項、[端點](#endpoints)、[停止](#stop)
- [5. `whirl devices`、`selftest`、`seqtest`、`vis-encode`](#other)
- [6. 結束代碼與錯誤訊息](#exit-codes)
- [7. 環境變數](#env)
- [8. WHIRL 會寫入的檔案](#files)

## <a id="programs"></a>1. 兩支程式

| 程式 | 用途 |
|---|---|
| `whirl.exe` | 命令列工具：`chat`、`bench`、`serve`、`devices`、`selftest`、`seqtest`、`vis-encode`、`help` |
| `whirl-server.exe` | OpenAI 相容的 HTTP 伺服器；與 `whirl serve` 完全相同 |

```
whirl <指令> [參數]
whirl --help | -h            一般說明
whirl --version | -V         版本、kernel 所針對的 GPU 架構、驅動程式需求
whirl help 指令              該指令的選項（等同 whirl 指令 --help）
whirl help env               每個環境變數與一行說明
whirl-server --help | --version
```

兩支程式都只接受 `general.architecture` 為 `qwen35`（dense）或 `qwen35moe`（混合專家，MoE）的 GGUF 檔，
見[支援的模型](../../README_zh-TW.md#支援的模型)。CLI 一律 greedy 生成；伺服器另外支援取樣（temperature、top-p 等）。

**第一次使用某個模型。** prefill kernel 會針對每個模型檔與 GPU 調校一次（stderr 會有提示）：27B Q4_K_M 約
1.5～2 分鐘，MXFP4 檔只要幾秒。結果會快取起來（[第 8 節](#files)），之後直接開始。每一種候選組態算出的位元都
相同，只有速度不同。

**一張 GPU 一個行程。** WHIRL 執行時會持有該 GPU 的鎖。同一張 GPU 上的第二個 WHIRL 行程會等前一個結束
（預設最多 30 分鐘），而不是共用 VRAM——因為一張 GPU 上同時有兩個大行程時，Windows 會把兩者都移到很慢的共享
記憶體（[windows-hip.md](windows-hip.md#wddm-demote)）。見 `WHIRL_GPU_WAIT` / `WHIRL_GPU_SHARE`。

## <a id="chat"></a>2. `whirl chat`

```
whirl chat MODEL.gguf "提示" | @提示檔 [選項]
```

對單一提示產生回覆。提示會套上模型的聊天樣板（user 回合，接著開啟 assistant 回合：思考開啟時接 `<think>\n`，
關閉時接空的 think 區塊）。文字邊生成邊輸出，遇到回合結束 token 或達到 `--max-tokens` 時停止。模型若帶 MTP
層，decode 會走推測解碼（MTP + n-gram 草稿），輸出與一般 greedy decode 完全相同。結尾的摘要會列出 prefill 與
decode 速度；開 MTP 時另列驗證回合數與草稿接受率。

| 選項 | 說明 |
|---|---|
| `"提示"` / `@檔案` | 提示文字，或 `@路徑` 指向存放提示的 UTF-8 檔（長提示用） |
| `--max-tokens N` | 生成的 token 數（預設 64） |
| `--ctx N` | context 大小：提示 + 生成的 token（預設 8192；也可用 `WHIRL_MAX_CTX`） |
| `--no-think` / `--think` | 關閉 / 開啟思考（預設開啟；也可用 `WHIRL_THINK=0\|1`） |
| `--no-stream` | 結束時一次印出回覆，不邊生成邊輸出 |
| `--raw` | 不套聊天樣板：提示文字原樣 tokenize |
| `--tokens ID,ID,...` | 直接給 token id，不給提示（不套樣板） |
| `--out FILE` | 把最後一個提示位置的 logits（f32，每個詞彙一個值）寫進 FILE；會停用 MTP |
| `--mmproj MMPROJ.gguf` | 視覺編碼器（Qwen3-VL 形式的 mmproj，F16 / BF16），`--image` 需要它 |
| `--image IMAGE` | 放在提示文字前面的圖片；可重複指定 |
| `--device SPEC` | GPU：`r9700`（預設）、`8060s`、索引，或名稱 / gfx 架構的子字串（也可用 `WHIRL_DEVICE`） |
| `-h`、`--help` | 說明 |

範例：

```powershell
.\whirl.exe chat C:\models\Qwen3.8-27B-UD-Q4_K_M.gguf "用兩句話說明 TCP 慢啟動（slow start）。" --max-tokens 400
.\whirl.exe chat C:\models\model.gguf @C:\work\long_prompt.txt --max-tokens 2000 --ctx 65536 --no-think
.\whirl.exe chat C:\models\model.gguf "這張圖裡有什麼？" --mmproj C:\models\mmproj-F16.gguf --image C:\pics\cat.png
```

## <a id="bench"></a>3. `whirl bench`

```
whirl bench MODEL.gguf [選項]
```

模型只載入一次，然後量測：(1) 每個長度的 prefill 速度——以中英混合的程式文字為輸入，8k 以內先暖機再取 2 次
最佳，超過 8k 量 1 次；(2) 在一個簡短的程式問題之後，各 decode 模式的速度。所有 decode 模式的 token 序列
必須完全相同，不同會被回報。

| 選項 | 說明 |
|---|---|
| `--prefill N,N,...` | prefill 長度（token，預設 `2048,8192,32768`） |
| `--decode N` | 每個 decode 模式生成的 token 數（預設 256） |
| `--prompt 文字` / `@檔案` | decode 量測用的提示（預設：一個中英混合的程式問題） |
| `--modes M,M,...` | decode 模式：`mtp-ngram`（MTP + n-gram 草稿，預設路徑）、`mtp`（只用 MTP）、`plain`（不開 MTP）；預設三種都量 |
| `--no-think` / `--think` | decode 提示的思考關閉 / 開啟（預設開啟） |
| `--device SPEC` | GPU，同 `chat` |

```powershell
.\whirl.exe bench C:\models\model.gguf --prefill 2048,8192 --decode 256 --modes mtp-ngram,plain
```

我們如何比較引擎（交錯執行、該報告什麼）見 [benchmarking.md](benchmarking.md)；實測結果見
[benchmarks.md](benchmarks.md)。

## <a id="serve"></a>4. `whirl serve` / `whirl-server`

```
whirl-server MODEL.gguf [選項]
whirl serve  MODEL.gguf [選項]      （同一支程式）
```

載入模型，提供 OpenAI 相容 API：以 `--parallel` 個請求 slot 做 continuous batching，VRAM 內有前綴快取，
閒置 session 另有主記憶體（RAM）/ SSD 分層快取。

| 選項 | 說明 |
|---|---|
| `--host ADDR` | 監聽位址。預設 `127.0.0.1`（只有這台電腦）。`0.0.0.0` 會監聽所有網路：任何連得到這台電腦的人都能使用——伺服器沒有身分驗證 |
| `--port N` | TCP port（預設 8080）。若已被占用，伺服器會在載入模型之前以代碼 6 結束 |
| `--alias NAME` | `/v1/models` 回報的模型 id（預設：去掉 `.gguf` 的檔名） |
| `--device D` | GPU：`r9700`（預設）、`8060s`、裝置索引，或名稱 / gfx 子字串 |
| `-np`、`--parallel N` | 同時處理的請求 slot 數（continuous batching），1～16，預設 4 |
| `-c`、`--ctx N` | 共用 KV 池的大小（token）。slot 依需要取用分頁；池滿時，閒置 slot 的前綴快取依最久未使用（LRU）逐出。預設：權重與緩衝區之後剩下的全部 VRAM 減 768 MiB（MoE：1.5 GiB） |
| `--ctx-per-slot N` | 單一請求的最長 context（預設 min(池大小, 131072)；最多 262144） |
| `--mtp-drafts N` | 每回合固定的 MTP 草稿數，1～10（預設：依模型類型由成本模型決定） |
| `--decode-min-tps N` | decode 保底速度：其他請求在 prefill 時，每個串流中（decode 中）的請求至少維持 N tok/s；做法是縮短 prefill forward、穿插 decode cycle（預設 20；`0` = 關閉，prefill forward 不受限）。任何 N 的輸出都相同（[server.md](server.md#batching)） |
| `--kv-ram-mb N` | 前綴快取的主記憶體層，單位 MiB 的 pinned 記憶體（預設：實體記憶體的 1/4，至少 8 GiB 或一個完整長度 session（若更大；27B 模型約 9 GiB），最多 32 GiB，且不超過啟動時可用記憶體的一半；64 GB 的電腦為 16 GiB；整合式 GPU 預設關閉）。啟動日誌會印出選定的大小與原因。閒置 session 會複製到這裡，下次直接還原而不必重新 prefill。`0` 會關閉兩個 host 層 |
| `--kv-ssd-dir PATH` | SSD 層目錄（預設 `%LOCALAPPDATA%\whirl\kvcache`） |
| `--kv-ssd-gb N` | SSD 層容量上限，GiB（預設 64；`0` = 不用 SSD 層） |
| `--mmproj FILE` | 視覺編碼器（Qwen3-VL 形式的 mmproj GGUF，F16 / BF16）。`image_url` 內容（base64 PNG / JPEG 等的 `data:` URL）會變成圖片 token。權重放在 pinned 主記憶體，收到圖片之前不占 VRAM |
| `--vis-idle-s N` | N 秒沒有圖片就釋放視覺編碼器（預設 60） |
| `--vis-mode M` | `auto`（預設）、`resident`（編碼器權重常駐 VRAM）、`stream`（逐層串流） |
| `--vis-cache-mb N` | 依內容雜湊快取圖片 embedding 的主記憶體容量，MiB（預設 1024） |
| `--allow-local-images` | 也接受本機檔案路徑 / `file://` URL 當圖片來源（預設關閉；`http(s)` 圖片網址一律不抓） |
| `--log-file PATH` | log 檔（預設 `%LOCALAPPDATA%\whirl\server.log`）；log 也會印在主控台 |
| `-h`、`--help` / `-V`、`--version` | 說明 / 版本 |

範例：

```powershell
# 27B 模型、4 個 slot、預設分層快取
.\whirl-server.exe C:\models\Qwen3.8-27B-UD-Q4_K_M.gguf --port 8080

# 單一使用者、長 context、不用 SSD 層
.\whirl-server.exe C:\models\model.gguf -np 1 --ctx-per-slot 262144 --kv-ssd-gb 0

# 開啟圖片輸入
.\whirl-server.exe C:\models\model.gguf --mmproj C:\models\mmproj-F16.gguf
```

記憶體提醒：預設設定下，伺服器會為主記憶體層 pin 住約四分之一的主記憶體（8～32 GiB，且不超過啟動時可用記憶體的一半；64 GB 的電腦為 16 GiB；指定 `--mmproj` 時再加約 0.9 GiB），
SSD 最多用到 64 GiB。記憶體或磁碟空間較少的電腦，可用 `--kv-ram-mb` / `--kv-ssd-gb` 縮小或關閉分層快取。

### <a id="endpoints"></a>端點

base URL `http://127.0.0.1:8080/v1`。任何 API key 都接受（沒有身分驗證）。

| 端點 | 說明 |
|---|---|
| `POST /v1/chat/completions` | messages、串流（SSE）或不串流、工具 / 工具呼叫、思考（`reasoning_content`）、圖片（需 `--mmproj`） |
| `POST /v1/completions` | 原始提示，不套聊天樣板 |
| `GET /v1/models` | 已載入的那一個模型（id = `--alias`）；`meta.n_ctx` 為每個 slot 的 context |
| `GET /health` | 忙碌時也會立刻回應；回報忙碌狀態與佇列長度 |
| `GET /props`（也接受 `/v1/props`） | 唯讀，llama.cpp server 形式的子集，供會自動偵測 context 長度的客戶端使用：`default_generation_settings.n_ctx`（每個 slot 的 context）、`default_generation_settings.model` 與 `model_alias`（= `--alias`）、`total_slots`、`model_path`（只有檔名，絕不含目錄）、`modalities`、`build_info` |
| `GET /version` | `{"version":"0.1.1","name":"whirl"}` |
| `OPTIONS`（CORS preflight） | 支援 |

LM Studio（`/api/v1/models`）與 Ollama（`/api/tags`、`/api/show`、`/api/version`）的原生端點不模擬
（客戶端偵測到它們就會改用 WHIRL 沒有的 API）：一律回 404，且每個路徑只在第一次被探測時記一行 `I` 級
日誌，不再每次請求都記警告。這類客戶端請改指向上面的 OpenAI 相容 base URL。

請求參數：`temperature`、`top_p`、`top_k`、`min_p`、`seed`、`max_tokens` / `max_completion_tokens`、`stop`、
`stream`、`tools`、`tool_choice`、`chat_template_kwargs.enable_thinking`（或最上層的 `enable_thinking`）、
`reasoning_effort`、`preserve_thinking`。沒有給取樣參數時套用模型卡建議值。`temperature` 為 0 時，輸出與
`whirl chat` 逐 token 相同。`presence_penalty` / `frequency_penalty` 會接受但忽略；`n` 必須為 1；不支援
`logprobs`、`response_format`，`tool_choice: "required"` 也不會強制。回應另含 `usage.cached_tokens` 與
llama.cpp 形式的 `timings` 物件。細節見 [server.md](server.md#sampling)。

思考與 reasoning effort 可用下列任一種寫法（依此順序套用，後者覆蓋前者）：
`chat_template_kwargs.{enable_thinking, reasoning_effort}`、OpenRouter / OpenAI Responses 形式的物件
`"reasoning": {"effort": "...", "enabled": true|false}`、最上層 `enable_thinking`、最上層 `reasoning_effort`。
effort 值（不分大小寫）：

| 送出的值 | 實際使用的 effort |
|---|---|
| `xhigh`、`high`、`max`、`ultra` | xhigh（沒給時的預設） |
| `medium` | medium |
| `low`、`minimal` | low |
| `none`（或 `"reasoning": {"enabled": false}`） | 關閉思考 |

不認得的 effort 值不會讓請求失敗：伺服器記一行警告並改用預設值。effort 只會改變 chat template 支援它的
模型（Qwen3.8）的提示；思考已被關閉時，給有效的 effort 也不會把思考重新打開。

```powershell
$body = '{"messages":[{"role":"user","content":"2+3 等於多少？"}],"temperature":0,"max_tokens":200}'
Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

### <a id="stop"></a>停止伺服器

按 **Ctrl+C**（或 Ctrl+Break，或關閉主控台視窗），伺服器會正常收尾：

1. 新的與排隊中的請求回應 HTTP 503；
2. 執行中的請求跑完；
3. 已完成的 session 複製到主記憶體層，主記憶體層中尚未寫入 SSD 的項目立刻寫入——最多 10 秒
   （log 會出現 `shutdown: tier writes done …`）。

因此下次啟動時，最近的對話會從 SSD 還原，不必重新 prefill。再按一次 Ctrl+C 會立即結束。關閉主控台視窗時，
不論是否做完，Windows 約 5 秒後都會結束行程。直接強制結束行程（工作管理員、`taskkill /F`）會跳過以上全部
步驟；伺服器沒有關機用的端點。

## <a id="other"></a>5. `whirl devices`、`selftest`、`seqtest`、`vis-encode`

| 指令 | 用途 |
|---|---|
| `whirl devices` | 列出驅動程式回報的 AMD GPU（索引、名稱、架構、記憶體、運算單元），以及這個建置有沒有該 GPU 的 kernel。先用它確認驅動程式與 GPU |
| `whirl selftest MODEL.gguf [--device SPEC]` | 用模型本身的權重做 GPU kernel 的逐位元自我檢查：int8 與 f32 GEMV 誤差、多 token GEMV == 單 token GEMV、prefill GEMM 在所有組態下不變、MoE token tile 不變、分組與逐 query 的 decode attention。印出 `selftest: ok` 或 FAIL（結束代碼 1） |
| `whirl seqtest MODEL.gguf [--decode N] [--device SPEC]` | 在兩個提示上比對多請求（伺服器）路徑與單請求執行：分段 prefill、批次 decode 列、以固定草稿做批次驗證。每個 token 都必須相同；印出 `seqtest: ok` 或 FAIL（結束代碼 1）。`--decode N`：每個序列生成的 token 數（預設 24） |
| `whirl vis-encode MMPROJ.gguf IMAGE [OUT.f32] [--reps N] [--mode auto\|resident\|stream]` | 診斷用：用視覺編碼器編碼一張圖並回報時間；`OUT.f32` 存投影後的 embedding；`--reps N` 重複編碼 N 次 |

## <a id="exit-codes"></a>6. 結束代碼與錯誤訊息

常見的失敗會印出白話訊息（發生了什麼、該怎麼做），並以不同的結束代碼結束：

| 代碼 | 意義 | 例子與處理方式 |
|---|---|---|
| 0 | 成功 | |
| 1 | 其他錯誤；`selftest` / `seqtest` FAIL | |
| 2 | 指令列有誤 | 未知選項、缺少 `MODEL.gguf`、提示 + `--max-tokens` 超過 `--ctx`、`--host` 不是這台電腦的位址 |
| 3 | GPU / 驅動程式問題 | 沒有 AMD GPU、沒有這張 GPU 架構的 kernel、`amdhip64_7.dll` 不存在或太舊（請安裝 AMD Software: Adrenalin Edition 26.8.1 或更新版）、GPU 忙碌中 |
| 4 | 模型檔問題 | 找不到檔案、不是 GGUF、不支援的架構（只支援 `qwen35` / `qwen35moe`）、不支援的 tensor 型別（例如 NVFP4）、檔案不完整 |
| 5 | GPU 記憶體不足 | 權重或 KV 快取放不下：試試較小的 `--ctx`、`WHIRL_KV=q8v`，或減少伺服器 slot |
| 6 | 伺服器 port 已被占用 | 已有其他程式在 `--host`:`--port` 監聽（載入模型前就會檢查）：改用別的 `--port` |

執行檔在需要時才載入 AMD GPU runtime（`amdhip64_7.dll`，隨顯示卡驅動程式安裝），並在啟動時先檢查，所以驅動
程式不存在或太舊時會回報代碼 3，而不是跳出 Windows「找不到 DLL」的對話框。其餘部分（包括 C++ runtime）都是
靜態連結。

## <a id="env"></a>7. 環境變數

兩支程式都以 `WHIRL_<名稱>` 讀取每個變數。在
PowerShell 中先用 `$env:WHIRL_KV = "q8v"` 設定再啟動程式。**一般使用完全不需要設定**；預設值就是測試過、
最快的路徑。標示 *（A/B）* 或 *（診斷）* 的變數是給實驗與量測用的。

### 7.1 GPU 與載入

| 變數 | 說明 |
|---|---|
| `WHIRL_DEVICE=SPEC` | 使用的 GPU：`r9700`（預設）、`8060s`、索引，或名稱 / gfx 子字串（= `--device`） |
| `WHIRL_HIP_DEVICE=N` | 裝置索引，略過裝置比對與「一張 GPU 一個行程」的鎖 |
| `WHIRL_GPU_SHARE=1` | 不等待同一張 GPU 上的其他 WHIRL 行程 |
| `WHIRL_GPU_WAIT=S` | 等待其他 WHIRL 行程釋放 GPU 的秒數（預設 1800） |
| `WHIRL_KV=auto\|f16\|q8\|q8h\|q8v` | KV 快取格式。`auto`（預設）：放得下就用 f16；dense 模型依序退到 q8v、q8h；MoE 一律 f16。見 [kv-and-caching.md](kv-and-caching.md#formats) |
| `WHIRL_PREFILL_BATCH=N` | 每次 forward 的 prefill 列數（預設 4096，最多 16384） |
| `WHIRL_MAX_CTX=N` | `chat` 的預設 context 大小（= `--ctx`；預設 8192） |
| `WHIRL_CODE_OBJECT=FILE` | 開發用：從這個 code object 載入 GPU kernel，取代內建的 |
| `WHIRL_MOE_FP8=0` | 專家為 MXFP4 的 MoE 模型：專家 prefill 改用 f16 activation，不用 fp8（預設 fp8，較快的路徑——Ornith MXFP4 在 2k token 約 11.7k 對 8.6k tok/s）。只影響 prefill |
| `WHIRL_MOE_MXW=0` | 專家為 MXFP4 的 MoE 模型：改用通用的 MXFP4 專家 decode kernel，不用整塊（whole-block）kernel（預設整塊）。影響 decode / 驗證 |
| `WHIRL_EMBD_HOST=0` | token embedding 表放在 VRAM（預設放 pinned 主記憶體；server 會把它的大小併入 pinned 宣告，讓 Shared Usage 監控扣除） |
| `WHIRL_TUNE_COLD=1` | 自動調校：每次計時前先清快取 *（診斷）* |
| `WHIRL_TUNE_MASK=BITS` | 自動調校：候選 GEMM 組態的遮罩 *（診斷）* |

### 7.2 思考

| 變數 | 說明 |
|---|---|
| `WHIRL_THINK=0\|1` | `chat` 與 `bench` 的思考關閉 / 開啟（預設開啟；= `--no-think` / `--think`） |

### 7.3 推測解碼

輸出永遠等於一般 greedy decode（取樣時則是相同的分佈）。背景說明見
[speculative-decoding.md](speculative-decoding.md)。

| 變數 | 說明 |
|---|---|
| `WHIRL_MTP=0` | 一般 decode，不用 MTP 草稿 |
| `WHIRL_MTP_DRAFTS=N` | 每回合固定的 MTP 草稿數，1～10（預設：成本模型，dense 最多 8、MoE 1） |
| `WHIRL_MTP_ADAPT=auto\|N` | 草稿數由成本模型決定，或取 ceil(最近接受數 + N) |
| `WHIRL_MTP_PMIN=P` | 草稿機率低於 P 時結束這串草稿（在 `WHIRL_MTP_NMIN` 個草稿之後才生效） |
| `WHIRL_MTP_NMIN=N` | `WHIRL_MTP_PMIN` 生效前一定會產生的草稿數 |
| `WHIRL_MTP_BATCH_DRAFTS=d1,d2,...` | 1、2、… 個 slot 同時 decode 時，每回合的草稿上限（伺服器） |
| `WHIRL_MTP_Q4=0` | 保留 MTP 區塊的 Q6_K / Q8_0 矩陣（預設：改用 Q4_K 副本，只用於草稿） |
| `WHIRL_DRAFT_HEAD=q4` | 草擬頭改用 Q4_K，不用 2-bit |
| `WHIRL_MTP_FULLHEAD=1` | 草稿使用完整的輸出頭 |
| `WHIRL_DRAFT_VOCAB=off\|64k\|48k\|檔案\|N` | MTP 草擬頭只涵蓋依頻率挑出的詞彙子集。預設：內嵌在執行檔中的 64k 子集，只用於有 2-bit 草擬頭、詞表 248,320 的 dense qwen35 模型（輸出不變）。`off` = 完整草擬頭；`48k` = exe 旁的 `draft_vocab\subset_48k.bin`；檔案 = uint32 little-endian token id；N = 前 N 列 |
| `WHIRL_NGRAM=0` | 不用 n-gram（prompt lookup）草稿 |
| `WHIRL_NGRAM_MIN=N` | n-gram 草稿的最短比對後綴（預設 3） |
| `WHIRL_NGRAM_MAX=N` | 每回合最多的 n-gram 草稿數（預設 15） |
| `WHIRL_NGRAM_FORCE=1` | 每個 n-gram 提案都採用 *（A/B）* |
| `WHIRL_NGRAM_DEBUG=1` | 每個 decode 回合都記錄到 stderr *（診斷）* |

### 7.4 伺服器

| 變數 | 說明 |
|---|---|
| `WHIRL_NO_PREFIX_CACHE=1` | 關閉前綴快取 |
| `WHIRL_PREFIX_CACHE_VERIFY=1` | 每次快取命中後重新 prefill 並比對 *（診斷）* |
| `WHIRL_SERVE_CKPTS=N` | 每個 slot 的前綴檢查點數（預設：1 個 slot 時 4、最多 4 個 slot 時 2、更多時 1） |
| `WHIRL_SYS_MIN=N` | 至少 N 個 token 的 system 訊息會有自己的檢查點（預設 2048，0 = 關閉） |
| `WHIRL_SYS_CKPTS=N` | 保留在 VRAM 的共用前綴檢查點數（預設 2） |
| `WHIRL_SYS_LCP=0` | 不在多個 session 共同的前綴處建立檢查點 |
| `WHIRL_CKPT_HOST=1` | 前綴檢查點放在 pinned 主記憶體，不放 VRAM |
| `WHIRL_POOL_RESERVE_MB=N` | 決定 KV 池大小時保留不用的 VRAM（預設 768，MoE 1536） |
| `WHIRL_PREFILL_CHUNK=N` | 合併後單次 prefill forward 的最多列數（1024 的倍數，預設 2048） |
| `WHIRL_SEG_PREFILL=0` | 每個請求各自 prefill，不把多個請求放進同一次 forward |
| `WHIRL_GATHER_MS=MS` | 收集一波新請求的等待時間窗（預設 30，0 = 關閉） |
| `WHIRL_DECODE_MIN_TPS=N` | 其他請求 prefill 時，每個串流請求的 decode 保底速度（= `--decode-min-tps`，預設 20，0 = 關閉） |
| `WHIRL_GDN_REPLAY=0` | DeltaNet 驗證改用快照組，不重播保留的列 |
| `WHIRL_SNAP_SETS=N` | 遞迴狀態快照組的最少數量（搭配 `WHIRL_GDN_REPLAY=0`） |
| `WHIRL_SLOT_DRAFTS=1` | 依預期接受率在 slot 之間分配草稿預算 |
| `WHIRL_TIMING_RESET=0` | 跨請求保留 MTP 計時表（不建議） |
| `WHIRL_KV_RAM_MB=N` | 主記憶體層，MiB（= `--kv-ram-mb`） |
| `WHIRL_KV_SSD_DIR=PATH` | SSD 層目錄（= `--kv-ssd-dir`） |
| `WHIRL_KV_SSD_GB=N` | SSD 層容量上限，GiB（= `--kv-ssd-gb`） |
| `WHIRL_KV_TIER_MIN=N` | 複製到 host 層的最小 session，單位 token（預設 2048） |
| `WHIRL_KV_SSD_DELAY_MS=MS` | 主記憶體層項目延後多久再寫入 SSD（預設 2000） |
| `WHIRL_MMPROJ=FILE` | 視覺編碼器（= `--mmproj`） |
| `WHIRL_VIS_IDLE_S=N` | N 秒沒有圖片就釋放視覺編碼器（= `--vis-idle-s`） |
| `WHIRL_VIS_CACHE_MB=N` | 圖片 embedding 快取，MiB（= `--vis-cache-mb`） |
| `WHIRL_VIS_CKPT_MIN=N` | 圖片結束位置 ≥ N token 時保留共用檢查點（預設 1024，0 = 關閉） |

### 7.5 視覺

| 變數 | 說明 |
|---|---|
| `WHIRL_VIS_MODE=auto\|resident\|stream` | 編碼器權重常駐 VRAM 或逐層串流（= `--vis-mode`） |
| `WHIRL_VIS_DUMP=PREFIX` | 把每張圖的 embedding 寫到 `PREFIX.<n>.f32` *（診斷）* |
| `WHIRL_VIS_DUMP_RGB=FILE` | 寫出前處理後的影像 *（診斷）* |
| `WHIRL_VIS_PRE=FILE` | 改為前處理這張圖 *（診斷）* |
| `WHIRL_VIS_PROF=1` | 編碼器各階段計時 *（診斷）* |

### 7.6 數值與速度開關 *（A/B）*

保留給 A/B 測試與數值比對的替代路徑；預設值就是測試過的路徑。「位元相同」的替代路徑算出的位元一樣，只差速度。

| 變數 | 說明 |
|---|---|
| `WHIRL_FP8=0` | MXFP4 prefill 改用 f16 activation，不用 fp8 |
| `WHIRL_FP8_MASK=BITS` | 使用 fp8 activation 的矩陣乘法類別（預設 7） |
| `WHIRL_G8T=0` | fp8 GEMM 改用列優先版本，不用 fragment 排列版本（位元相同） |
| `WHIRL_GDN_WMMA=0\|1` | DeltaNet prefill 區塊用 f16 WMMA（MXFP4 預設開啟） |
| `WHIRL_FFN_H16=0\|1` | GEMM 以 f16 輸出給逐元素運算（MXFP4 預設開啟） |
| `WHIRL_Q4_RELAXED=1` | 其他模型也套用 MXFP4 速度模式的 prefill 開關 |
| `WHIRL_ACT_FUSE=0` | prefill 不融合 activation（位元相同） |
| `WHIRL_GEMMH=0` | 不用 f16 輸出的 prefill GEMM（位元相同） |
| `WHIRL_GEMMHQ=1` | attention 投影也用 f16 輸出的 GEMM |
| `WHIRL_ATTN_KX=0` | prefill attention 不用 K-exchange kernel（位元相同） |
| `WHIRL_ATTN_KG=0` | prefill attention 不用依 GQA 分組的 kernel（位元相同） |
| `WHIRL_GDN_SEQ=1` | DeltaNet prefill 改用逐步計算，不用分塊掃描 |
| `WHIRL_GDN_V0=1` | 逐列的 DeltaNet decode 步驟 kernel（數值相同） |
| `WHIRL_NAIVE_ATTN=1` | 參考用 attention 路徑 |
| `WHIRL_NO_FUSE=1` | 不用融合的 decode kernel（參考路徑） |
| `WHIRL_FLOAT_GEMV=1` | GEMV 用 f32，不用 int8（參考路徑） |
| `WHIRL_NO_GRAPH=1` | 一般 decode 步驟不用 HIP graph |
| `WHIRL_GEMV_MAX=N` | 多 token GEMV 一次最多的 token 數 |
| `WHIRL_GEMV_R=nt:r,...` | 多 token GEMV 每 block 列數的覆寫 |
| `WHIRL_GEMV_W=nt:v,...` | 多 token GEMV kernel 變體的覆寫 |
| `WHIRL_GEMV_WH=nt:v,...` | 多 token GEMV 變體的覆寫（f16 輸出） |
| `WHIRL_GV_GROUP=0` | 不合併同輸入的 GEMV launch |
| `WHIRL_GV_NMAX=N` | 同輸入 GEMV 一組的最大數量 |
| `WHIRL_MOE_BN=32\|64` | 分組專家 GEMM 的 token tile |
| `WHIRL_DBG=BITS` | 1 = 不合併 gdn_abconv、2 = 純量 split attention、4 = 每個 attention 群組一個 query |
| `WHIRL_ATTN_WIDE=0` | 驗證 attention 每群最多 16 欄（`attn_wsplit1`），不用最多 32 欄的 `attn_wsplit2` |

### 7.7 診斷

| 變數 | 說明 |
|---|---|
| `WHIRL_TOKENIZE_ONLY=1` | 印出提示的 token id 後停止 |
| `WHIRL_PRINT_IDS=1` | 印出生成的 token id 與其 FNV-1a 雜湊 |
| `WHIRL_PROFILE=1` | 各類運算的 GPU 時間；會扭曲速度數字（`whirl bench`：每個 prefill 長度各一份；伺服器：1 或 2） |
| `WHIRL_TRACE_TPS=N` | 每 N 個 token 印出該區間的 tok/s（stderr） |
| `WHIRL_DUMP_LOGITS=FILE` | （不開 MTP）把每個位置 ≥ `WHIRL_DUMP_FROM` 的下一 token logits 以 f16 列寫出 |
| `WHIRL_DUMP_FROM=N` | `WHIRL_DUMP_LOGITS` 寫出的第一個提示位置 |
| `WHIRL_DUMP_MOE=FILE` | 每個 prefill MoE 區塊選到的專家（i32） |
| `WHIRL_TRACE_ND=1` | 記錄每個 decode 回合的草稿數 |
| `WHIRL_LOOP_LOG=1` | 記錄每圈各階段的時間（伺服器） |
| `WHIRL_TIER_VERIFY=1` | 每次 host 層寫出 / 還原都逐位元組驗證（伺服器） |
| `WHIRL_TIER_MIN_GAIN=N` | 只有能省下至少 N 個 token 時才從 host 層還原（預設 512） |

## <a id="files"></a>8. WHIRL 會寫入的檔案

| 路徑 | 內容 |
|---|---|
| `%LOCALAPPDATA%\whirl\tune4-<模型檔名>-<檔案大小>…txt` | 每個模型檔與 GPU 調校好的 prefill GEMM 組態。刪掉就會重新調校 |
| `%LOCALAPPDATA%\whirl\kvcache\` | 伺服器前綴快取的 SSD 層（`--kv-ssd-dir`，上限 `--kv-ssd-gb`） |
| `%LOCALAPPDATA%\whirl\server.log` | 伺服器 log（`--log-file`） |

執行檔旁邊不會寫入任何東西，也不會安裝任何系統層級的元件。
