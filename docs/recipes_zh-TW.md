[English](recipes.md) | **繁體中文**

# 情境食譜：用 WHIRL 做常見的事

以任務為主的簡短解答：「我想做 X——照這樣做。」這裡提到的每個選項在 WHIRL 0.1.1 都存在；完整參考見
[使用參考](guide/zh-TW/usage.md)，背景說明見 [server.md](guide/zh-TW/server.md) 與
[kv-and-caching.md](guide/zh-TW/kv-and-caching.md)。指令都是 PowerShell，在 `whirl-server.exe` 所在的資料夾
執行。第一次用？請先看[快速上手](quickstart_zh-TW.md)。

範例使用以下檔名（換成你自己的）：

```powershell
$model  = "C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf"   # 推薦的 dense 27B
$mmproj = "C:\models\mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf"        # 只有圖片輸入需要
```

- [1. 接上 agent 或聊天前端](#connect)
- [2. 長時間 agent 工作階段（主 agent + 子 agent）](#agents)
- [3. 長上下文（最多 256k token）](#long-context)
- [4. 多使用者或多 slot](#multi-user)
- [5. 圖片](#images)
- [6. 看懂伺服器 log](#log)
- [7. 疑難排解](#troubleshooting)

## <a id="connect"></a>1. 接上 agent 或聊天前端

用一個好打的模型 id 啟動伺服器：

```powershell
.\whirl-server.exe $model --alias swift-27b --port 8080
```

然後在用戶端的「OpenAI 相容」／「自訂 OpenAI」供應者設定中填入下列值。各產品的欄位名稱不同，值都一樣：

| 欄位（常見名稱） | 值 |
|---|---|
| Base URL／API base／endpoint | `http://127.0.0.1:8080/v1` |
| API key | 任何非空字串（WHIRL 沒有驗證），例如 `none` |
| Model／model id | `--alias` 的值（`swift-27b`）；`GET /v1/models` 會列出它 |
| Context window／最大上下文 | `--ctx-per-slot`（預設 131072），見下方 |
| 供應者類型 | 「OpenAI-compatible」、「OpenAI（自訂 base URL）」之類——不要選 Ollama 或 LM Studio 供應者 |

agent 框架（例如 Hermes agent）、聊天前端（例如 Open WebUI）與編輯器助理（Continue、Cline 等 VS Code
類工具）都一樣：選通用的 OpenAI 相容供應者，照表填寫。如果用戶端在另一台電腦，用 `--host 0.0.0.0` 啟動
伺服器並填這台 PC 的位址——這樣任何連得到它的人都能使用，因為沒有 API key 檢查。

在 PowerShell 檢查：

```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/models | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8080/props    | ConvertTo-Json -Depth 5   # 每個 slot 的上下文、slot 數、alias
```

**上下文長度自動偵測（0.1.1）。** 會探測 llama.cpp 風格 `GET /props` 的用戶端，從
`default_generation_settings.n_ctx` 讀到每個請求的上下文；`GET /v1/models` 以 `meta.n_ctx` 回報同一個值。
兩者都不讀的用戶端要手動填 context window：填 `--ctx-per-slot` 的值。會探測 Ollama／LM Studio 路徑
（`/api/tags`、`/api/v1/models`…）的用戶端會在這些路徑得到 404——這是預期的；把它指向上面的 `/v1` base URL。

**思考（thinking）。** 預設開啟；思考內容放在 `reasoning_content`（串流時為 `delta.reasoning_content`），
答案放在 `content`。每個請求可以這樣改：

| 你要 | 送出下列任一種 |
|---|---|
| 關閉思考 | `"reasoning_effort": "none"`、`"reasoning": {"enabled": false}`、`"chat_template_kwargs": {"enable_thinking": false}` 或 `"enable_thinking": false` |
| 思考短一點 | `"reasoning_effort": "low"`（也可 `minimal`）或 `"medium"`，或 `"reasoning": {"effort": "low"}` |
| 完整思考（預設） | `"reasoning_effort": "high"`（也可 `xhigh`、`max`、`ultra`） |

不認得的 effort 值只會記一行警告並改用預設。無法送額外欄位的用戶端就是思考開啟。

**工具呼叫**支援標準 OpenAI `tools`／`tool_calls`／`role: "tool"` 訊息，串流或非串流皆可；參數值會依各工具的
JSON schema 轉成對應型別。限制：`tool_choice: "none"` 會把工具從提示詞移除，但 `"required"` 或指定函式
**不會強制**（沒有文法約束解碼——由模型自己決定）；不支援 `response_format`（JSON 模式／JSON schema）；
`n` 必須為 1；沒有 `logprobs`；`presence_penalty`／`frequency_penalty` 接受但忽略。細節見
[server.md](guide/zh-TW/server.md#tools)。

## <a id="agents"></a>2. 長時間 agent 工作階段（主 agent + 子 agent）

WHIRL 的伺服器就是為這個情境做的：agent 每回合都重送一段不斷變長的對話（系統提示詞、工具定義、歷史），
有時還同時跑好幾個子 agent。

```powershell
.\whirl-server.exe $model --mmproj $mmproj --alias swift-27b --port 8080
```

這裡相關的預設值：4 個 slot（`-np 4`）、每個請求最多 131,072 token、填滿剩餘 VRAM 的 KV 池、MTP + n-gram
推測解碼、`--decode-min-tps 20`、pinned RAM 層與 64 GiB 的 SSD 層。`--mmproj` 可省略（它多佔約 0.9 GiB 的
pinned RAM，在圖片到來前不佔 VRAM，見[第 5 節](#images)）。

**為什麼之後的回合不到一秒就開始輸出。** 每回合伺服器會把新提示詞和各 slot 已有的內容比對，從共同前綴內
最後一個已存的檢查點接著算，所以只有新的尾段（最後的工具結果和新的使用者訊息）需要 prefill。Qwen3.8 這類
混合架構模型無法從任意 token 接續——DeltaNet 狀態只存在於檢查點（提示詞結尾、生成結尾、`think-open`、長的
系統訊息、和其他工作階段共用的前綴）——所以重用的 token 數會比共同前綴少一點。讓提示詞開頭保持不變：
系統提示詞裡放會變的時間戳記，或工具清單換了順序，都會迫使整段重新 prefill。

**一次真實工作階段**（R9700 上的 Swift-1.5 27B MXFP4-A，Hermes agent 帶 25 個工具，temperature 0.6 取樣，
伺服器全用預設選項）：

| 項目 | 實測 |
|---|---|
| 某回合的上下文 | 約 31k token：31,067 個中有 29,460 個從前綴快取重用，prefill 1,607 個 |
| 首 token 時間（TTFT） | 0.89 s |
| 該回合 decode | 78.6 tok/s（取樣，MTP + n-gram） |
| 另一回合，約 29k 上下文 | 98 tok/s，峰值 112 tok/s（MTP + n-gram 草稿接受率高） |
| 同時 4 個子 agent | 每個約 40–48 tok/s，合計 180–195 tok/s |

**子 agent。** 每個子 agent 是獨立的對話，佔一個 slot。子 agent 通常共用系統提示詞和工具定義；≥ 2048 token
的系統訊息會得到共用的 `system` 檢查點，多個工作階段共有的前綴會得到共用的 `prefix` 檢查點，所以新的子 agent
能跳過這段 prefill（新對話重用 26k token 的系統提示詞：這個模型 TTFT 0.12 s，
[benchmarks.md](guide/zh-TW/benchmarks.md#8-冷-prefill快取前綴與-kv-還原ttft)）。同時超過 4 個請求時，其餘的會排隊；
如果主 agent 加上 4 個以上子 agent 會同時跑，就把 `-np` 調高（最多 16）。slot 越多，每個 slot 每個 cycle
分到的草稿越少，而且超過 4 個 slot 時每個 slot 的前綴檢查點從 2 個變 1 個。

**子 agent 送長提示詞時讓主串流保持順暢：`--decode-min-tps`。** 其他請求在 prefill 長提示詞時，每個串流中的
請求都保持 ≥ N tok/s（預設 20）。實測：一個請求在 25.7k token 上下文串流，同時來了三個 16–19k token 的提示詞
（Swift 27B MXFP4-A）：

| `--decode-min-tps` | 0（關閉） | 10 | **20（預設）** | 30 | 40 |
|---|---|---|---|---|---|
| 它們 prefill 期間，串流請求的速度 tok/s | 3.3 | 12.0 | **23.0** | 30.7 | 40.5 |
| 串流最長停頓，s | 1.22 | 1.13 | **0.47** | 0.46 | 0.48 |
| 3 個提示詞的平均 TTFT，s | 17.8 | 19.3 | **18.6** | 21.3 | 29.0 |
| 同時的 prefill 吞吐量，tok/s | 2791 | 2341 | **1862** | 1551 | 1137 |

只在意總吞吐量的批次工作用 `0`；如果你盯著主 agent 的串流、子 agent 可以等，就用較高的值（30–40）。任何值的
輸出都相同（[server.md](guide/zh-TW/server.md#batching)）。

**RAM 層大小。** 閒置的工作階段會複製到 pinned RAM（再從那裡寫到 SSD），被擠出 VRAM 的工作階段、或重啟前的
工作階段就能直接還原，不必重新 prefill。RAM 層預設大小：從 v0.1.2 起約為系統 RAM 的 1/4（8–32 GB）；v0.1.1
約 9 GB——用 `--kv-ram-mb` 自行指定。一筆項目約每 256 token 13 MiB（q8v KV，dense 模型的預設；f16 為 17 MiB），
再加每個 DeltaNet 檢查點約 150 MiB（啟動時的 `kv tier:` 行會印出你的模型的這兩個數字），所以 30k token 的
工作階段約 1.8–2.6 GiB。給 RAM 層足夠容納你常切換的那幾個工作階段；`--kv-ram-mb 0` 會同時關閉兩個主機層。

```powershell
.\whirl-server.exe $model --alias swift-27b --kv-ram-mb 16384 --kv-ssd-gb 128   # 加大分層
```

**停止與重啟而不丟快取。** 按一次 **Ctrl+C**：排隊中的請求得到 503，執行中的跑完，所有還沒寫到 SSD 的 RAM 層
項目會寫出（最多 10 秒；log 顯示 `shutdown: tier writes done …`）。再按一次 Ctrl+C 或 `taskkill /F` 會跳過這步；
關閉主控台視窗也會正常停止，但 Windows 約 5 秒後就會結束行程。用**相同的模型、執行檔和影響數值的選項**
（KV 格式、MTP 開關）再次啟動：啟動時會掃描 SSD 索引，最近工作階段的下一回合會被還原（log 為
`kv tier: restoring … from SSD`）——這個模型 25.7k token 的工作階段 TTFT 0.73 s。不同建置或不同數值設定寫入的
項目會被忽略。

## <a id="long-context"></a>3. 長上下文（最多 256k token）

```powershell
.\whirl-server.exe $model -np 1 --ctx-per-slot 262144
```

- `--ctx-per-slot` 是單一請求的最長上下文（預設 131,072，最大 262,144）。只有一個使用者時，`-np 1` 把整個
  KV 池留給他。
- KV 格式自動選擇，啟動時會連同原因印出。dense 模型取 **f16 → q8v → q8h** 中第一個放得下所需池大小的：
  `-np 1` 搭預設上下文得到 f16，預設 4 個 slot 得到 q8v，`--ctx-per-slot 262144` 得到 q8h。在我們的成對 QA
  （350 題，上下文最長 70k）中，沒有任何格式和 f16 有可量測的差異；q8h 在 64k–128k 的 decode 慢 3.6–5.2%
  （[kv-and-caching.md](guide/zh-TW/kv-and-caching.md#formats)）。要強制指定，啟動前設 `$env:WHIRL_KV = "q8v"`
  （或 `f16`、`q8`、`q8h`）。
- MoE 模型（Ornith-1.5-35B-A3B）一律用 f16 KV，每 token 22 KiB（27B dense 模型 f16 為 68 KiB），所以預設池就有
  約 414k token，長上下文的 decode 也快得多（16k token 之後 301.5 對 69.5 tok/s，
  [benchmarks.md](guide/zh-TW/benchmarks.md#6-16k-token-context-之後的-decode)）。
- 冷的長提示詞需要時間：Swift 27B MXFP4-A prefill 32k token 約 2,600 tok/s（約 13 秒）；在我們的 needle 測試中，
  27B 模型在 256k 時約 400 tok/s（約 11 分鐘）。之後的回合靠前綴快取與分層就很快。
- **VRAM。** 伺服器會把 KV 池開到填滿剩餘 VRAM，所以行程永遠顯示約 30–31 GiB 專用記憶體
  （[benchmarks.md 第 9 節](guide/zh-TW/benchmarks.md#9-vram)）——這是預期行為，不是洩漏。要把 VRAM 留給其他程式，
  用 `-c N`（token 數）限制池大小。工作管理員裡 8–16 GiB 的「共用 GPU 記憶體」是 pinned RAM 層，不是溢出的 VRAM。

## <a id="multi-user"></a>4. 多使用者或多 slot

```powershell
.\whirl-server.exe $model --host 0.0.0.0 -np 8
```

- `-np N`（1–16，預設 4）是同時服務的請求數；更多的會排隊（log 裡的 `queued (…)`）。所有 slot 共用一個 KV 池：
  slot 隨上下文變長取用頁面，池滿時最久沒用的*閒置* slot 會被逐出——先溢出到 RAM／SSD 層，所以那位使用者
  回來時會被還原。
- 4 個並行使用者的合計吞吐量（約 1.1k token 提示詞、每個生成 256 token、含 prefill）：Swift 27B MXFP4-A
  **198.6 tok/s**、Ornith MoE MXFP4 **396.7 tok/s**、Qwen3.8-27B Q4_K_M **134.1 tok/s**
  （[benchmarks.md 第 7 節](guide/zh-TW/benchmarks.md#7-server-併發)）。每個並行輸出都和同一請求單獨執行時相同。
- 推測解碼的草稿和使用者在同一次驗證中競爭，所以同時 decode 的使用者越多，每人的加速越少；到 16 個使用者時
  MTP 已經沒有幫助。
- `--host 0.0.0.0` 會在所有網路上聆聽且沒有驗證：只在你信任的網路上使用（Windows 防火牆可能會詢問一次）。

## <a id="images"></a>5. 圖片

```powershell
.\whirl-server.exe $model --mmproj $mmproj --alias swift-27b
```

用 OpenAI `image_url` 片段搭配 `data:` URL（base64 PNG／JPEG／…）送圖：

```powershell
$b64  = [Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\pics\chart.png"))
$body = @{ messages = @(@{ role = "user"; content = @(
          @{ type = "text"; text = "這張圖表顯示什麼？" },
          @{ type = "image_url"; image_url = @{ url = "data:image/png;base64,$b64" } }) });
        max_tokens = 800 } | ConvertTo-Json -Depth 8
Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType 'application/json; charset=utf-8' `
  -Body ([Text.Encoding]::UTF8.GetBytes($body)) | ForEach-Object { $_.choices[0].message.content }
```

（提示詞含中文時用 UTF-8 位元組送出，避免 Windows PowerShell 5.1 的編碼問題。）

- 永遠不會去抓 `http(s)` 圖片 URL。本機路徑／`file://` URL 只有加 `--allow-local-images` 才接受。
- 成本：編碼器權重放在 pinned RAM（約 0.9 GiB），圖片到來前不佔 VRAM。預設的 27B 設定 VRAM 很緊，所以每張圖
  會串流權重：在我們的 USB4 eGPU 上每張約 0.26–0.29 s，第一次編碼 0.30–0.54 s
  （[benchmarks.md 第 10 節](guide/zh-TW/benchmarks.md#10-visionswift-mxfp4-a-搭配-f16-mmproj)）。連續
  `--vis-idle-s`（預設 60）秒沒有圖片就會釋放編碼器。
- 同一張圖再出現（之後的回合或其他請求）不會重新編碼：embedding 依內容雜湊快取（`--vis-cache-mb`，預設 1024），
  前綴快取也會重用圖片之後的 KV。
- 支援：Qwen3-VL 風格的 mmproj，F16／BF16；一個請求可多張圖；不支援影片（[vision.md](guide/zh-TW/vision.md)）。

## <a id="log"></a>6. 看懂伺服器 log

log 同時輸出到主控台和 `%LOCALAPPDATA%\whirl\server.log`（`--log-file`）。每行以本地時間和等級 `I`／`W`／`E`
開頭；請求相關的行帶有 `req <id> | slot <n>`。以下是一個 agent 回合的節錄（數值僅為示意，已移除時間戳記）：

```
I req 412 | POST /v1/chat/completions | stream on | 58 messages, 25 tools (tool_choice auto), thinking on
I req 412 | slot 0 | prompt 31067 tok, common prefix 31040, reused 29460 tok (checkpoint 'think-open' @29460), prefill 1607 new
I req 412 | slot 0 | prefill: 1607 tok in 870.0 ms (1847.1 tok/s)
I req 412 | slot 0 | gen: n_gen 100, tg 96.50 t/s, draft acceptance 71.2% (3.85 tok/cycle)
I req 412 | slot 0 | MTP: drafted 610, accepted 434, acceptance rate 71.1%, 3.84 tokens per cycle (154 verify cycles)
I req 412 | slot 0 | n-gram drafts in 41 of 154 cycles
I req 412 | slot 0 | finish_reason tool_calls, completion 592 tok, reasoning 1210 B, content 0 B, tool_calls 1, cache now 31659 tok
I slot 0 | kv tier: spill 31488 tok to RAM (in place, pages 115..123, checkpoints 301 MiB, 418.0 MiB in all; RAM tier 6.10 / 9.00 GiB)
I batch | slots busy 1/4 (decode 1, prefill 0) | 154 cycles, 1.00 slots/cycle, 4.96 rows/cycle, 3.96 drafts/cycle | aggregate tg 92.4 tok/s, prefill 321 tok/s over 6.4 s
I kv tier: entry 3f2a9c0d81b7e655 (31488 tok) written to SSD: 418 MiB in 160 ms (file 1802 MiB)
I kv tier | RAM 6.10 / 9.00 GiB in 4 entries, SSD 12.40 GiB in 9 entries | spills 37 (9120 MiB), restores RAM 2 / SSD 1 (3410 MiB), …
```

| 行 | 代表什麼 |
|---|---|
| `POST … \| stream on \| 58 messages, 25 tools (tool_choice auto), thinking on` | 用戶端送了什麼；有指定時後面會加上 `reasoning_effort …` |
| `prompt N tok, common prefix L, reused R tok (checkpoint 'K' @P), prefill M new` | 提示詞 N 個 token，其中 L 個和 slot 已有的內容相同；從位置 P 的已存檢查點 K 接續，所以只計算 M = N − R 個 token。`reused 0 tok (no usable checkpoint)` 代表整段 prefill（新對話或提示詞開頭變了） |
| 檢查點種類 | `prompt-end`／`gen-end`（上一次提示詞／回覆的結尾）、`think-open`（用戶端沒把上一輪思考內容送回時，仍能命中快取）、`system`／`prefix`（和其他工作階段共用） |
| `prefill: M tok in X ms (Y tok/s)` | 新段落花的時間；加上排隊時間大致就是首 token 時間 |
| `gen: n_gen …, tg … t/s, draft acceptance …% (x tok/cycle)` | 每約 100 token 或 5 秒的進度：目前 token 數、decode 速度、草稿被接受的比例、每個驗證 cycle 產出的 token 數（1.0 = 推測解碼沒有加速） |
| `MTP: drafted D, accepted A, acceptance rate …, x tokens per cycle (C verify cycles)` | 該請求最終的推測解碼摘要 |
| `n-gram drafts in X of Y cycles` | 用到提示詞查找（n-gram）草稿的 cycle 數——模型從上下文抄文字時（改程式碼、引用工具輸出）會很高 |
| `finish_reason …, completion … tok, reasoning … B, content … B, tool_calls …, cache now … tok` | 請求如何結束；`cache now` 是 slot 留給下一回合的內容 |
| `batch \| slots busy b/n (decode d, prefill p) \| …` | 忙碌時每約 5 秒一行：使用中的 slot、每 cycle 平均 slot／列／草稿數、合計 decode 與 prefill tok/s |
| `batch \| decode floor: …` | decode 保底（[第 2 節](#agents)）穿插 decode 與 prefill 的次數 |
| `kv tier: spill … tok to RAM (…; RAM tier a / b GiB)` | 一個閒置工作階段複製到 RAM 層（`in place` = 只複製變動部分） |
| `kv tier: entry … written to SSD` | 該項目現在重啟後也還在 |
| `kv tier: restoring … from RAM\|SSD`／`restored … in X ms (Y GB/s)` | 工作階段從分層還原，而不是重新 prefill |
| `kv tier \| RAM a/b GiB in n entries, SSD …` | 伺服器轉為閒置時印出的摘要 |
| `VRAM use: …` 與 `context … tokens per request, pool … tokens` | 啟動時：VRAM 用在哪裡、每請求上下文與池大小；KV 格式那一行會說明選擇原因 |
| `GET /api/tags -> 404 (probe for another server type; …)` | 用戶端在找 Ollama／LM Studio API。從 0.1.1 起每個這類路徑只在 `I` 等級記一次（0.1.0 每個請求印一行 `W`）；無害——把用戶端指向 `/v1` |

回應 JSON 也帶有 `usage.cached_tokens` 和 llama.cpp 風格的 `timings` 物件，數字相同
（[server.md](guide/zh-TW/server.md#logging)）。

## <a id="troubleshooting"></a>7. 疑難排解

| 症狀 | 原因與解法 |
|---|---|
| 「Windows 已保護您的電腦」，或程式被封鎖 | 執行檔沒有程式碼簽章：按**其他資訊 → 仍要執行**，或在解壓縮前對官方 zip 執行 `Unblock-File`；智慧型應用程式控制可能直接封鎖——見 [Windows 安全性提示](windows_security_zh-TW.md) |
| 新模型第一次啟動卡一陣子 | 一次性的 kernel 自動調校：MXFP4 檔幾秒，27B Q4_K_M 檔約 1.5–2 分鐘（會有訊息說明）；結果快取在 `%LOCALAPPDATA%\whirl` |
| 啟動後一直等、不載入 | 另一個 WHIRL 行程佔著 GPU（每張 GPU 一個行程；第二個最多等 30 分鐘，`WHIRL_GPU_WAIT`）。先停掉另一個 |
| 結束代碼 **2** | 命令列錯誤（未知選項、缺模型路徑、`--host` 不是這台電腦的位址）：`.\whirl-server.exe --help` |
| 結束代碼 **3** | GPU／驅動問題：安裝 AMD Software: Adrenalin Edition 26.8.1 或更新版；用 `.\whirl.exe devices` 檢查 |
| 結束代碼 **4** | 模型檔問題：檔案不存在、不是 GGUF、架構不是 `qwen35`／`qwen35moe`、不支援的 tensor 型別、下載不完整 |
| 結束代碼 **5**（GPU 記憶體不足） | 關掉其他使用 GPU 的程式（兩個大型 GPU 行程會讓 Windows 把兩者都移到慢速共用記憶體）；用 `-c 131072` 限制池大小；縮短 `--ctx-per-slot`；減少 slot（`-np 2`）；`$env:WHIRL_KV = "q8v"`（或 `q8h`） |
| 結束代碼 **6**（連接埠被佔用） | 另一個程式在聽那個連接埠；改用 `--port 8081`，或用 `Get-Process -Id (Get-NetTCPConnection -LocalPort 8080).OwningProcess` 找出是誰 |
| 用戶端說「找不到模型」或列表是空的 | 用 `/v1` base URL、OpenAI 相容供應者類型，模型 id 填 `--alias` |
| 用戶端太早截斷長對話 | 它不知道上下文長度：把 `--ctx-per-slot`（預設 131072）填進它的 context window |
| 每回合都很慢（log 沒有 `reused`） | 提示詞開頭每回合都在變（系統提示詞有時間戳記、工具順序改變），或設了 `WHIRL_NO_PREFIX_CACHE` |

所有結束代碼與訊息：[使用參考](guide/zh-TW/usage.md#exit-codes)。
