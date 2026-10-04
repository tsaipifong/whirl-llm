[English](../en/server.md) | **繁體中文**

# OpenAI 相容伺服器

> **狀態。** 本文描述本儲存庫中的 C++ 伺服器（`whirl-server.exe`，或 `whirl serve`；原始碼在 `src/server/`）。
> 所有選項、環境變數與端點列在[使用參考](usage.md#serve)。數據：R9700（32 GB，USB4 eGPU）。

**對誰有幫助：** 想把 OpenAI 用戶端（Open WebUI、Cline、Continue、`openai` Python 套件）指向
WHIRL 的使用者；以及想了解連續批次處理（continuous batching）、推測解碼與前綴快取在混合架構模型
server 中如何互動的 server 作者。

## 1. 啟動

```
whirl serve MODEL.gguf --port 8080
```

Base URL 為 `http://127.0.0.1:8080/v1`；接受任何 API key。模型只載入一次並常駐。第一次載入新的模型
檔案時，會自動調校 prefill（提示詞預填）的 GEMM（數十秒；依模型與 GPU 快取在磁碟上）。

| 旗標 | 意義 | 預設 |
|---|---|---|
| `--host` / `--port` | 綁定位址 / 連接埠 | `127.0.0.1` / `8080` |
| `--parallel N` | 同時服務的 slot 數（連續批次處理），最多 16 | 4 |
| `--ctx N` | 共用 KV 池的大小，單位 token | 其他緩衝區配置後剩下的全部 VRAM，扣掉一段保留量 |
| `--ctx-per-slot N` | 每個請求的上下文上限 | 131,072（最大 262,144） |
| `--mtp-drafts N` | MTP 草稿上限 1–10 | dense：自動，上限 8；MoE：1 |
| `--decode-min-tps N` | 其他請求 prefill 時，每個串流請求的 decode 保底速度（[§6](#batching)）；0 = 關閉 | 20 |
| `--kv-ram-mb N` | pinned RAM KV 層大小；0 會停用 RAM 與 SSD 層 | 實體記憶體的 1/4，範圍 [max(8 GiB, 一個完整 f16 session + 檢查點), 32 GiB]，且不超過啟動時可用記憶體的一半；64 GB 的電腦為 16 GiB；整合式 GPU 預設關閉 |
| `--kv-ssd-dir` / `--kv-ssd-gb N` | SSD 層目錄 / 大小上限；0 GB 停用 SSD | 每位使用者的本機 app-data 目錄 / 64 |
| `--log-file` | 記錄檔（同時印到主控台） | 每位使用者的本機 app-data 目錄 |
| `--alias` | `/v1/models` 回報的模型 id | GGUF 檔名 |
| `--mmproj FILE`、`--vis-idle-s`、`--vis-mode`、`--vis-cache-mb` | 影像輸入（[vision.md](vision.md)） | off / 60 / auto / 1024 |

實用的環境變數（完整清單見[使用參考](usage.md#env)）：

| 變數 | 效果 |
|---|---|
| `WHIRL_KV=auto\|f16\|q8\|q8h\|q8v` | KV 格式；`auto` 套用下限規則（[kv-and-caching.md](kv-and-caching.md#kv-auto)） |
| `WHIRL_POOL_RESERVE_MB` | 池之後保留的 VRAM（預設 dense 768 / MoE 1536） |
| `WHIRL_NO_PREFIX_CACHE=1` | 停用前綴快取 |
| `WHIRL_PREFIX_CACHE_VERIFY=1` | 每次快取命中後重新 prefill 並比對 logits（診斷用） |
| `WHIRL_SERVE_CKPTS` | 每個 slot 的檢查點數（`--parallel` > 1 時預設 2，否則 4） |
| `WHIRL_MTP=0`、`WHIRL_NGRAM=0` | 停用 MTP / n-gram 共同草擬 |
| `WHIRL_MTP_BATCH_DRAFTS=d1,d2,…` | 依 decode 中 slot 數決定的草稿上限 |
| `WHIRL_GATHER_MS` | 新請求的突發收集視窗（預設 30，0 = 關閉） |
| `WHIRL_DECODE_MIN_TPS` | decode 保底速度（= `--decode-min-tps`，預設 20，0 = 關閉） |
| `WHIRL_PREFILL_CHUNK` | 每次合併 prefill forward 的最大列數（1024 的倍數，預設 2048） |
| `WHIRL_SYS_MIN`、`WHIRL_SYS_CKPTS`、`WHIRL_SYS_LCP=0` | system prompt 檢查點門檻（2048）、VRAM 中的數量（2）、停用 `prefix` 檢查點 |
| `WHIRL_KV_TIER_MIN` | 各層的最小項目大小（2048 tokens） |
| `WHIRL_TIMING_RESET=0` | 跨請求保留 MTP 計時表（不建議） |
| `WHIRL_LOOP_LOG=1`、`WHIRL_TIER_VERIFY=1`、`WHIRL_TIER_MIN_GAIN=n`、`WHIRL_TIMER_PROBE=1` | 診斷：每輪迴圈各階段計時、每次寫出/還原都逐位元組驗證、還原門檻（預設 512）、server 設定 1 ms 計時器解析度前後的實際 sleep 長度 |
| `WHIRL_PROFILE=1\|2` | 每個 cycle 的 GPU 時間拆解；**會扭曲計時**——絕不要用它量速度 |

一張 GPU 同一時間只能有一個引擎行程使用：執行檔會對每個裝置取得一個具名 mutex，第二個實例會等待
（[windows-hip.md](windows-hip.md#wddm-demote)）。

**停止。** Ctrl+C、Ctrl+Break 或關閉主控台視窗會讓 server 正常結束：排隊中的請求回 503，執行中的跑完，
完成的 session 寫出到 RAM 層，所有還沒寫進 SSD 層的 RAM 層項目（平常在最後一次變更 2 秒後才寫）立刻寫入，
最多等 10 秒（log 會有 `shutdown: tier writes done …`），下次啟動就能從 SSD 還原最後幾輪對話。
再按一次 Ctrl+C 會立即結束；關閉主控台時 Windows 約 5 秒後一定結束行程。直接殺行程（工作管理員、
`taskkill /F`）則以上都不會發生。

## 2. API 介面

| 端點 | 說明 |
|---|---|
| `POST /v1/chat/completions` | 串流（SSE）或非串流；工具；思考；影像（需 `--mmproj`） |
| `POST /v1/completions` | 原始 prompt |
| `GET /v1/models` | 一個模型 |
| `GET /health` | 即使忙碌也立即回應；回報忙碌狀態與佇列長度 |
| CORS preflight | 支援 |

回應為標準 OpenAI JSON：`finish_reason` 為 `stop` / `length` / `tool_calls`；`usage` 含
`cached_tokens`；另外加上 llama.cpp 風格的 `timings` 物件（prompt 與預測 token 的每秒數、草稿
計數），讓現有的 llama.cpp 基準測試用戶端能讀取 WHIRL 的數據。

```
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+3?"}],"temperature":0,"max_tokens":200}'
```

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = c.chat.completions.create(
    model="x",
    messages=[{"role": "user", "content": "Write a Python quicksort"}],
    temperature=0.6, seed=1,
    extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
print(r.choices[0].message.content)
print(getattr(r.choices[0].message, "reasoning_content", None))
```

**Greedy 一致性。** temperature 0 時，server 的輸出與 CLI 的輸出逐 token 相同（chat、completions、
串流、思考開與關皆然）。一個常設的關卡（gate）會針對**不帶任何環境變數、不帶任何選項**啟動的 server
檢查這點，因為曾有一個 bug 只在這條預設路徑上產生亂碼（它在載入時決定 KV 格式之後又重新載入了
kernel 表；[pitfalls.md](pitfalls.md#srv-defaultenv)）。

## <a id="sampling"></a>3. 取樣

- 參數：`temperature`、`top_p`、`top_k`、`min_p`、`seed`、`max_tokens`（或
  `max_completion_tokens`）、`stop`。
- 未提供取樣參數時，套用模型卡的建議值：思考模式 temperature 0.6 / top_p 0.95 / top_k 20；
  非思考模式 0.7 / 0.8 / 20。
- GPU 選出前 256 個候選；CPU 套用過濾條件並抽樣。
- 亂數由（seed, token 索引）產生：帶 seed 的請求無論單獨執行或並行執行、MTP 開或關，都產生相同文字。
- 推測取樣：草稿以機率 p(draft) 被接受；被拒絕時，從移除該草稿並重新正規化的分布中抽出 token——
  輸出分布與不用 MTP 時相同（[speculative-decoding.md](speculative-decoding.md#sampling)）。
- `presence_penalty` / `frequency_penalty` 會被接受但**忽略**（會有一行 `W` 記錄說明）；
  `n` 必須為 1；不支援 `logprobs`；不支援 `response_format`。

## <a id="thinking"></a>4. 思考與 `reasoning_content`

- 思考由 `chat_template_kwargs.enable_thinking` 或頂層的 `enable_thinking` 控制；
  `reasoning_effort` 與 `preserve_thinking` 依 GGUF chat template 的方式支援。也接受 OpenRouter 形式的
  `"reasoning": {"effort", "enabled"}` 物件與 effort 別名（`max` / `ultra`、`minimal`、`none`），見
  [usage.md](usage.md#endpoints)。
- 思考文字放在 `message.reasoning_content` 回傳（串流：`delta.reasoning_content`），答案放在
  `content`。
- 大多數用戶端在下一輪不會把 `reasoning_content` 送回來。此時 template 會產生一個空的
  `<think>\n\n</think>` 區塊，其 tokenize 結果與上一輪 prompt 結尾的 `<think>\n` 不同。server 會
  儲存一個 `think-open` 檢查點，讓這類回合仍能命中前綴快取
  （[kv-and-caching.md](kv-and-caching.md#think-open)）。

## <a id="tools"></a>5. 工具呼叫

- 支援的模型以 XML 風格格式輸出工具呼叫：
  `<tool_call><function=NAME><parameter=KEY>VALUE</parameter>…</function></tool_call>`。server
  將其轉換為 OpenAI `tool_calls`；JSON 風格的呼叫也會解析。參數值會轉換成工具 JSON schema 中宣告的型別。
- 歷史中的工具結果（`role: tool`）與 assistant 工具呼叫由 chat template 產生。與參考 Jinja 產生方式
  的刻意差異（記錄於 [phase1_parity.md](../../phase1_parity.md)）：以 JSON 字串給的 `arguments` 會被
  解析成參數；沒有函式名稱的工具呼叫與未知角色會被拒絕，而不是照樣產生或略過。
- `tool_choice: "none"` 會把工具從 prompt 中移除。`"required"` 或指定函式**不會強制執行**——沒有
  文法約束的 decode。
- 工具呼叫 id 是隨機的；測試只比對名稱 + 參數。

## <a id="batching"></a>6. 連續批次處理

**Slot。** `--parallel N`（預設 4，最多 16）個 slot，每個在共用池中有自己的 KV 頁清單、DeltaNet
狀態、MTP 隱藏狀態、取樣器狀態與前綴檢查點。新請求會分配到能重用最長前綴（或能從各層還原一段前綴）
的 slot。

**所有 decode 中 slot 共同的 decode（逐 token 生成）cycle：**

1. 在一個多序列批次中為所有 slot 產生 MTP 草稿（或每個 slot 各自的 n-gram 草稿）。
2. 所有 slot 共用一次驗證 forward pass：GEMV 的列橫跨所有序列；attention 的列使用各自的 KV 頁與
   位置；DeltaNet 使用分段 kernel，每段一個狀態。
3. 每個 cycle 同步一次。

**依負載的草稿上限。** 所有 decode 列都要塞進一次最多 16 列的驗證，因此草稿與使用者互相競爭。
Dense 依 decode 中 slot 數的上限：1–4 個 slot 分別為 8 / 7 / 4 / 3；成本模型在上限內挑選。MoE：
最多 8 位使用者時 1 個草稿。草稿數為零時，MTP 層仍會處理已接受的 token，以維持其 KV 完整。

**Prefill 排程。** Prefill 以 ≤ 1024 token 的區塊執行（最後一塊可多吸收 ≤ 256 個），與 decode cycle
交錯進行。多個請求的區塊會合併成一次分段 forward：逐列工作（embedding、norm、GEMM、MoE）對所有列
執行一次；逐序列工作（attention、DeltaNet conv 與 chunk scan）則以單獨模式的 kernel 逐段執行。
最舊的 prefill 中請求一定會執行；其他請求在下一塊超過 16 列（避開小批次路徑）且總數維持 ≤ 4096
時才加入。由於每種 GEMM 配置與 MoE tile 都與列無關（row-invariant），批次 prefill 的一列與單獨執行
的一列逐位元相同。沒有 slot 在 decode 時，同一請求的連續區塊會合併成最多 2048 列的 forward
（[kv-and-caching.md](kv-and-caching.md#merge)）。有 slot 在 decode 時，由下述的 decode 保底速度限制
forward 大小。

**Decode 保底速度（`--decode-min-tps N`，預設 20）。** 沒有它時，主迴圈每個 decode cycle 會跑一次
prefill forward（合併的區塊最多 4096 列，27B 模型約 1.5 s），所以其他請求在 prefill 長 prompt 時，
串流中的請求每次 forward 只前進一個 cycle——約 3 tok/s，看起來像卡住。

保底保護**同一批到達以外的所有 decode 中 slot**：只有在某個仍在 prefill 的請求的突發收集視窗（30 ms）
內到達的 decode slot 不受保護。所以一起到達的請求（情境 A：C = 4 同時到達）互不牽制，可全速合併 prefill；
而較早開始的串流，或是在子代理 prompt 之後才送來、而它們仍在 prefill 時的主對話下一輪，不論到達順序都受保護。

設定保底後，且有符合條件的 slot 在 decode 時：
- 一次 prefill forward 最多帶一個列數預算：完整的排程區塊（最舊請求的下一塊一定會執行，所以 prefill
  永遠有進度），依量測到的 prefill 速度決定大小，使一次 forward 造成的停頓約等於保底速度下 10 個
  token 的時間（N = 20 時約 1 塊）；
- 每次 prefill forward 之後，單獨執行 decode cycle，直到每個受保護的 slot 在「從這次 forward 開始
  的這段期間」內產生 ≥ N tok/s。cycle 數每個 cycle 都依實際產生的 token（MTP 接受率、decode 中 slot 數）
  與量測到的 prefill 時間調整（decode cycle 時間、每 cycle token 數、prefill 列/秒的 EMA）；
- 若連純 decode 都達不到保底速度，每段期間的純 decode 時間上限為該期間 prefill 時間的 4 倍，prefill
  仍保有約 20% 的 GPU 時間。

沒有受保護的 slot 在 decode 時，prefill 與沒有保底時完全相同（完整大小的合併 forward），突發收集也不變。
改變的只有「哪些列放進同一次 forward」與 decode cycle 的時機；每種 GEMM / MoE tile 都與列無關，
所以任何 N 的輸出都逐位元相同（已驗證：25.7k、97.4k、123.7k context 與反向順序的 N = 0 對 20，v0.1.1 在
25.7k 的 N = 0 / 10 / 20 / 30 / 40，短 prompt 測試與 Ornith）。附帶效果：列數預算
小時，等待中的 prompt 改為依先後順序 prefill，而不是並排進行，所以第一個很早就得到回應，最後一個較晚。

| 情境 B：Swift-1.5 27B MXFP4-A，R9700：主對話在不同 context 下串流，接著 3 個子代理 prompt（15.8k / 17.1k / 18.9k token）同時到達 | 25.7k context（N = 0） | 25.7k context（**N = 20**） | 97.4k context（**N = 20**） | 123.7k context（**N = 20**） |
|---|---|---|---|---|
| 它們 prefill 期間主對話串流速度，tok/s | 3.2 | **23.7** | **23.0** | **23.1** |
| 主對話串流最長停頓，s | 1.217 | **0.499** | **0.482** | **0.479** |
| 3 個子代理的 TTFT，s | 17.0 / 17.8 / 18.5 | **8.5 / 17.0 / 27.3** | **10.1 / 18.8 / 28.6** | **9.7 / 22.2 / 31.8** |
| 串流期間子代理 prefill 吞吐量，tok/s | 2797 | **1897** | **1813** | **1629** |

Ornith-1.5-35B-A3B MXFP4（MoE），同一情境：N = 0 → 20 時串流請求從 6.7 提高到 31.5 tok/s（單獨約
230；最長停頓 0.39 → 0.43 s），最後一個 TTFT 從 5.8 變為 6.8 s（prefill 8986 → 7653 tok/s）。短
prompt 不受影響：Swift MXFP4-A，約 1.1k token prompt、生成 256，C = 4 同時到達時（情境 A）總時間 4.67 s，
decode 穩態 288.8 tok/s，保底不會誤啟動，3 個請求合併 prefill（第四個因合併後超過 4096 上限循序排程）。
預設取 20：串流維持可讀（> 20 tok/s、停頓 < 0.5 s），代價約三分之一的 prefill 吞吐量，而等待中 prompt
的平均 TTFT 不變。只在乎總吞吐量的批次工作可用 0；想要更順的串流可設更高的 N，代價是 prefill 變慢。

**突發收集。** 仍在接收、解析或 tokenize 中的請求會被計數；只要還有這種請求，新請求的第一個 prefill
區塊最多等待 30 ms，讓一波突發請求進入同一次分段 forward。單一使用者永遠不會等待（自己的請求早已
排入佇列）。沒有這個機制時，突發中的第一個請求會單獨 prefill，然後在其他請求 prefill 時停頓 2.4 s。

**停頓。** 當另一位使用者送出 14k token 的 prompt 時，串流請求最長的停頓為 0.90 s（MoE：0.25 s），
這是加入 decode 保底速度之前的數字；三個長 prompt 同時到達時為 1.22 s，串流掉到 3 tok/s（見上表），
預設保底下為 0.48–0.50 s、23 tok/s。

| 並行度（27B Q4_K_M / MXFP4，約 1.1k token prompt，生成 256，greedy） | C = 1 | C = 2 | C = 4 |
|---|---|---|---|
| 實際時間總計 tok/s（含 prefill），MTP + n-gram | 61.6–62.4 | 98.0–99.6 | 138.7–139.6 |
| MXFP4，同上 | 68.5–70.7 | 118.9–119.3 | 190.3–190.7 |
| decode 迴圈內的穩態，Q4_K_M / MXFP4（情境 A） | 109.5–110.1 | — | 252.9 / **288.8** |
| llama.cpp ROCm `-np 4`，MTP 開（較舊的量測） | 39.8 | 43.3 | 64.1 |

每個並行輸出都與同一請求單獨執行時相同（有 gate 把關）。較早的 8/16 使用者量測（在後來數項優化之前；
每位使用者 4096 上下文）：27B 無 MTP 總計 100.0 / 115.5 tok/s；16 位使用者時 MTP 已無幫助
（113.1 vs 115.5），因為 16 列沒有留給草稿的空間，而 16 列 GEMV 已經是運算受限（compute-bound）。

## <a id="logging"></a>7. 記錄

每一行以本地時間 `yyyy-mm-dd HH:MM:SS.mmm` 與等級 `I` / `W` / `E` 開頭。內容在合理之處沿用
llama.cpp server：啟動資訊、每個請求的參數、重用的前綴、prefill 速度、定期的生成速度（約每 100
token）、最終摘要、MTP 接受率與結束原因。範例（省略時間戳記）：

```
I req 6 | POST /v1/chat/completions | stream off | 1 messages, 0 tools (tool_choice auto), thinking on, reasoning_effort medium
I req 6 | prompt 35 tok, common prefix 0, reused 0 tok (no usable checkpoint), prefill 35 new
I req 6 | gen: n_gen 101, tg 70.00 t/s, draft acceptance 59.3% (2.78 tok/cycle)
I req 6 | MTP: drafted 216, accepted 128, acceptance rate 59.3%, 2.78 tokens per cycle (72 verify cycles)
I req 6 | finish_reason length, completion 200 tok, reasoning 954 B, content 0 B, tool_calls 0, cache now 235 tok
I req 23 | prompt 31 tok, common prefix 31, reused 31 tok (checkpoint 'prompt-end' @31), prefill 0 new
I req 45 | queued (2 request(s) ahead)
I batch | slots busy 0/4 (decode 0, prefill 0) | 80 cycles, 3.79 slots/cycle, 11.44 rows/cycle, 2.09 drafts/cycle | aggregate tg 158.7 tok/s, prefill 0 tok/s over 4.9 s
```

其他值得認識的行：啟動時的 `VRAM use:` 行（權重 + 緩衝區、slot、其他、檢查點、KV 池、剩餘）——
這是看出一項改動要花多少 VRAM 最快的方法；KV 格式的選擇及其理由（例如
`(auto: the needed pool does not fit as f16)`）；system prompt 重用時的 `shared prefix:` 與
`checkpoint 'system' @B`；每次層寫出 / 還原 / SSD 寫入各一行，以及閒置時的 `kv tier |` 摘要；
影像輸入的 `vision:` 行。

## <a id="limits"></a>8. 已知限制

- 只支援 `qwen35` / `qwen35moe` GGUF（[architecture.md](architecture.md)）。
- 沒有文法約束的 decode：`tool_choice: "required"` 與 `response_format` 不會強制執行。
- 懲罰參數會被接受但忽略；`n` 只能 = 1；不支援 `logprobs`。
- 沒有 HTTP keep-alive（每個回應都是 `Connection: close`）；沒有關機端點（請用 Ctrl+C 停止伺服器，見第 1 節）。
- 串流輸出在引擎執行緒上寫出；非常慢的用戶端可能拖慢整個批次。
- 有快取與無快取的執行在數值上等價，但不是逐位元相同（歷史 KV 由 decode kernel 計算 vs 由 prefill
  GEMM 計算；[kv-and-caching.md](kv-and-caching.md#think-open)）。帶有 ≥ 2048 token system message
  的請求會在 system 邊界切開，因此與不切開的執行檔不是逐位元相同。
- 使用 MoE 模型時，專家路由幾乎平手的情況可能讓有快取的多輪對話與無快取的對話分歧（在一次思考模式
  測試中看過一次）；兩者都是有效的計算。
- 自動草稿數取決於量測到的計時，因此各次執行之間速度約有 ±2–4% 的差異；輸出則不會變。
- 影像輸入：只支援無 deepstack 的 Qwen3-VL 形式 mmproj；不支援影片；不抓取 `http(s)` 圖片網址（[vision.md](vision.md)）。

## 9. Server 如何測試

每次改動都會對 dense 與 MoE 模型各跑一次完整的 server 測試套件（各 50 項檢查），包括：各端點；
CLI MTP == 無 MTP；server == CLI（chat、completions 與串流）；三個並行請求 == 循序執行；有快取的
多輪 == 無快取（文字不同時以 KL 為準）；**預設環境**階段（無變數、無選項）；分段 prefill（同時 8 個
160–2,489 token 的請求，每個 == 單獨執行）；池逐出與重新計算；多輪快取命中率；層、system 檢查點與
負載下還原的 gate；以及 MXFP4 專屬的並行測試（C = 4 greedy == 單獨執行）。詳見
[benchmarking.md](benchmarking.md#gates)。
