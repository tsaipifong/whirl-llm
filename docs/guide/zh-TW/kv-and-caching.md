[English](../en/kv-and-caching.md) | **繁體中文**

# KV 快取、檢查點與分層快取

> **狀態。** 已在 C++ 伺服器中實作（`src/server/`、`src/tier/`）。所有旗標與 `WHIRL_*` 變數列在
> [使用參考](usage.md)。除非另有註明，所有數字皆為 Qwen3.8-27B 在 R9700（32 GB）上的結果。

**對誰有幫助：** 任何部署混合 attention + 遞迴模型（DeltaNet、類 Mamba）的人——這類模型的前綴快取需要遞迴狀態檢查點，而不只是 KV 區塊；任何要在 32 GB 卡上規劃 KV 格式預算的人；任何在 Windows 上建置 RAM/SSD KV 層的人。

## 1. 記憶體預算

| 項目 | 大小 |
|---|---|
| 權重（UD-Q4_K_M） | 15.32 GiB |
| 每 token KV，f16（16 層 attention + MTP 層 = 17 × K/V × 4 個 KV head × 256 × 2 B） | 68 KiB |
| 每 token KV，`q8v`（K f16、V int8） | 52.1 KiB |
| 每 token KV，`q8` / `q8h`（K 與 V 皆 int8） | 36.1 KiB |
| 每 token KV，MoE 模型（10 層 attention + MTP、2 個 KV head），f16 | 22 KiB |
| 一個 DeltaNet 檢查點（48 層的 conv + 遞迴狀態，外加一列 MTP hidden 與一列 logits） | 150.6 MiB（MoE：63.8 MiB） |
| 131,072 個 token 的 f16 KV | 8.5 GiB |

DeltaNet 狀態本身每個序列是固定大小，與上下文長度無關。server 啟動時會印出 VRAM 分配；一個例子（四個 slot，RAM 層出現之前）：權重 + prefill（提示詞預填）緩衝區 17.85 GiB（batch-4096 的每列暫存約 538 KB/列，約 2.1 GiB）、slot 遞迴狀態 + 重播緩衝區 0.82 GiB、其他 0.31 GiB、前綴檢查點 1.56 GiB（4 個 slot × 2）、KV 池 10.41 GiB、保留 0.75 GiB。

## <a id="paged"></a>2. 所有 slot 共用的分頁 KV 池

**為什麼不用虛擬記憶體。** 最初的計畫是給每個 slot 一段連續的虛擬 KV 範圍，按需映射實體頁，這樣 kernel 不用改。在 Windows/WDDM 上，kernel 只看得到映射範圍內的第一塊實體配置（[windows-hip.md](windows-hip.md#vmm)），所以 WHIRL 改用真正的頁表。

**配置。** 每一層的 K 與 V 存放在以 **256 個 token** 為一頁的池中。序列的位置 p 位於池的第 `ptab[tab + p / 256] * 256 + p % 256` 列。kernel 以傳值方式接收一個 `KvArgs { k, v, ks, vs, ptab, kvbase, tab0 }`：批次中的各列使用每列的頁表偏移（`kvbase[row]`），單一序列的啟動則使用 `tab0`。每個 key tile（32 或 64 個 key）都對齊在同一頁內，因此每個 tile 只讀一次頁表。decode（逐 token 生成）、驗證、prefill 與 MTP 層全都使用同一套機制。

- f16 分頁版本與舊的連續快取**逐位元相同**（兩個模型 × 4 種提示詞類型：prefill logits、MTP 文字、純文字）。
- 查表成本佔 decode 時間的 **0.3–0.5%**（例如 64k 純 decode 35.8 vs 35.7 ms/token）。

**server 的池管理。** 一個空閒頁堆疊；每個 slot 擁有一份頁清單（閒置時保留，供前綴重用）。在每個 prefill chunk 與每個 decode 週期之前，slot 會確保頁數涵蓋到 `pos + drafts + 2`。池用盡時，會逐出最久未使用的*閒置* slot（丟棄其前綴快取並收回其頁——若啟用分層，則先寫出，見 §8）。若所有頁都被執行中的請求佔用，該請求以 `finish_reason: length`（decode）或 HTTP 500（prefill）結束。頁帶有參考計數，讓共享前綴（§5）能以唯讀方式映射到多個 slot；仍被背景複製讀取中的頁會進入隔離清單，等該複製的 fence 通過後才回到空閒清單。

- `--ctx N` = 池大小（以 token 計）；`--ctx-per-slot N` = 每個請求的上限。
- 預設池 = 扣除權重、緩衝區、slot 狀態與檢查點後剩下的全部 VRAM，再減去保留量：**dense 模型 768 MiB，MoE 1.5 GiB**（MoE 模型會較晚配置約 0.5 GiB，保留量較小時會溢入共享記憶體）。

## <a id="formats"></a>3. KV 格式：f16、q8、q8h、q8v

| 格式 | K | V | Bytes/token（27B） | 備註 |
|---|---|---|---|---|
| `f16` | f16 | f16 | 68 KiB | 精確 |
| `q8` | int8 + 每 32 個一個 f16 scale | 同左 | 36.1 KiB | |
| `q8h` | 同 q8，但先對 q 與 k 做 256 維 Walsh–Hadamard 旋轉 | int8 | 36.1 KiB | 旋轉把 K 的離群通道攤開 |
| `q8v` | f16 | int8 + 每 32 個一個 f16 scale | 52.1 KiB | **dense 模型目前的預設** |

**量化只在寫入時做一次。** 在 `kv_st32` 中，一個 32 lane 的 wave 恰好是一個 scale 群組：wave 最大值 → `scale = amax / 127` 以 f16 儲存 → `q = round(x / scale)`。每次讀取都以一次 f16 乘法反量化；WMMA 仍吃 f16。kernel 以格式為模板參數（f16 / q8 / q8v），因此 f16 仍然可用。

**Hadamard（`q8h`）。** 在 RoPE 之後，`attn_prep` 對每個 q 與 k head 套用正交的 256 點快速 Walsh–Hadamard 轉換。在精確算術下 q·k 在旋轉下不變；K 的離群通道被攤到所有維度上，適合每 32 個一個 scale 的做法。成本：在融合的 prep kernel 中多 8 個 butterfly 階段。27B 的 KL 再降 2–3 倍。

### 3.1 品質

最後一個 token 分佈相對 f16 KV 的 KL，同一個執行檔。「路徑雜訊」是兩條同樣正確的 f16 路徑之間的 KL（分塊 vs 循序 DeltaNet prefill）；prefill chunk 4096 vs 512 在兩個模型上都是 0（逐位元相同）。

| 提示詞 | 27B 路徑雜訊 | 27B q8 | 27B q8h | MoE 路徑雜訊 | MoE q8 | MoE q8h |
|---|---|---|---|---|---|---|
| arch 1,107 tok | 6.9e-8 | 2.1e-7 | 2.3e-7 | 7.3e-4 | 5.7e-4 | 9.4e-4 |
| p4k | 1.8e-5 | 6.4e-4 | 3.1e-4 | 1.5e-2 | 6.1e-2（top-1 翻轉 0.48→0.39） | 2.0e-2 |
| p12k | 1.7e-6 | 3.4e-6 | 1.1e-6 | 3.1e-3 | 2.4e-1（p 0.73→0.41） | 1.6e-1 |
| p24k | 2.5e-7 | 1.7e-6 | 5.4e-7 | 1.1e-3 | 4.1e-4 | 1.1e-3 |

之後比較 q8v 與 q8h（27B，相對 f16）：q8v 在 10 個提示詞中有 8 個 KL 較低（多數低 2–10 倍），例如 code32k 2.0e-4 vs 3.8e-4、long64k 4.7e-4 vs 1.0e-3、code64k 7.4e-6 vs 9.3e-5；所有提示詞 top-1 都相同，top-10 皆 10/10。

**成對 QA**（350 題：150 題「這段 Python 會印出什麼」，以實際執行檢查；100 題多步驟整數應用題；100 題從 17k–70k token 的 C++ 原始碼中做 key/value 檢索，含 `_OLD` 誘餌；greedy，在 `ANSWER:` 之前先推理；McNemar 精確檢定）：

| | f16 | q8h | q8 | q8v |
|---|---|---|---|---|
| 27B 總計 | 265/350 | 263（p = 0.63） | 266（p = 1.0） | 264（p = 1.0，另一次執行） |
| 27B 程式 / 算術 / 檢索 | 67 / 98 / 100 | 65 / 98 / 100 | 67 / 99 / 100 | 66 / 98 / 100 |
| MoE 總計 | 333/350 | 334（p = 1.0） | — | — |

沒有任何格式在任務層級上出現可量測的差異（檢索全部 100%，包括含誘餌的 70k 上下文）。較早一版只要求輸出答案的 QA 太不敏感（兩個模型都只有 5–7% 正確），已被取代。

**大海撈針：** 27B 搭配 q8h，128k 在 10%/50%/90% 深度、256k 在 10%/90% 深度全部找到（256k：261,909 token 的提示詞，prefill 398 tok/s，MTP decode 30 tok/s，專用 VRAM 29.1 GiB）。

### 3.2 速度——出乎意料

| 純 decode，ms/token（CLI，同一個執行檔） | 16k | 64k | 128k |
|---|---|---|---|
| f16 | 30.0 | 35.8–35.9 | 42.0–42.1 |
| q8h | 30.3（+1.0%） | 37.1–37.2（+3.6%） | 44.2–44.3（+5.2%） |
| q8v | — | 35.7（−0.3%） | 41.7（−1.0%） |

| Prefill，tok/s | 64k | 128k |
|---|---|---|
| f16 | 1125.7–1134.5 | 926.6–927.0 |
| q8v | 1102.9–1105.4（−2.3%） | 891.0–891.1（−3.9%） |
| q8h | 1100.2–1100.8（−2.6%） | 890.2–891.3（−3.9%） |

KV 位元組減半反而讓 decode **變慢**：split-K decode kernel 內部對 K 做 int8 反量化的成本，高於它省下的頻寬（加入反量化後，decode attention 就不再是純粹受頻寬限制）。decode 的成本完全來自 K；只量化 V（`q8v`）的 decode 與 f16 一樣快，甚至略快。prefill 的成本在 V（PV 乘積），q8v 與 q8h 付出的代價相同。

### <a id="kv-auto"></a>3.3 自動選擇策略

- **MoE 模型：一律 f16。** 它在 q8 下的單一提示詞 KL 遠高於自身的路徑雜訊（p12k 時 77 倍；即使用 q8h 也有 50 倍），因此精確模式保留 f16，儘管 QA 沒有顯示差異。`q8h` 仍是經 QA 驗證、可換取更大 MoE 池的選項。（以 22 KiB/token 計，MoE 的 f16 池已約 340k token。）
- **Dense 模型：** 依 **f16 → q8v → q8h** 的順序，選第一個*下限池*放得下的格式（§4）。預設四個 slot 時是 q8v；`--parallel 1` 得到 f16；`--ctx-per-slot 262144` 得到 q8h。
- CLI 依其最大上下文以同樣方式決定（16k 與 128k 的 CLI 執行使用 f16）。
- 強制指定格式（`WHIRL_KV=f16|q8|q8h|q8v`）時，若池最後低於下限，會印出警告。

## <a id="floor"></a>4. 128k + 64k 下限規則

專案政策：在預設四個 slot 下，一個 session（工作階段）必須能達到 **128k** token，同時另一個執行中的 session 仍能達到 **64k**。閒置的 session 可以被逐出。下限池 = 131,072 + 65,536 + 每個請求 2 個 decode 頁 = **197,120 個 token**。

| KV 格式 | 預設 `--parallel 4` 下的池 | 符合下限 |
|---|---|---|
| f16 | 160,512 | 否（差 2.37 GiB） |
| **q8v** | 209,664（剛引入時）→ 目前 **199,936** | 是 |
| q8h | 302,080 | 是，但 decode +3.6%（64k）/ +5.2%（128k） |

自 q8v 引入以來池變小了，因為後來的功能在決定池大小之前就先保留了 VRAM：分層 stream 與 pinned arena（HIP 可見 12.8 MiB）、兩個共享系統提示詞檢查點（2 × 150.6 MiB，−7,424 token），以及較大的 code object（−256 token）。目前相對下限的餘裕是 **2,816 個 token**；任何新的常駐緩衝區都必須在別處省回來，否則預設會退回 q8h。

為保住 f16 而考慮過的無損 VRAM 節省手段：prefill batch 4096 → 3072 可省 0.52 GiB 且無速度代價（3072 vs 4096：4k +1.2%、16k −0.7%、64k 0.0%；2048 則慢 1.3–2.0%）；暫存緩衝區共用別名（估計 ≤ 0.75 GiB，未實作）；檢查點減半（0.78 GiB，會傷害前綴快取）。合起來仍不夠讓 f16 放得下，所以用 q8v。另外也量測過並否決：prefill batch 8192（長提示詞受 attention 限制，並沒有更快；+2.1 GiB）、把 embedding 或檢查點放在 pinned host 記憶體（鎖頁主機記憶體）（檢查點存取要約 40 ms，因為是 96 次小複製、約 3.75 GB/s，agent 回合 −25%；而且 pinned 記憶體會顯示為共享使用量（Shared Usage），[windows-hip.md](windows-hip.md#shared-usage)）。

由此得出的 server 預設：dense 27B → `--parallel 4`、每個請求 ≤ 131,072 token（`--ctx-per-slot`，上限 262,144）、q8v。MoE → f16、每個請求 131,072、池約 341k token。

## <a id="checkpoints"></a>5. 混合模型的前綴快取：檢查點

transformer 可以重用任何 KV 區塊已被快取的前綴。DeltaNet 層做不到：它在位置 p 的狀態無法從 KV 頁重建。只有在引擎**存了檢查點**的位置才能重用：每個 DeltaNet 層的 conv 與遞迴狀態，加上 MTP hidden 列（繼續草擬時需要），以及在有意義時的 logits 列（`has_logits`：若新提示詞恰好在該處結束就需要）。

| 檢查點種類 | 儲存位置 | 用途 |
|---|---|---|
| `prompt-end` | 提示詞結尾 | 同一個提示詞再來一次 / 若生成內容被丟棄時的下一回合 |
| `gen-end` | 最後一個生成的 token 之後 | 對話的下一回合 |
| `think-open` | 思考提示詞結尾 `<think>\n` 的正後方（無 logits） | 客戶端丟掉思考內容時的下一回合 |
| `system` | system 訊息結尾（≥ 2048 token） | 使用相同系統提示詞的新 session |
| `prefix` | 長共同前綴內的某個 chunk 起點 | 結尾帶有各 session 專屬文字的系統提示詞 |

當 `--parallel` > 1 時每個 slot 保存 2 個檢查點，否則 4 個。新請求會被放到可重用前綴最長的 slot；KV 以截斷方式重用，slot 的狀態則從該點或之前最佳的檢查點載入。

### <a id="think-open"></a>5.1 多回合快取為何沒命中，以及修正

server 日誌顯示有兩個根本原因，而不是一開始懷疑的單一原因：

1. **MTP 在 EOS 之後仍接受草稿**（[speculative-decoding.md](speculative-decoding.md#eos)）：狀態跑過 `<|im_end|>` 1–3 個 token，所以 `gen-end` 落在下一回合的共同前綴之外（例如：`cache now 42`，下一回合的共同前綴是 41）。
2. **思考提示詞以 `<think>\n` 結尾。** 當客戶端不回傳 `reasoning_content`（多數 agent 客戶端都不回傳）時，歷史會被重新渲染成 `<think>\n\n</think>\n\n…`，而 `\n\n` 是單一個 token，和提示詞最後的 `\n` 不同。`prompt-end` 檢查點差一個 token 沒命中，因此**每一回合都重新 prefill 整段對話**（命中率 0.000）。修正：將這類提示詞的 prefill 在 N−1 處切開，並在 `<think>` 之後存一個 `think-open` 檢查點。CLI 以相同方式切分，因此 server == CLI 仍維持逐位元相同；成本：思考提示詞多一個 1-token 步驟（27B 上約 28 ms）。

| 多回合命中率（後續回合：已快取 / 提示詞） | 27B 修正前 → 後 | MoE 修正前 → 後 |
|---|---|---|
| agent（4k 系統提示詞 + 2 個工具） | 0.961 → 0.972 | 0.968 → 0.963 |
| 思考，有回傳思考內容 | 0.681 / 0.447 → 0.862 | 0.923 → 0.848 |
| 無思考 | 0.742 → 0.759 | 0.754 → 0.747 |
| agent、思考、丟掉思考內容 | **0.000 → 0.928** | **0.000 → 0.914** |
| 思考、丟掉思考內容 | **0.000 → 0.515** | **0.000 → 0.523** |

（短對話的命中率本來就低；重點是不匹配歸零了。）在 tokenizer 空白字元修正之後，agent 情境升到 0.978 / 0.975。

剩下一個已接受的限制：模型有時生成的 token 序列並非 tokenizer 對同一段文字的標準切分（例如程式碼中 `"""'` 附近）。重新 tokenize 該回覆會在回合中途分岔，下一回合便退回前一個 `prompt-end`（損失一個回覆的 prefill）。要修正就得改成餵入生成的 token 而非重新 tokenize 的文字，這會讓有快取與無快取的輸入不同——不值得。llama.cpp 也有相同行為。

**有快取與無快取在設計上就不是逐位元相同：** 重用的歷史 KV 由 decode kernel 計算，重新 prefill 的歷史則由 prefill GEMM 計算。診斷模式會在每次快取命中後重新 prefill，並要求 KL ≤ 1e-2 且 top-1 相同（27B 實際上 KL ≤ 9e-5）。

## <a id="system-ckpt"></a>6. 系統提示詞檢查點：新 session 跳過 prefill

agent 客戶端會把 10–30k token 的工具 schema 與 skills 文字當作系統提示詞送出，而每個新 session 都重複一次。若 system 訊息結尾沒有檢查點，每個新 session 都要把它整段重新 prefill。

- **邊界。** 若提示詞以 `<|im_start|>system` 開頭，B = 第一個 `<|im_end|>\n<|im_start|>` 之後的位置（工具區塊在 system 訊息內，所以包含在內）。B 只取決於前面的 token，因此所有 system 訊息 + 工具相同的提示詞都會得到相同的 B。第一版只搜尋第一個 `<|im_end|>`；一個引用 tokenizer 文件的 30k 系統提示詞內含字面上的 `<|im_end|>`，被解析為特殊 token，邊界因此落在錯誤位置。改用三個 token 的樣式解決了這個問題（若系統文字字面上就含有 `<|im_end|>\n<|im_start|>`，B 仍會落在文字內——結果精確，只是少重用幾百個 token）。
- **一律在 B 切開。** 當 B ≥ 2048 時，prefill *一律*分成兩段 chunk 執行：[0, B) 與 [B, N)，冷請求也一樣。因此冷執行與從 B 繼續的執行跑的是完全相同的 chunk → 逐位元相同。（若用 1024 對齊、讓冷執行維持不變的檢查點，平均要重算約 512 個 token、約 0.35 s——比切分的 +0.5% 成本更差。）
- **共享檢查點。** 當某個 prefill chunk 結束於 B 時，server 會存一個 `system` 檢查點（狀態 + MTP hidden 列，預設在 VRAM 保留兩個），並持有 [0, B) 的 KV 頁參考：包含 B−1 那一頁之前的頁以參考計數共享；最後那一頁則複製一份（slot 會在 B 之後寫入）。新請求若最佳重用點是共享檢查點，就把共享頁以唯讀方式映射進自己的頁表、複製最後一頁、載入狀態，並從 B 開始 prefill。任何 slot 要在其重用點或之後寫入前，會先把共享頁換成私有頁（copy-on-write）。
- **`prefix` 檢查點**（簡化版的自動前綴快取）。有些客戶端會在系統提示詞後附加日期、工作目錄或 git 狀態，因此每個 session 的 system 訊息都不同。對新請求，server 會計算它與每個已知 token 序列（slot、共享檢查點、分層項目）的最長共同前綴 L；若 L ≥ 2048，就在其**自身** chunk 起點中 ≤ L 的最後一個位置存一個 `prefix` 檢查點（本來就是 chunk 邊界，不需額外切分）。之後的 session 只有在該位置也是自己排程中的 chunk 起點時才會使用它，因此與冷執行保持逐位元相同。第二個 session 建立它，第三個起受益。像 vLLM 的 APC 那樣的逐頁雜湊在這裡不可行：每個重用點都需要一份 150 MiB 的狀態。
- **突發。** 若排隊中請求的 system 訊息正由另一個請求 prefill 中，它會等待那個檢查點，而不是平行重算（平行的 sub-agent）。
- **分層。** 共享檢查點會寫出到 RAM/SSD 層（§8），並能在重啟後保留。在這裡發現一個 bug：寫出在分層 stream 上讀取檢查點時，產生它的主 stream 複製還沒完成——以 event 等待修正。

成本：兩個 VRAM 檢查點讓預設池縮小 7,424 個 token；冷 prefill +0.5%（MXFP4 30k +0.5…2%）。

| 新 session 第一回合 TTFT | 修正前（一律冷） | 修正後：後續 session | 重啟後（SSD 還原） |
|---|---|---|---|
| Q4_K_M，13.1k token 系統提示詞 | 10.22–10.30 s | **0.153–0.185 s** | 0.45–0.48 s |
| Q4_K_M，30.5k | 25.51–25.72 s | **0.567–0.608 s** | 1.27–1.28 s |
| MXFP4，13.1k | 4.31–4.40 s | **0.085–0.110 s** | 0.40 s |
| MXFP4，30.5k | 11.79–11.98 s | **0.361–0.379 s** | 1.06–1.07 s |

agent 情境（8 個 session，各自「詢問某個檔案 → `read_file` → 回答」，共享 14k token 系統提示詞 + 12 個工具）：循序總耗時 150.6 → 65.3 s（−56.6%），4 個並行 123.8 → 54.1 s（−56.3%）；第一回合 TTFT 平均 11.00 → 1.50 s 與 27.68 → 1.90 s。

取捨：system 訊息 ≥ 2048 token 的 server 請求，不再與未在 B 切分的執行檔逐位元相同（數值上等價，與 `think-open` 相同）；CLI 不受影響。

## <a id="merge"></a>7. 前向合併與檢查點粒度

較大的 prefill 前向計算較快，但檢查點（`prefix`、`system`、`think-open`）只能位於 chunk 邊界。把 server 的 chunk 改成 2048 列能讓冷 prefill 快 +18…+21%，但跨 slot 的後續請求因此重用的是位置 27,418 而非 28,442 的 `prefix` 檢查點，30k 上下文 + 300 / + 1000 個新 token 的 TTFT 從 980 → 1681 ms 與 1548 → 2230 ms。

採用的設計是**排程**維持 1024 token 的 chunk（檢查點位置不變），而在沒有其他 slot 正在 decode 時，把連續的完整 chunk 合併成一次最多 2048 列（+256 尾段）的前向計算來**執行**，且絕不跨越檢查點邊界。這依賴 logits 與前向大小無關——已驗證：batch 1024 / 2048 / 4096 的最後一個 token logits 相同，兩個模型、MTP 開/關皆然。

| Server，MTP + n-gram | Q4_K_M 修正前 → 後 | MXFP4 修正前 → 後 |
|---|---|---|
| 冷 prefill 8k / 30k（tok/s） | 1304 → 1560（+19.6%）/ 1196 → 1413（+18.1%） | 3135 → 3235 / 2595 → 2637 |
| 30k 已快取 + 1000 個新 token，跨 slot | 1540 → 1376 ms | 849 → 833 ms |

## <a id="tiers"></a>8. 分層前綴快取：VRAM → pinned RAM → SSD

當某個 session 的 slot 被其他 session 拿走，或 server 重啟時，它的上下文過去得從頭重新 prefill（126k token：147 s）。分層把這件事變成一次還原。

### 8.1 儲存內容

一個項目是某個閒置 slot 的快照：到最後一個檢查點位置 P 為止的 token id、該 slot 的檢查點（DeltaNet conv/遞迴狀態、MTP hidden 列、logits 列），以及位置 0..P−1 的 KV 頁（256 token 一頁、每個 attention 層與 MTP 層、KV 格式的每個陣列：K、V、K scales、V scales）。**還原時把同樣的位元組複製回去**，因此還原後的 slot 與從未離開 VRAM 的 slot 逐位元相同。

配置（RAM 與檔案中相同）：檢查點 k 位於 `k × ck_stride`（150.6 MiB 向上取整到 2 MiB），接著 KV 頁 j 位於 `nck × ck_stride + j × page_bytes`（q8v 每頁 13.0 MiB，f16 為 17.0 MiB）。

### 8.2 RAM 層

- 啟動時一次配置的 pinned arena，以 512 MiB 為單位分塊配置、以 2 MiB 區塊管理；項目對應到區塊清單；LRU 逐出（已在 SSD 上的項目退回其 SSD 副本，否則直接丟棄）。
- 所有 GPU↔host 複製都在單一**非阻塞分層 stream**（FIFO）上以 ≤ 1 MiB 的片段執行，因此 decode 的小量回讀永遠不會排在長傳輸後面。讓 kernel 直接寫 host 記憶體（zero-copy）在這台機器上行不通（[windows-hip.md](windows-hip.md#zero-copy)）。
- 必須拿掉全裝置同步：`hipDeviceSynchronize` 會等待分層複製。
- 預設大小：max(8 GiB, 一個完整長度 f16 session + 檢查點) = 27B 在 128k 時為 **9 GiB**。它會在 GPU 計數器中顯示為共享使用量（Shared Usage）；server 會宣告其 pinned 大小，讓監控可以扣除。

### 8.3 SSD 層

- 每個項目一個檔案（4 KiB 標頭、token id、資料區）。一個 I/O 執行緒以無緩衝的定位寫入只寫髒區塊：標頭標記無效 → 資料 → flush → 有效標頭。項目在 2 s 內未變動後才寫入。
- 啟動時掃描並索引目錄；檢查 magic、fingerprint 與 token checksum，刪除損壞的檔案。超過大小上限（預設 64 GiB）時刪除 LRU 檔案。
- **Fingerprint：** 執行檔（大小 + 修改時間）、模型檔名 + tensor 數量 + 位元組數、KV 格式、MTP 開/關、配置、prefill batch，以及所有會改變數值的引擎設定（排程/日誌/分層設定除外）。由不同 build 或不同數值設定寫出的項目永遠不會被還原。
- 讀取：最多 4 個重疊讀取同時進行，依序完成（NVMe 2 MiB QD4 7.0 GB/s vs QD1 4.1 GB/s）；還原是管線化的——一個區塊讀完，其片段就立刻做 H2D 複製。

### 8.4 寫出與還原策略

- **寫出：** 每個請求結束後，P ≥ 2048 token 的閒置 slot 會被非同步複製到 RAM。同一 session 的下一回合會**原地增量**更新該項目（只有變動的檢查點，以及從第一個被修改的頁開始的頁；一個 agent 回合 ≈ 2 個檢查點 + 2 頁 ≈ 318 MiB）。完全沒有重用某 slot 任何內容的請求，會把該 slot 與其舊項目解除連結（發現的 bug：否則不相關的 session 會覆寫另一個 session 的項目）。
- **絕不等待寫出。** 若新請求必須覆寫某個寫出仍在讀取的位置，slot 會保留重用點之前的頁、複製包含重用點的那一頁（device-to-device），並在其後取用新頁；舊頁進入隔離，直到分層 stream 通過 fence。（第一版會等待：當請求命中正在寫出的 slot 時，主 stream 會停頓約 150 ms——以該連線速度，692 MiB 的寫出要 180 ms。）
- **還原：** server 先照舊挑出 VRAM 前綴最長的 slot；若分層中有明顯更長的可用前綴（多 ≥ 512 個 token 且 ≥ 重用量/40），就從該項目填入 slot（在分層 stream 上做 H2D，其他 slot 照常 decode），然後執行正常的工作啟動流程（載入檢查點會丟棄待處理的重播列），因此接下來的 prefill chunk 與從未被逐出的 slot 完全相同。TTFT 包含還原時間。
- **主體/尾段切分：** 命中的檢查點與 KV 頁（「主體」）先複製，請求在它們到位時就開始；slot 的其他檢查點隨後跟上（「尾段」）。若檢查點儲存遇上仍在搬移中的尾段，會丟棄尚未排入的部分，而不是等待。
- **延遲命中：** 排隊中請求的最佳前綴若是正在還原中的項目，會等待那次還原，而不是啟動第二次；共享項目還原後會成為 VRAM 共享檢查點，其他請求直接接上。
- **SSD → RAM 預取：** 排隊中請求的最佳前綴若是只在 SSD 上的項目，會在等待期間開始把它讀進 RAM。
- 延遲命中與命中時預取的構想來自 Strata 論文（arXiv 2508.18572）；未使用其程式碼。

### <a id="page-return"></a>8.5 頁歸還 bug

在負載下，一次 126k token 的還原始終沒有發生。某個收到短請求、完全沒有重用任何內容的 slot，保留了舊 session 的頁（459 頁）——執行中的 slot 無法被逐出——只剩 310 個空閒頁。還原放不下，請求退回完整的 117k token prefill（約 75 s，期間其他三個 slot 的 decode 只有 9–15 tok/s），最後以 `KvPoolFull`（HTTP 500）失敗。修正：工作開始並把快取截斷到重用點時，超過 `N + drafts + 2` 的每一頁都立即歸還到池中（若有寫出正在讀取則經由隔離；共享頁則經由參考計數）。這個 bug 從引入分頁起就存在；只有在長 session 與短請求混合時才會出現。

### 8.6 結果

| Q4_K_M，上下文 | 完整 prefill TTFT | 從 RAM 還原 | 重啟後從 SSD | 從未被逐出（VRAM 命中） |
|---|---|---|---|---|
| 28.3k | 23.06 s | 0.590 s | 0.654 s | 0.18 s |
| 67.2k | 約 59 s（62.5k：58.83 s） | 1.221 s | 1.270 s | 0.23 s |
| 125.1k | 約 147 s（126.2k：147.14 s） | 2.178 s | 2.207 s | 0.35 s |
| 125.1k，同時有 3 個其他 slot 在 decode | （頁修正前：無法還原，KvPoolFull） | **2.225 s** | 2.280 s | |

一次還原每 token 約 14–16 µs——這是這台 eGPU 的主機連線上限（實測約 3.65 GB/s）——外加兩個檢查點約 80 ms；比重新 prefill 快 35–65 倍。因為讀取與 H2D 複製是管線化的，SSD 只比 RAM 慢 2–10%。MXFP4 28.3k：完整 10.70 s、RAM 0.62 s、SSD 0.71 s。

| 突發：30.6k token 系統提示詞已被逐出 VRAM，同時 3 個新 session | 修正前 | 修正後 |
|---|---|---|
| 共享項目在 RAM | 2.78 / 23.57 / 3.68 s | 0.64 / 0.83 / 0.82 s |
| 重啟後（項目在 SSD） | 42.1 / 42.1 / 41.9 s | 0.89 / 0.71 / 0.89 s |

逐出壓力下的多 session agent（6 個 session 共享約 12k token 系統提示詞，各自讀取一個約 3.5k token 的檔案並問 3 個問題，4 個 slot，請求交錯使每個 session 都發現自己的 slot 被佔走）：關閉分層 378.1 s 總耗時 / 435,697 個新 prefill token；開啟分層 **146.4 s** / 129,968 個 token——快 2.58 倍，新 prefill 數量與從未被逐出的單獨執行相同，文字完全相同。正常路徑（無逐出）沒有變慢（CLI 平均 182.43 vs 182.43 tok/s）。

為什麼被逐出的 session 在某一種情況下會與單獨執行不同：若它自己的項目比共享系統檢查點長不到 512 個 token，server 會刻意從系統檢查點繼續，並重新 prefill 那幾個 token（原本由 decode 產生）。結果是那些 token 的正確冷計算——數值上等價，但與 decode 產生的 KV 並非逐位元相同。強制一律還原（`WHIRL_TIER_MIN_GAIN=1`）讓 28/28 次呼叫與單獨執行完全相同，而位元組驗證模式也確認每一次寫出/還原都位元組相同（共 115 次）。

**[eGPU] 環境限制：** 當還原佔滿 USB4 連線時，kernel 派送會變慢，其他 slot 的 decode 週期在還原進行的 0.5–2.0 s 內從 43.0 拉長到 50.4 ms（+17%）。未做最佳化（[windows-hip.md](windows-hip.md#egpu)）；在直接 PCIe 上預期不會發生。

### 8.7 關卡（gate）

- `tier_gate`：一個不帶分層的參考 server 在無逐出情況下跑每個 session 的第 1、2 回合；分層 server 跑完所有第一回合後被強制逐出，再跑第二回合（從 RAM 還原：文字與已快取 token 數必須與參考相同），然後重啟再跑一次（從 SSD 還原）。涵蓋 `--parallel 1`（f16 KV，思考與非思考 session，包括 `think-open`）與 `--parallel 4`（q8v，五個 session）。兩個模型都測。
- `sys_gate`：共享系統檢查點重用（cached == B、text == cold）、4 個並行新 session、逐出到 RAM 與還原、重啟後的 SSD、關閉分層、突發，以及 `prefix` 檢查點。
- `restore_conc_gate`：在 3 個正在 decode 的 slot 旁做 RAM 還原 == 從未被逐出；共享項目在 RAM / SSD 上的突發；為排隊中請求做 SSD 預取。
- 每個測試 server 都有自己的 SSD 目錄；否則前一個測試寫出的項目會把下一個測試的 prefill 變成還原（fingerprint 相同），改變了受測的內容。

## 9. 未解項目

- 每個 session 的分層項目仍會複製一份共享系統提示詞的 KV；使用 30k token 系統提示詞時，9 GiB 的 RAM 層很快就滿了。項目可以改為參考共享項目的區塊。
- VRAM 只放得下兩個系統檢查點；多個交替使用的系統提示詞（主 agent + 不同的 sub-agent）會互相逐出（13k token 的還原約需 0.3 s）。
- 對同一個 session 項目的第二個請求會再從 RAM 還原一次；它其實可以共享第一個 slot 的 VRAM 頁。
- pinned 大小宣告檔以 PID 為鍵；PID 重用造成誤報（應改用 PID + 啟動時間）。
- q8v 的 prefill 比 f16 慢 2–4%（PV 迴圈中的 V 反量化）；把 V tile 一次反量化進 LDS 或許能追回來。
