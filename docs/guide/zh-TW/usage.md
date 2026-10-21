[English](../../usage.md) | **繁體中文**

# 使用參考：`whirl` 與 `whirl-server`

本頁列出 WHIRL 0.2.2.1 的每一個指令、選項、環境變數與伺服器端點。內容依發行版執行檔的內建說明
（`whirl --help`、`whirl <指令> --help`、`whirl help env`、`whirl-server --help`）整理；若有出入，以你手上
執行檔的說明文字為準。

第一次使用？請先看[快速上手](../../quickstart_zh-TW.md)。伺服器的設計背景見 [server.md](server.md)，快取見
[kv-and-caching.md](kv-and-caching.md)。

- [1. 兩支程式](#programs)
  ——[數值模式：precise、balance、fast](#modes)
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

**第一次使用某個模型。** prefill kernel 會針對每個模型檔與 GPU 調校一次（stderr 會有提示）：MXFP4 檔只要幾秒，
K-quant 檔（混用多種 tensor 型別）會久一些。結果會快取起來（[第 8 節](#files)），之後直接開始。每一種候選組態算出的位元都
相同，只有速度不同。

**一張 GPU 一個行程。** WHIRL 執行時會持有該 GPU 的鎖。同一張 GPU 上的第二個 WHIRL 行程會等前一個結束
（預設最多 30 分鐘），而不是共用 VRAM——因為一張 GPU 上同時有兩個大行程時，Windows 會把兩者都移到很慢的共享
記憶體（[windows-hip.md](windows-hip.md#wddm-demote)）。見 `WHIRL_GPU_WAIT` / `WHIRL_GPU_SHARE`。

### <a id="modes"></a>數值模式：balance（預設）、precise、fast

一個行程只用一種模式，啟動時決定（`--precise`、`--balance`、`--fast`，或 `--mode M`；也可用環境變數
`WHIRL_MODE=precise|balance|fast`，命令列優先）。`chat`、`bench`、`selftest`、`seqtest` 與伺服器都適用，R9700 與 Radeon 8060S 都一樣。
沒指定模式時跑 **balance**；`--precise` 與 `--fast` 需要明確指定（`--balance` 仍可用）。

**KV 快取格式：依模式與模型類型固定**，每張卡（32 GB、16 GB、8060S；fast 有一個例外，見表）都一樣，`chat`、`bench` 與伺服器也一樣：

| 模式 | dense 模型 | MoE 模型 |
|---|---|---|
| precise | f16 | f16 |
| balance（預設） | q8h（K、V 都是 int8，每 32 個值一個 f16 scale，q / k 先做 Hadamard 旋轉；在 128k 通過相對 f16 的 balance KL 門檻，KL-1） | f16 |
| fast | q4（Radeon 8060S：q8h，在那裡 decode 比 q4 快） | q4 |

格式不會因為放不放得下而切換：選定的格式放不下要求的 context 時，沒有指定 context 就降低 context 並發出警告；
有指定（`--ctx`、`--ctx-per-slot`）就停止並說明（結束代碼 5）。絕不自行改用更低精度的格式。20 GiB 以下的卡只會預設一個伺服器 slot 並保留 VRAM 給桌面。

| 模式 | 內容 |
|---|---|
| **precise**（`--precise`） | GGUF 權重反量化成 f16；prefill 的 activation 與累加用 f16 / f32；DeltaNet prefill 用 f32 區塊路徑；**KV 快取在每張卡上都是 f16**。decode、MTP / n-gram 驗證與其他小 batch 也用 f16 activation 進同一個 f16 GEMM（f32 累加，不用 int8 activation）。除了檔案本身的量化權重，不再額外量化。f16 KV 放不下時：沒有指定 context 就降低 context 並發出警告；有指定（`--ctx`、`--ctx-per-slot`）就停止並說明（結束代碼 5）——絕不自行改用 int8 KV |
| **balance**（預設；`--balance`） | 沒指定模式時的預設：下表以速度為主的項目，dense 模型用 q8h KV、MoE 模型用 f16 KV；dense 與 MoE 模型開的項目相同。近乎無損但不是逐位元相同：輸出可能與 precise 有些微差異（KL / 準確度實測見 [quantization.md](quantization.md)） |
| **fast** | balance 再加上更積極的有損項目，每項都要先通過 KL 與配對準確度門檻。已實作：`kvq4` 與 `relaxacc`；其他列為略過 |

項目（用 `--balance=fp8,kvq8` 或 `--fast=...` 只選一部分；`WHIRL_MODE=balance:fp8,kvq8`）：

| 項目 | 模式 | 內容 | R9700（gfx1201） | Radeon 8060S（gfx1151） |
|---|---|---|---|---|
| `fp8` | balance | MXFP4 dense prefill GEMM 用 fp8（e4m3）activation，含 MXFP4 指數折疊的捨入 | MXFP4 模型 | 略過（沒有 fp8 WMMA） |
| `moefp8` | balance（只限 MoE 模型） | MoE 專家 prefill GEMM 用 fp8 activation | MXFP4 MoE（Ornith MXFP4） | 略過 |
| `gdnwmma` | balance | DeltaNet prefill 區塊用 f16 WMMA，取代 f32 區塊路徑 | MXFP4 模型，以及任何權重格式的 dense 模型（Q4_K_M、Q5_K、Q6_K、IQ4_XS 等；KG-1）；MoE 模型只限 MXFP4 | 略過（還沒有 kernel） |
| `h16` | balance | FFN / DeltaNet 的 GEMM 輸出先存成 f16 再進逐元素運算 | MXFP4 模型，以及任何權重格式的 dense 模型（KG-1）；MoE 模型只限 MXFP4 | MXFP4 模型，只有 f16 權重 |
| `kvq8` | balance | dense 模型一律用 q8h KV（不是 f16 放不下才換）；MoE 模型維持 f16 | dense 模型 | dense 模型 |
| `kvq4` | fast | 4-bit KV（int4，每 32 個值一個 f16 scale，q / k 先做 Hadamard 旋轉）；8060S 的 decode attention 用 `attn_dq4` 讀取（約 180 GB/s）：Ornith-1.5-35B-A3B MXFP4 decode 128k 58.5 tok/s（f16 KV 32.7）、64k 67.0（46.4）；R9700（FAST-1c）Swift-1.5 27B 在 131k 的 decode：plain 30.98 tok/s（f16 KV 25.08）、MTP 68.79（63.12），128k prefill −1.9%；對 f16 的 KL 在 fast 門檻內 | dense 與 MoE | MoE（dense 模型在 fast 維持 q8h KV：那裡 decode 較快） |
| `relaxacc` | fast | 寬鬆推測接受（只對 MTP draft；n-gram draft 維持精確接受）。放寬接受若會讓結尾的 6-gram 與最近 4096 個 token 內重複，就不接受（重複防護）。greedy：draft 不是驗證列的 argmax 時，只要它在該列前 `WHIRL_RELAX_K` 名（預設 4）且 p(draft) >= `WHIRL_RELAX_ALPHA` × p(argmax)（預設 0.1）就保留；最後一個保留的 draft 之後那個 token 一定是目標模型自己的 argmax。temperature > 0：typical acceptance（Medusa）：在請求過濾後的分布上 p(draft) >= min(`WHIRL_RELAX_EPS` 0.09, `WHIRL_RELAX_DELTA` 0.3 × exp(−熵)) 就保留（此時不用 `specsample`）。**會改變 greedy 輸出**：fast 的 greedy 不等於 plain greedy，MTP ≠ plain；同樣的 prompt、seed、設定仍是確定的（此時 draft 數量策略改用固定的合成週期成本，不看牆鐘時間；並發請求共用 draft 數量，可能互相影響輸出）。`WHIRL_RELAX=0` 關閉；MoE 模型略過（每輪只有一個 MTP draft，量不到收益；`WHIRL_RELAX=1` 強制開） | dense MTP 模型 | dense MTP 模型 |
| `headq`、`moeskip`、`a8`、`a4` | fast | 低位元輸出頭、略過 MoE 專家、W4A8 / W4A4 prefill | 尚未實作 | 尚未實作 |
| `q8dec` | balance、fast | decode / verify GEMV 的 activation 用 int8（每 32 個值一個 f32 scale，同 llama.cpp 的 q8_1；balance 與 fast 一律包含） | 開 | 開 |

balance 與 fast 依模型類型開了什麼（R9700；Radeon 8060S 沒有 fp8 WMMA，所以 `fp8` / `moefp8` 在任何模式都略過）：

| 模式 | dense 模型（MXFP4 dense；Q4_K_M 等 K-quant 檔也適用） | MoE 模型（Ornith-1.5-35B-A3B MXFP4） |
|---|---|---|
| precise | 全程 f16 / f32，f16 KV | 全程 f16 / f32，f16 KV |
| balance | `fp8`（只限 MXFP4 檔）、`gdnwmma`、`h16`、`kvq8`（q8h KV）、`specsample`、`q8dec` | `fp8`、`moefp8`、`gdnwmma`、`h16`、`specsample`、`q8dec`，f16 KV |
| fast | balance + `kvq4`（q4 KV）+ `relaxacc` | balance + `kvq4`（q4 KV）；`relaxacc` 略過 |

balance 是以允許使用的項目定義（fp8 與 moefp8 activation、int8 decode activation、f16 中間結果、int8 KV），不是以逐位元貼近 precise 定義。MoE balance 從 MOE-BAL（2026-10-13）到 MODES-MXFP4（2026-10-16）不開 `fp8` / `moefp8`，現已放回：R9700 上 Ornith-1.5-35B-A3B MXFP4 的 balance prefill 在 2k +40%、8k +35%、32k +26%（13.0k / 13.0k / 9.6k tok/s，decode 不變）。`--balance=gdnwmma,h16,specsample`（明確列出的項目照用）可得到先前的 MoE balance。

MoE balance 保留 `q8dec`（decode / verify 用 int8 activation，即 llama.cpp 自己的 q8_1 做法；MOE-BAL-3，2026-10-15）：KL gate 對 MoE balance 的 top-1 容許比噪音底線低 3.0 個百分點（原為 2.4）：llama.cpp 本身 decode 就是用 int8（q8_1）activation，所以這與 llama.cpp 是同一個數值等級；KL 限制（全窗平均 ≤ 2 N、每長度 ≤ 3 N）不變。短暫存在的 MOE-BAL-2 表（不含 `q8dec`）讓 R9700 上 MoE balance 的 decode 慢約 20%。`WHIRL_Q8DEC=0` 可改用 f16 decode activation。

precise 模式關閉 `q8dec`：decode、MTP / n-gram 驗證與並發 batch 的每個 matmul 都用 f16 activation 進 f16 WMMA GEMM（權重從 GGUF 型別解成 f16、f32 累加、每列加總順序固定），所以每一列不論 batch 大小都得到相同的位元；如果 GPU 的 kernel 對模型某種權重型別缺少這條路徑，precise 模式會拒絕載入，不會退回 int8。
不適用於該 GPU 或模型的項目會記錄為略過（不會出錯）。啟動 log 會有一行，例如

```
numerics: balance - enabled: fp8, gdnwmma, h16, kvq8 (q8h KV (int8 + f16 scale / 32, Hadamard-rotated q/k)); skipped: moefp8 (dense model); ...
```

伺服器在 `GET /props` 回報模式（`"numerics": {"mode", "label", "enabled", "skipped", "overrides", "kv", "decode", "always_on"}`；`decode` 是 `f16` 或 `q8dec`）。
[7.6](#env-numerics) 的單項變數（`WHIRL_FP8`、`WHIRL_MOE_FP8`、`WHIRL_GDN_WMMA`、`WHIRL_FFN_H16`、`WHIRL_Q4_RELAXED`）與
除錯用的 `WHIRL_KV`（[7.7](#env-debug)）仍然有效，而且會覆寫模式；在 precise 模式下設成有損的值，log 會標示 `user-requested`，模式顯示為 `precise+overrides`。
在 precise 與 balance 模式內（以及 fast 加 `WHIRL_RELAX=0`），MTP / n-gram 解碼等於純 greedy 解碼，並發請求等於單獨執行；fast 開著 `relaxacc` 時輸出取決於 draft：同一請求重送時，串流與否、冷啟動或從它自己的前綴快取檢查點續接，文字都相同；但從另一個 prompt 的共用前綴續接時分塊不同，fast 的輸出可能因最後幾個位元而改變（與並發請求相同）。
fast 模式的目標是品質落在它的門檻內（相對 precise 的平均 KL <= 0.03 且 top-1 >= 93%，由 tools/klgate 在 8k–128k 檢查 dense 模型；MoE 模型改用自身噪音底線 N 判定：fast 平均 KL <= 8 N、每個長度 3 個窗的平均 <= 12 N、top-1 不低於 N 的 top-1 減 8 個百分點；R9700（每窗 512 列）上 Qwen3.8-27B fast 的 KL <= 0.0072、top-1 >= 96.09%，Ornith-1.5-35B-A3B fast 在兩份提示都是 6.1 N），不保證並發與單獨執行的輸出逐位元相同：開著 `relaxacc` 時 draft 節奏隨負載而變，並發的文字可能與單獨執行不同（盡力而為；precise 與 balance 則保證相同）。

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
| `--device SPEC` | GPU：`r9700`、`8060s`、索引，或名稱 / gfx 架構的子字串（也可用 `WHIRL_DEVICE`）。預設：第一張 R9700，沒有的話用第一張這個建置有 kernel 的 GPU（例如只有 Radeon 8060S 的機器） |
| `--precise` / `--balance` / `--fast` / `--mode M` | 數值模式（預設 balance；也可用 `WHIRL_MODE`）；`--balance=項目` 只選部分項目。見[數值模式](#modes) |
| `-h`、`--help` | 說明 |

範例：

```powershell
.\whirl.exe chat C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf "用兩句話說明 TCP 慢啟動（slow start）。" --max-tokens 400
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
| `--precise` / `--balance` / `--fast` / `--mode M` | 數值模式，同 `chat` |

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
| `--device D` | GPU：`r9700`、`8060s`、裝置索引，或名稱 / gfx 子字串（預設同 `chat`） |
| `-np`、`--parallel N` | 同時處理的請求 slot 數（continuous batching），1～16，預設 4（VRAM 小於 20 GiB 的卡預設 1，見 `WHIRL_VRAM_HEADROOM_MB`） |
| `-c`、`--ctx N` | 共用 KV 池的大小（token）。slot 依需要取用分頁；池滿時，閒置 slot 的前綴快取依最久未使用（LRU）逐出。預設：權重與緩衝區之後剩下的全部 VRAM，減去可用 VRAM 的 768 MiB（MoE：1.5 GiB），並在 Windows（WDDM）預算之下替其他程式留 margin（`WHIRL_POOL_BUDGET_MARGIN_MB`）；Radeon 8060S 上為 262144 |
| `--ctx-per-slot N` | 單一請求的最長 context（預設 min(池大小, 131072)；最多 262144） |
| `--precise` / `--balance` / `--fast` / `--mode M` | 數值模式（預設 balance；`--precise`：f16 KV；f16 池放不下一個完整請求時，會降低每個請求的 context 並警告；若有指定 `--ctx` / `--ctx-per-slot` 則拒絕啟動）。見[數值模式](#modes) |
| `--mtp-drafts N` | 每回合固定的 MTP 草稿數，1～10（預設：依模型類型由成本模型決定） |
| `--decode-min-tps N` | decode 保底速度：其他請求在 prefill 時，每個串流中（decode 中）的請求至少維持 N tok/s；做法是縮短 prefill forward、穿插 decode cycle（預設 20；`0` = 關閉，prefill forward 不受限）。任何 N 的輸出都相同（[server.md](server.md#batching)） |
| `--kv-ram-mb N` | 前綴快取的主記憶體層，單位 MiB 的 pinned 記憶體（預設：實體記憶體的 1/4，至少 8 GiB 或一個完整長度 session（若更大；27B 模型約 9 GiB），最多 32 GiB，且不超過啟動時可用記憶體的一半；64 GB 的電腦為 16 GiB；0.2.2.1 起不論指定或自動，RAM 層一律以 20480 MiB 為上限（測試機上更大的 pinned RAM 層會在第一個請求當機，原因調查中；0.2.2.2 起自動大小以一般訊息註明，指定的大小印 `adjusted:`，且先套上限再檢查是否超過可用記憶體的一半，超過時降到一半）；整合式 GPU（Radeon 8060S）：作業系統可見記憶體的 1/8，下限為一個最大 context 的條目（27B 模型在 262k 約 6 GiB），上限 16 GiB，且不超過啟動時可用記憶體的 1/4；作業系統可見記憶體低於 32 GB 則關閉（BIOS 的 VRAM 保留區不計；64 GB 的電腦為 8 GiB））。啟動日誌會印出選定的大小與原因。閒置 session 會複製到這裡，下次直接還原而不必重新 prefill。`0` 會關閉兩個 host 層。**Radeon 8060S**（內顯）：預設開啟，大小如上（KV pool 是 BIOS 保留區，主記憶體層是另外 pin 住的作業系統記憶體；128k 的 session 從 RAM 或 SSD 還原，不必重新 prefill）；`--kv-ram-mb 0` 關閉兩個 host 層，`--kv-ssd-gb 0` / `--kv-ssd-mode off` 只關 SSD 層 |
| `--kv-ssd-dir PATH` | SSD 層目錄（預設 `%LOCALAPPDATA%\whirl\kvcache`） |
| `--kv-ssd-gb N` | SSD 層容量上限，GiB（預設 64；`0` = 不用 SSD 層） |
| `--kv-ssd-mode M` | SSD 層何時寫入：`shutdown`（預設；只寫重用過的項目：關閉 WHIRL 時——Ctrl+C、關閉主控台視窗、登出、作業系統關機，後三者只有約 5 秒——、主記憶體層必須擠出時，以及閒置約 5 分鐘後；一次性的 prompt 永遠不寫）、`always`（閒置超過 `--kv-ssd-delay` 後寫入）、`off`（不寫 SSD 檔案）。見[SSD 層如何儲存 session](#ssd-tier) |
| `--kv-ssd-delay SEC` | `--kv-ssd-mode always`：session 連續 SEC 秒沒有變動才寫入 SSD（預設 300）。在這之前又被使用的 session 直接從主記憶體服務，不會寫入 |
| `--mmproj FILE` | 視覺編碼器（Qwen3-VL 形式的 mmproj GGUF，F16 / BF16）。`image_url` 內容（base64 PNG / JPEG 等的 `data:` URL）會變成圖片 token。權重放在 pinned 主記憶體，收到圖片之前不占 VRAM |
| `--vis-idle-s N` | N 秒沒有圖片就釋放視覺編碼器（預設 60） |
| `--vis-mode M` | `auto`（預設）、`resident`（編碼器權重常駐 VRAM）、`stream`（逐層串流） |
| `--vis-cache-mb N` | 依內容雜湊快取圖片 embedding 的主記憶體容量，MiB（預設 1024） |
| `--allow-local-images` | 也接受本機檔案路徑 / `file://` URL 當圖片來源（預設關閉；`http(s)` 圖片網址一律不抓） |
| `--cors-origin ORIGIN` | 允許來自 `ORIGIN`（例如 `https://app.example.com`）的網頁從瀏覽器呼叫伺服器；可重複指定。預設只有 `http(s)://localhost`、`127.0.0.1`、`[::1]`（任何 port）上的網頁會拿到 CORS 標頭（回傳原本的 `Origin`，不回 `*`）；其他網頁（包括 `file://` 開啟的頁面）讀不到回應。`*` 恢復 v0.1.4 之前的行為（對所有網頁回 `Access-Control-Allow-Origin: *`）。`Origin` 不被允許的網頁送來的 `POST` 直接回 403、不執行，包括不需要 preflight 的 `text/plain`「簡單請求」。非瀏覽器的用戶端（SDK、curl、IDE 代理）不送 `Origin`，不受影響。本機圖片檔（`--allow-local-images`）對 localhost 以外網頁送來的請求一律不讀，不管 `--cors-origin` 怎麼設 |
| `--allow-host NAME` | 也回應 `Host` 標頭是 `NAME` 的請求（可重複；`*` = 任何名稱）。預設只回應 `localhost`、`127.0.0.1`、`[::1]` 和 `--host` 的位址（`--host 0.0.0.0` 時是任何 IP 位址）；其他名稱回 403。這是擋 DNS rebinding（網頁把自己的網域改解析到 127.0.0.1）。用戶端用主機名稱連線時才需要，例如 `--host 0.0.0.0 --allow-host mypc.lan` |
| `--log-file PATH` | log 檔（預設 `%LOCALAPPDATA%\whirl\server.log`）；log 也會印在主控台 |
| `-h`、`--help` / `-V`、`--version` | 說明 / 版本 |

範例：

```powershell
# 27B 模型、4 個 slot、預設分層快取
.\whirl-server.exe C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf --port 8080

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
| `GET /health` | 忙碌時也會立刻回應；回報忙碌狀態與佇列長度；Windows 上另有 `vram`（最近一次顯示記憶體取樣：`local_mib`、`budget_mib`、`shared_mib`、`spill`）與 `startup`（啟動配適後實際生效的值：`ctx_per_request`、`pool_tokens`、`parallel`、`kv_ram_mb`、`wddm_margin_mib`、`other_processes_mib`、`display_attached`、`adjusted`：伺服器調低的設定；`wddm_margin_mib` 是決定 KV 池大小時用的 margin，`other_processes_mib` 是啟動時其他行程的用量） |
| `GET /props`（也接受 `/v1/props`） | 唯讀，llama.cpp server 形式的子集，供會自動偵測 context 長度的客戶端使用：`default_generation_settings.n_ctx`（每個 slot 的 context）、`default_generation_settings.model` 與 `model_alias`（= `--alias`）、`total_slots`、`model_path`（只有檔名，絕不含目錄）、`modalities`、`build_info`，以及與 `GET /health` 相同的 `startup` 物件 |
| `GET /version` | `{"version":"0.2.2.1","name":"whirl"}` |
| `OPTIONS`（CORS preflight） | 支援（允許的來源見 `--cors-origin`） |

LM Studio（`/api/v1/models`）與 Ollama（`/api/tags`、`/api/show`、`/api/version`）的原生端點不模擬
（客戶端偵測到它們就會改用 WHIRL 沒有的 API）：一律回 404，且每個路徑只在第一次被探測時記一行 `I` 級
日誌，不再每次請求都記警告。這類客戶端請改指向上面的 OpenAI 相容 base URL。

請求參數：`temperature`、`top_p`、`top_k`、`min_p`、`seed`、`max_tokens` / `max_completion_tokens`、`stop`、
`stream`、`tools`、`tool_choice`、`chat_template_kwargs.enable_thinking`（或最上層的 `enable_thinking`）、
`reasoning_effort`、`preserve_thinking`。沒有給取樣參數時套用模型卡建議值。`temperature` 為 0 時，輸出與
`whirl chat` 逐 token 相同。`presence_penalty` / `frequency_penalty` 會接受但忽略；`n` 必須為 1；不支援
`logprobs`、`response_format`，`tool_choice: "required"` 也不會強制。`stop` 最多 16 個字串、每個最多
256 位元組（超過回 HTTP 400）；`temperature` 小於 0.001 視為 greedy。回應另含 `usage.cached_tokens` 與
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

### <a id="prefix-cache"></a>前綴快取：新請求會重用什麼

DeltaNet 層只能從伺服器存過**檢查點**的位置接著跑（細節：[kv-and-caching.md](kv-and-caching.md#checkpoints)）。每個 slot 保存：提示詞結尾（`prompt-end`：同一個提示詞再來一次，例如重新生成）、回答結尾（`gen-end`：聊天或代理的下一回合），以及 0.2.0 起每個 4096 token 以上提示詞結尾附近的**尾端檢查點**。`--precise` 下（0.2.2 起）下一回合不從 `gen-end` 接續，而是從上一個提示詞的尾端檢查點（precise 下每個提示詞都保留一個）接續並重新 prefill 其餘部分，輸出與冷算逐位元相同；balance／fast 則從 `gen-end` 接續（較快，與冷算數值等價、非逐位元）。跨 session 共用的有：長 system 訊息，以及與先前 session 共同的前綴。

**長文件問答。** 對同一份長文件問好幾個問題、每題一個請求（同一則 user 訊息裡放「文件＋問題」、沒有 system 訊息）：第二題會從第一題的尾端檢查點（最後一個 1024-token chunk 的起點，距提示詞結尾 257 到 1280 個 token）接著跑，只 prefill 剩下的部分，而不是整份文件。128k 文件約為 1.1k token 取代 131k。條件：問題只在該點之後不同（約 256 token 以內的問題一定符合；更長的問題，或把文件放進 2048 token 以上的 system 訊息），且每個 slot 保留 3 個以上檢查點（預設：1 個 slot 時 4、2 到 4 個 slot 時 3）。輸出與不用快取時相同：尾端檢查點落在冷啟動的 chunk 排程上（precise 與 balance 與冷啟動逐位元一致）。每個 slot 檢查點佔 VRAM：Qwen3.8-27B 150.6 MiB、Ornith-1.5-35B-A3B 63.8 MiB；啟動日誌會列出數量與大小，VRAM 估算在 KV pool 會太小時把數量降到 2（尾端檢查點關閉）。`WHIRL_TAIL_CK=0` 可關閉（除錯用）。

### <a id="ssd-tier"></a>SSD 層如何儲存 session

- **何時寫入。** `--kv-ssd-mode shutdown`（預設）：只有「重用過的項目」才會寫進 SSD——前綴至少被還原過一次、
  被後續回合接續過的 session，或其他請求共用過的 system 訊息 checkpoint，且至少 2048 token。寫入時機有三種：
  關閉 WHIRL 時（Ctrl+C、關閉主控台視窗、登出、作業系統關機），下次啟動即可使用；主記憶體層滿了必須擠出它時
  （在背景寫入而不是直接丟掉，受下面的容量上限限制）；伺服器約 5 分鐘沒有請求（沒有執行中、也沒有新來的）時，
  在背景一次寫一筆——新請求一到就停止（最多只等已經在寫的那一筆），持續閒置期間不會再寫。一次性的 prompt
  在以上任何情況都不會寫入：它只留在主記憶體層，重啟後就沒了。Ctrl+C 會完整寫完（最多 60 秒）；關閉主控台視窗、
  登出、Windows 關機只有約 5 秒（之後 Windows 會強制結束行程），待寫的量大時可能只寫入一部分——要保證完整寫完請用
  Ctrl+C。當機時只在主記憶體裡的內容會遺失。
  `always`：項目連續 `--kv-ssd-delay` 秒（預設 300，即 5 分鐘）沒有變動後才寫入。
  對話在這段時間內又來了下一個請求，就直接從主記憶體服務，不產生任何磁碟寫入；主記憶體吃緊時，還沒到時間的項目若有至少 2048 token
  就先寫入，否則直接丟棄。`always` 只寫值得保留的 session：前綴至少被重用（還原）過一次，或已被後續回合接續過；沒有人再回來用的一次性 prompt 不會寫入（benchmark 連續送不同 prompt 幾乎不寫）。正常關閉時由最近使用的 session 開始寫。`off`：完全不寫 SSD 檔案（主記憶體層照常運作）。原因：benchmark 或 agent 連續送出大量不同的長 prompt 時，
  每個請求會寫數 GB，幾分鐘就是數百 GB，會損耗 SSD。
- **容量上限與過期。** 資料夾實際佔用的磁碟空間（以真正配置的位元組計，稀疏檔中共用的區塊是洞，不重複計算）不會超過
  `--kv-ssd-gb`（預設 64 GiB）。滿了就刪最久沒用的 session（連同借用其區塊的檔案）；其他模型、版本或 KV 格式的檔案最先刪，
  用另一個 per-slot context（`--ctx`／`--parallel`，或依空閒 VRAM 自動縮小）寫的檔也一樣：這些檔會保留、計入容量，
  換回那個 context 的 server 會再用到。主記憶體裡的 session 若檔案放不進容量上限就不寫入（只留在主記憶體，log 記一次）。
  超過 7 天沒被使用的檔案會自動刪除（啟動時與每小時一次）；無法使用的檔案（當機寫一半、損毀、不是 WHIRL 快取檔）在啟動時刪除。
  啟動 log 有一行：檔案數、佔用 GiB、容量上限，以及刪了什麼、為什麼。
- **只寫有變動的部分。** 續接的對話只寫新的 KV page 與變動的 checkpoint，不會整段重寫。SSD 上其他檔案已存有、
  且 token 前綴相同的 KV 區塊（共用的系統提示與工具定義、retry 或 compact 之前的部分）不再重寫：新檔案只記錄引用
  （稀疏檔，該區塊從另一個檔案讀取）。只有 token 前綴相同且 128-bit 內容雜湊相同的區塊才會共用。
- **被取代的 session。** agent 刪減歷史時（同一個系統提示和第一則任務訊息、中間被刪掉、新對話明顯較短），
  在新對話完成第二輪、且舊 session 期間沒有再被使用時，把舊 session 截到共用的部分。新對話只和舊 session 共用系統提示時
  （用摘要取代歷史的 compact，或只是同一個系統提示的另一段對話，例如主 agent 等待時在跑的 subagent），舊 session 不會被截斷：
  整段保留，只在主記憶體層或 SSD 層需要空間時最先被逐出（主記憶體逐出時也不寫入 SSD）；之後再被使用就恢復正常。
  retry、regenerate、修改最後一則訊息、不同系統提示都不會觸發。`WHIRL_KV_SUPERSEDE=0` 可關閉。
- **太短的 session**（少於 2048 token，`WHIRL_KV_TIER_MIN`）不保留：重新 prefill 和還原差不多快。
- **安全性。** 快取以 token 前綴精確比對；還原的就是當初存下的位元組。每個檔案有含校驗的 token 清單、header 最後才寫
  （前後有 flush 屏障，當機或斷電不會留下有效 header 蓋在其他位元組上）；被截短的檔、引用的共用區塊已不存在或與記錄的雜湊不符的檔、
  其他版本／模型／模式／KV 格式的檔，一律不會還原（其他版本的檔計入容量並優先刪除）。KV 與 checkpoint 資料本身讀取時不做校驗：
  這些檢查擋得住當機、斷電、截斷與過期的檔，擋不住檔案內容被 bit rot 或壞軌悄悄改掉；這類故障仰賴磁碟與檔案系統本身。
- **一個資料夾一個 server。** 第二個 server 指到使用中的資料夾（`whirl.lock`）時會不使用 SSD 層並在 log 說明；
  請給每個 server 各自的 `--kv-ssd-dir`。
- **寫入量。** log 的 `kv tier |` 行與 `GET /props`（`kv_ssd.bytes_written`、`bytes_shared`、`bytes_used`、`entries`、
  `superseded`）會顯示啟動以來寫入 SSD 的位元組，以及以共用取代寫入的位元組。
- **手動刪除。** 先關閉 server，刪掉資料夾中的 `*.wkv` 檔（或整個資料夾）即可，下次從空的開始。這裡不存放其他東西。
- **隱私。** 這些檔案以 token id 形式保存你的 prompt（能讀檔的人就能還原文字）與其 KV 資料，放在你的使用者設定檔目錄、
  沿用其權限。要刪除：先關閉 server 再刪掉該資料夾；`--kv-ssd-mode off` 則從不寫入。

### <a id="stop"></a>停止伺服器

按 **Ctrl+C**（或 Ctrl+Break，或關閉主控台視窗；登出與 Windows 關機也一樣），伺服器會正常收尾：

1. 新的與排隊中的請求回應 HTTP 503；
2. 執行中的請求跑完；
3. 已完成的 session 複製到主記憶體層，主記憶體層中 SSD 上還不是最新的項目立刻寫入——預設 `--kv-ssd-mode shutdown`
   寫所有重用過的項目（一次性的 prompt 不寫），最多 60 秒；`always` 寫它的「看過一次」過濾留下的項目，最多 10 秒
   （log 會出現 `shutdown: tier writes done …`）。

因此下次啟動時，最近的對話會從 SSD 還原，不必重新 prefill。再按一次 Ctrl+C 會立即結束。關閉主控台視窗、登出或 Windows 關機時，
不論是否做完，Windows 約 5 秒後都會結束行程，只寫得完這幾秒內的量（Ctrl+C 才會完整寫完）。直接強制結束行程（工作管理員、`taskkill /F`）會跳過以上全部
步驟；伺服器沒有關機用的端點。

## <a id="other"></a>5. `whirl devices`、`selftest`、`seqtest`、`vis-encode`

| 指令 | 用途 |
|---|---|
| `whirl devices` | 列出驅動程式回報的 AMD GPU（索引、名稱、架構、記憶體、運算單元）、這個建置有沒有該 GPU 的 kernel、是否為內顯，以及哪一張是預設。先用它確認驅動程式與 GPU |
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
| 5 | GPU 記憶體不足 | 權重或 KV 快取放不下：試試較小的 `--ctx`、減少伺服器 slot，或改用較小的量化 |
| 6 | 伺服器 port 已被占用 | 已有其他程式在 `--host`:`--port` 監聽（載入模型前就會檢查）：改用別的 `--port` |

執行檔在需要時才載入 AMD GPU runtime（`amdhip64_7.dll`，隨顯示卡驅動程式安裝），並在啟動時先檢查，所以驅動
程式不存在或太舊時會回報代碼 3，而不是跳出 Windows「找不到 DLL」的對話框。其餘部分（包括 C++ runtime）都是
靜態連結。

## <a id="env"></a>7. 環境變數

兩支程式都以 `WHIRL_<名稱>` 讀取每個變數。在
PowerShell 中先用 `$env:WHIRL_MODE = "precise"` 設定再啟動程式。**一般使用完全不需要設定**；預設值就是測試過、
最快的路徑。標示 *（A/B）* 或 *（診斷）* 的變數是給實驗與量測用的。

### 7.1 GPU 與載入

| 變數 | 說明 |
|---|---|
| `WHIRL_DEVICE=SPEC` | 使用的 GPU：`r9700`、`8060s`、索引，或名稱 / gfx 子字串（= `--device`；預設：第一張 R9700，否則第一張支援的 GPU） |
| `WHIRL_HIP_DEVICE=N` | 裝置索引，略過裝置比對與「一張 GPU 一個行程」的鎖 |
| `WHIRL_GPU_SHARE=1` | 不等待同一張 GPU 上的其他 WHIRL 行程 |
| `WHIRL_GPU_WAIT=S` | 等待其他 WHIRL 行程釋放 GPU 的秒數（預設 1800） |
| `WHIRL_MODE=precise\|balance\|fast[:項目]` | 數值模式（= `--precise` / `--balance` / `--fast` / `--mode`；預設 balance）。見[數值模式](#modes) |
| `WHIRL_PREFILL_BATCH=N` | prefill 批次列數（預設 4096，最多 16384）；CLI 與 server 用同一個排程，一次 forward 最多是這個值減 256 列（CLI-SCHED） |
| `WHIRL_MAX_CTX=N` | `chat` 的預設 context 大小（= `--ctx`；預設 8192） |
| `WHIRL_CODE_OBJECT=FILE` | 開發用：從這個 code object 載入 GPU kernel，取代內建的 |
| `WHIRL_MOE_FP8=0\|1` | 專家為 MXFP4 的 MoE 模型：專家 prefill 用 fp8 activation（項目 `moefp8`：balance 與 fast 開，precise 關；fp8 是較快的路徑——Ornith MXFP4 在 2k token 約 11.7k 對 8.6k tok/s）。只影響 prefill |
| `WHIRL_MOE_MXW=0` | 專家為 MXFP4 的 MoE 模型：改用通用的 MXFP4 專家 decode kernel，不用整塊（whole-block）kernel（預設整塊）。影響 decode / 驗證 |
| `WHIRL_MOE_WIDE=1` | MoE 模型（balance / fast）：verify 可到 32 列（預設 16），第 17–32 列和 1-token decode 一樣走 int8 decode 專家 kernel。輸出相同；Ornith-1.5-35B-A3B 四人時約慢 2%（每列各讀自己的專家），所以預設關 |
| `WHIRL_VRAM_LIMIT_MB=N` | 模擬只有 N MiB VRAM 的 GPU（例如在 R9700 上設 `16384` 模擬 16 GB 的 RX 9070 XT）：可用／總 VRAM、WDDM budget 與本行程的配置都以 N 為上限，載入、KV 池大小與自動縮小 prefill batch／checkpoint 數都會跟那張卡一樣。預設關閉 |
| `WHIRL_UMA_CHECK=0` | Radeon 8060S：略過「模型、緩衝區與 KV 快取是否放得進 Armoury Crate／BIOS 設定的 GPU 記憶體（專用 carve-out）」的檢查，以及讓 KV 快取留在其中的上限。不受支援：carve-out 以外的記憶體較慢，carve-out 太小時 GPU 執行階段會出現 hipError 719 |
| `WHIRL_DQ4_DYN=0` | Radeon 8060S、q4 KV（fast、MoE）：注意力改用固定的 split 格子，而不是只啟動有 key 的 split（預設開啟；兩種輸出相同） |
| `WHIRL_FAKE_DEDICATED_GB=N` | 測試用：假裝 Radeon 8060S 只有 N GiB 專用 GPU 記憶體（例如 `0.5`），不必改 BIOS 就能看到拒絕訊息或縮小後的 KV 快取。預設關閉 |
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
| `WHIRL_COLD_RESUME=0\|1` | 對話的快取只從與冷算 prefill 逐位元相同的檢查點接續（預設：`--precise` 開啟，其他模式關閉；precise 下設 `0` 會回到較快但非逐位元的 0.2.1 行為） |
| `WHIRL_SERVE_CKPTS=N` | 每個 slot 的前綴檢查點數（預設：1 個 slot 時 4、最多 4 個 slot 時 3、更多時 1；最多 8；3 個以上才保留[尾端檢查點](#prefix-cache)） |
| `WHIRL_SYS_MIN=N` | 至少 N 個 token 的 system 訊息會有自己的檢查點（預設 2048，0 = 關閉） |
| `WHIRL_SYS_CKPTS=N` | 保留在 VRAM 的共用前綴檢查點數（預設 2） |
| `WHIRL_SYS_LCP=0` | 不在多個 session 共同的前綴處建立檢查點 |
| `WHIRL_CKPT_HOST=1` | 前綴檢查點放在 pinned 主記憶體，不放 VRAM |
| `WHIRL_HIP_CACHE_LIMIT=0` | 保留 HIP runtime 自己對已釋放 GPU 記憶體的快取（R9700 實測：KV 池縮小 3.91 GiB 時它留住 0.87 GiB，Swift-1.5 27B）。預設 whirl-server 會替自己的行程設 `GPU_RESOURCE_CACHE_SIZE=256`（MiB；你自己設了就不動），讓它釋放的記憶體（縮小重配的 KV 池）真的還給 Windows（0.2.2.2）。R9700 實測（Swift-1.5 27B MXFP4、另一個程式佔 4 GiB）：有這個限制時，governor 重配 KV 池後的第一個請求是 17–69 tok/s（原本 89，3 次中 3 次），之後的請求 3 次中 2 次回到全速；設 `=0` 時縮小 3.91 GiB 只還回 3.04 GiB，且池長回去時超過預算（3 次中 3 次；這是 0.2.2.2 修正前的版本，修正後 governor 遇到這種長回去會立刻處理，尚未重測），所以 governor 開著時不建議設 `=0`。Radeon 8060S（這個限制是那裡唯一的改動）：對 0.2.2.1 的 2k／32k prefill 與 decode 三輪平均 -1.9% 到 +0.6%（Swift 2k prefill 有一輪 -5.7%），視覺編碼器閒置卸下後的第一張圖在 8060S 上未量測。R9700 加 `--mmproj`：最終 0.2.2.2 版本上編碼器閒置卸下後的第一張圖待重測，詳見 CHANGELOG |
| `WHIRL_GOVERNOR=0` | 關閉伺服器執行中的記憶體調節（0.2.2.2）：WDDM 預算在執行中下降時，它會釋放快取、讓新請求排隊、所有 slot 閒置時以較小的大小重新配置 KV 池，預算恢復後再長回來（見 server.md 第 7 節） |
| `WHIRL_POOL_BUDGET_MARGIN_MB=N` | 決定 KV 池大小與事後重查時，在 Windows（WDDM）顯示記憶體預算之下替其他程式成長留的空間。預設取 2048 MiB（GPU 接螢幕時 3072）與預算 6% 的較大者。啟動時其他程式的 VRAM 中預算還沒反映的部分另外保留 |
| `WHIRL_POOL_RESERVE_MB=N` | 舊的「只看可用 VRAM」規則：決定 KV 池大小時保留不用的 VRAM，同時關掉 WDDM 預算、margin 與池重查（預設保留 768，MoE 1536）。要替其他程式多留空間，請用 `WHIRL_POOL_BUDGET_MARGIN_MB` |
| `WHIRL_VRAM_HEADROOM_MB=N` | 在保留量與 WDDM margin 之外，再額外留給桌面與其他程式的 VRAM。預設 0（0.2.2.1 起由 WDDM margin 涵蓋桌面：接螢幕時 3 GiB；之前 VRAM 小於 20 GiB 的卡預設 1536）。VRAM 小於 20 GiB 的卡（16 GB 的 RX 9070／9070 XT／9060 XT）仍預設 `--parallel 1`（KV 格式跟其他卡一樣由模式決定） |
| `WHIRL_PREFILL_CHUNK=N` | 合併後單次 prefill forward 的最多列數（1024 的倍數，預設 2048，Radeon 8060S 上 MoE 模型為 4096；上限為 prefill batch） |
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
| `WHIRL_KV_SSD_DELAY_MS=MS` | `always` 模式下主記憶體層項目延後多久再寫入 SSD（預設 300000；`--kv-ssd-delay` 優先） |
| `WHIRL_KV_SSD_MODE=always\|shutdown\|off` | = `--kv-ssd-mode`（預設 shutdown） |
| `WHIRL_KV_SSD_EXPIRE_S=S` | 超過 S 秒沒用的 SSD 快取檔刪除（預設 604800 = 7 天；測試用） |
| `WHIRL_KV_SSD_DEDUP=0` | 即使其他 SSD 檔案已有相同區塊也照寫（預設：共用） |
| `WHIRL_KV_SUPERSEDE=0` | 保留被 agent 刪減取代的 session（預設：連第一則任務訊息都共用時截到共用部分；只共用系統提示時只是最先逐出） |
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

### <a id="env-numerics"></a>7.6 數值與速度開關 *（A/B）*

有損的開關就是[數值模式](#modes)的項目：預設值跟著模式走，在這裡設定會覆寫模式。

保留給 A/B 測試與數值比對的替代路徑；預設值就是測試過的路徑。「位元相同」的替代路徑算出的位元一樣，只差速度。

| 變數 | 說明 |
|---|---|
| `WHIRL_FP8=0\|1` | MXFP4 prefill 用 fp8 activation（項目 `fp8`；balance / fast 預設開）。`0` 也會關掉 `WHIRL_MOE_FP8` |
| `WHIRL_FP8_MASK=BITS` | 使用 fp8 activation 的矩陣乘法類別（預設 7） |
| `WHIRL_G8T=0` | fp8 GEMM 改用列優先版本，不用 fragment 排列版本（位元相同） |
| `WHIRL_GDN_WMMA=0\|1` | DeltaNet prefill 區塊用 f16 WMMA（項目 `gdnwmma`；balance / fast 下 MXFP4 模型，以及 R9700 上任何格式的 dense 模型預設開啟） |
| `WHIRL_Q8DEC=0\|1` | decode / verify activation：`1` 用 int8（項目 `q8dec`，balance / fast 預設；precise 下屬於有損覆寫），`0` 用 f16 進 f16 GEMM（precise 預設） |
| `WHIRL_FFN_H16=0\|1` | GEMM 以 f16 輸出給逐元素運算（項目 `h16`；balance / fast 下 MXFP4 模型，以及 R9700 上任何格式的 dense 模型預設開啟） |
| `WHIRL_Q4_RELAXED=1` | 原本略過的模型（非 MXFP4 的 MoE、Radeon 8060S）也套用 balance 模式的 prefill 開關（`gdnwmma`、`h16`） |
| `WHIRL_ACT_FUSE=0` | prefill 不融合 activation（位元相同） |
| `WHIRL_GEMMH=0` | 不用 f16 輸出的 prefill GEMM（位元相同） |
| `WHIRL_GEMMHQ=1` | attention 投影也用 f16 輸出的 GEMM |
| `WHIRL_ATTN_KX=0` | prefill attention 不用 K-exchange kernel（位元相同） |
| `WHIRL_ATTN_KG=0` | prefill attention 不用依 GQA 分組的 kernel（位元相同） |
| `WHIRL_ATTN_DQF=0` | q8 / q8h（與 q4）KV 的 prefill attention 直接讀量化快取（`attn_kg_q8`；Radeon 8060S 為 `attn_prefill_wmma_q8` / `_q4`），不先把序列的 key 轉成 f16 副本再跑 f16 kernel（位元相同） |
| `WHIRL_ATTN_SEG=N` | Radeon 8060S：prefill attention 依每 N 個 key（256 的倍數）分段，段間帶 softmax 狀態（`attn_prefill_wmma_s`），f16 KV 也適用（位元相同；測試用旋鈕，端到端沒有加速） |
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
| `WHIRL_ATTN_DQ4=0` | q4 KV：decode / 驗證 attention 改回 `attn_wsplit1_q4`（較慢，0.2.0 之前），不用 `attn_dq4` |

### <a id="env-debug"></a>7.7 診斷與除錯

| 變數 | 說明 |
|---|---|
| `WHIRL_KV=f16\|q8\|q8h\|q8v\|q4` | *（除錯）* 強制使用某種 KV 快取格式，取代模式的選擇（[數值模式](#modes)）；`auto` 或未設定 = 由模式決定。強制的格式不做容量檢查（pool 依剩餘記憶體決定），precise 模式下設成量化格式會記為使用者要求的有損覆寫。GPU 沒有對應 kernel 的格式會被拒絕。見 [kv-and-caching.md](kv-and-caching.md#formats) |
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
| `WHIRL_TAIL_CK=0` | *（除錯）* 長提示詞結尾附近不保留[尾端檢查點](#prefix-cache)（日誌標示為覆寫）；同一份文件的第二個問題會重新 prefill 整份文件 |
| `WHIRL_TIER_MIN_GAIN=N` | 只有能省下至少 N 個 token 時才從 host 層還原（預設 512） |

## <a id="files"></a>8. WHIRL 會寫入的檔案

| 路徑 | 內容 |
|---|---|
| `%LOCALAPPDATA%\whirl\tune4-<模型檔名>-<檔案大小>…txt` | 每個模型檔與 GPU 調校好的 prefill GEMM 組態。刪掉就會重新調校 |
| `%LOCALAPPDATA%\whirl\kvcache\` | 伺服器前綴快取的 SSD 層（`--kv-ssd-dir`，上限 `--kv-ssd-gb`） |
| `%LOCALAPPDATA%\whirl\server.log` | 伺服器 log（`--log-file`） |

執行檔旁邊不會寫入任何東西，也不會安裝任何系統層級的元件。
