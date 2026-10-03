[English](../en/architecture.md) | **繁體中文**

# 架構

> **狀態（WHIRL 0.1.0）。** 本文件描述的是本儲存庫中已實作的設計：C++20 host 程式碼與 HIP C++ device 程式碼，
> 目標為 R9700（gfx1201），以及預覽版（結果正確、尚未調校）的 Radeon 8060S（gfx1151）。部分數字是在 C++ 引擎之前的研究建置上量測的；
> C++ 引擎的輸出與其逐 token 相同（[benchmarking.md](benchmarking.md#gates)）。

**對誰有幫助：**正在評估「針對特定模型、特定 GPU 的引擎」是否值得打造的人，以及想知道 WHIRL 各部分放在哪裡、為何長成這樣的人。

## 1. WHIRL 是什麼

WHIRL（*Windows HIP Inference for RDNA LLMs*）是以 C++20 與 HIP C++ 撰寫、針對 AMD RDNA GPU、在 Windows 上原生執行的 LLM 推論引擎。它有兩個決定性的特點：

1. **它走特化路線。** WHIRL 遵循 [NInfer](https://github.com/Neroued/ninfer)（Apache-2.0）的理念：與其做一個每個模型都跑得還可以的通用引擎，不如打造一個讓*選定*的一組模型檔（checkpoint）在*選定*的一組 GPU 上盡可能逼近硬體極限的引擎。[zynfer](https://github.com/thanos/zynfer) 把同樣的想法帶到 RDNA 4。WHIRL 只共享理念，不含這兩個專案的任何程式碼或文字。
2. **它是純原生 Windows。** 沒有 WSL、沒有 Linux VM、沒有 llama.cpp runtime。Windows 上的 HIP SDK 跑在 AMD 的 PAL 驅動程式堆疊上，其行為與 Linux 的 ROCr/KFD 堆疊不同，而這些差異對效能工作很重要。這些差異記錄在 [windows-hip.md](windows-hip.md)。

WHIRL **不是**「Qwen 引擎」。它一次加入一種模型架構，並針對每種架構、每張 GPU 調校每個 kernel。第一個以這種方式完成的家族是 Qwen3.5 世代的混合架構（`qwen35` 密集模型、`qwen35moe` 混合專家（MoE））。其他任何 `general.architecture` 值都會在載入時以 `UnsupportedArch` 拒絕；沒有會默默跑得更慢或產生不同數值結果的通用後備路徑。

### 目標硬體

| GPU | ISA | 角色 | 實測極限（我們自己的探測，非規格表） |
|---|---|---|---|
| AMD Radeon AI PRO R9700（RDNA 4，Navi 48） | gfx1201 | 主要目標 | 32 GB GDDR6，256-bit，規格 640 GB/s；GEMV 串流 604–626 GB/s；WMMA f16→f32 180–184 TFLOPS；WMMA iu8 343–374 TOPS |
| AMD Radeon 8060S（Ryzen AI Max，RDNA 3.5） | gfx1151 | 次要目標（預覽、未調校） | 統一記憶體；串流讀取 238 GB/s；WMMA f16 43 TFLOPS；WMMA int8 45 TOPS（與 f16 同速率） |

開發用的 R9700 是以 **USB4 eGPU** 連接。其主機連結實測為 3.78–3.81 GB/s（D2H）與 3.84–3.86 GB/s（H2D）。在這台機器上，所有在主機與 GPU 之間搬移資料的動作（載入、RAM/SSD KV 層、視覺權重串流）都受限於這條連結。多數使用者有直接的 PCIe 插槽，主機頻寬會高得多；我們在整份文件中把 eGPU 造成的效應標示為*環境限制*，且沒有為它們撰寫變通做法。

### 模型家族 1：`qwen35` / `qwen35moe`

| | Qwen3.8-27B（`qwen35`） | Ornith-1.5-35B-A3B（`qwen35moe`） |
|---|---|---|
| 層數 | 64 + 1 層 MTP（`blk.64`，nextn） | 40 + 1 層 MTP |
| 層模式 | Gated DeltaNet ×3、gated attention ×1，重複（48 DeltaNet + 16 attention） | 相同模式（30 DeltaNet + 10 attention） |
| Embedding | 5,120 | 2,048 |
| Attention | 24 heads / 4 KV heads，head_dim 256，前 64 維套用 NEOX RoPE，base 1e7，sigmoid 輸出 gate | 16 heads / 2 KV heads |
| Gated DeltaNet | 16 個 k-head、48 個 v-head，head dim 128，conv kernel 4，q/k 做 L2 正規化；v-head h 使用 k-head h % 16 | 相同結構，較小 |
| FFN | 密集 SwiGLU，n_ff 17,408 | 256 個專家，top-8（每個寬 512）+ 1 個共享專家（512），帶 sigmoid gate |
| 詞彙表 | 248,320 | 相同 tokenizer（在我們 110 個檔案的對等性語料上 token 數完全相同） |

混合架構正是這個家族有趣又困難之處：每個 decode（逐 token 生成）步驟都必須同時推進 attention 的 KV 快取與遞迴的 DeltaNet 狀態。遞迴狀態無法靠截斷快取來「回捲」，這一點形塑了推測解碼（[speculative-decoding.md](speculative-decoding.md)）與前綴快取（[kv-and-caching.md](kv-and-caching.md)）的設計。

## 2. 管線

```
 GGUF file ──► GGUF reader ──► loader ──► device weights (+ per-load requantized copies)
                (mmap, v2/v3)   (type check,  │
                                reorder,      ▼
                                requant)   forward pass ◄── kernels (one code object per GPU arch)
                                             │  prefill: WMMA GEMM, flash attention, chunked DeltaNet
                                             │  decode : int8 GEMV, split-K attention, recurrent step
                                             ▼
 tokenizer + chat template ──► scheduler (CLI or server) ──► sampler ──► text / tool calls
                                  │
                                  └─ KV pool (paged), DeltaNet checkpoints, RAM/SSD tiers
```

### 2.1 GGUF 讀取器 — `src/gguf/`（已在儲存庫中）

透過記憶體映射讀取 GGUF v2 與 v3：支援所有中繼資料值型別、巢狀陣列、對齊，以及完整的 ggml 型別表（包含 `mxfp4`、`nvfp4`、`iq*` 與 `tq*` 家族）。已在 13 個檔案上與 gguf-py 0.19.0 比對驗證（每個 key、value、tensor 名稱、型別、形狀、偏移與位元組大小都相同）。15.3 GiB 的 Qwen3.8 檔案（866 個 tensor）約 7 ms 就能解析完，因為完全不做複製。

### 2.2 Tokenizer 與聊天範本 — `src/tokenizer/`、`src/chat/`（已在儲存庫中）

- `qwen35` pre-tokenizer 切分 + byte-level BPE，直接從 GGUF 詞彙表載入（247,587 條 merge，耗時 0.10–0.15 s）。預切分以及特殊 token/BPE merge 迴圈改寫自 llama.cpp（MIT），並在 [THIRD_PARTY_NOTICES.md](../../../THIRD_PARTY_NOTICES.md) 中標註出處。
- Unicode 表由我們自己的腳本產生，並以 **Unicode 15.1** 等級輸出以與 llama.cpp 一致（見 [pitfalls.md](pitfalls.md) 中的坑 *Unicode 15.1 vs 16.0*）。
- 對等性：440 種檔案/模型/模式組合，約 150 萬個 token，與 `llama-tokenize` 完全相同；28 個渲染後的聊天提示詞完全相同。
- 聊天範本渲染器重現兩種受支援範本變體的 GGUF Jinja 範本（工具、思考開/關、`preserve_thinking`、`reasoning_effort`、`tool_choice`），並逐案與 llama.cpp 的 Jinja 引擎比對。

Tokenizer 不是小細節：原型中一個空白切分的 bug，讓每個有縮排的程式碼提示詞都被切成模型從未見過的 token 序列。這是本專案中最具教育意義的單一 bug；見 [pitfalls.md](pitfalls.md#tok-1)。

### 2.3 載入器 — `src/model/loader.cpp`

- 讀取 `general.architecture`；只接受 `qwen35` 與 `qwen35moe`。
- 將每個 tensor 型別與該 GPU 的 kernel 集合比對。未實作的型別（例如 NVFP4）會中止載入；沒有「反量化成 f16 然後碰運氣」的後備做法。
- 對 GGUF 配置不適合 kernel 的格式重新排列。MXFP4 區塊（32 個值 = 一個 E8M0 指數 + 16 位元組）以列為單位重組成 256 值的 super-block `e[8] | qs[8][16]`（136 位元組，與 IQ4_XS 大小與對齊相同），並為 fp8 prefill（提示詞預填）路徑計算每列的參考指數（[quantization.md](quantization.md)）。
- 在 GPU 上建立每次載入時衍生的副本：
  - 從 Q6_K 輸出 head 重新量化的 **2-bit 草稿 head**（「D2」，2.5 bpw），只用來挑選 MTP 草稿；
  - **MTP 區塊的 Q4_K 副本**（密集模型），讓草擬更便宜。
  兩者都只影響*提出哪些 token*；每個輸出的 token 都由完整模型驗證，因此輸出不變。
- 以讀取檔案再上傳的方式載入，而非把檔案映射進 GPU。（llama.cpp 預設的 mmap 載入在這個驅動程式堆疊上會失敗；見 [windows-hip.md](windows-hip.md#mmap)。）

### <a id="kernels"></a>2.4 Kernel — `kernels/`（每種 GPU 架構一個 code object）

主機程式碼由 MSVC 編譯。裝置程式碼由 HIP SDK 的 clang 編譯成每種架構（`gfx1201`、`gfx1151`）各一個 code object，由 `tools/bin2c` 轉成位元組陣列，並嵌入執行檔。執行時以 `hipModuleLoadData` 載入與所選裝置相符的 code object。

為何用每張 GPU 各自的 code object，而不是一套可攜的 kernel：

- **矩陣指令不同。** gfx12 WMMA 把 A/B fragment（片段）緊湊地保存在暫存器中，且其 C/D fragment 配置等同於 B 的配置，我們的 DeltaNet scan 與 flash attention 正是利用這一點。gfx11（RDNA 3.5）WMMA 需要在兩個半 wave 中都複製一份 B fragment，並交錯輸出列。為其中一種寫的 kernel，在另一種上不是錯就是慢。
- **成本平衡不同。** 在 R9700 上 int8 WMMA 是 f16 的 2×；在 8060S 上兩者同速率，而且 gfx11 WMMA 會與 VALU 工作互相競爭。在一張 GPU 上勝出的設計，在另一張上可能落敗（例如 W8A8 prefill — 見 [kernels.md](kernels.md)）。
- **選用 kernel。** 有些 kernel 只存在於某一種架構（int8 WMMA 中批次 GEMV、MXFP4×fp8 GEMM、`q8v` KV 格式只存在於 gfx1201）。主機以 `getFunctionOpt` 查找它們；缺少的 kernel 會回傳 null 並清除 HIP 黏著的 last error，接著在載入時改走後備路徑或拒絕載入。
- **看能力旗標，不看架構名稱。** 光靠「kernel 不存在」無法決定的地方，會從載入的 code object 探測一次 `kernels::Caps`（`include/whirl/kernels_abi.h`）：`fp8_gemm`、`kv_q8v`、`kv_q8h`、`gemvw`、`gdn_replay`、`mrope`，以及只有 gfx1151 object 才有的兩個標記 kernel —— `xd_sum`（int8 activation 的 scale word 內含區塊總和）與 `attn_group1`（其 WMMA P·V 只有每個 attention 群組一個 query 時才逐位元精確）。KV 格式自動選擇、attention query 分組、DeltaNet replay 與 kernel 測試都讀這些旗標；剩下的架構判斷只用來選調校表、tune 快取檔名與視覺 code object（只有 gfx1201）。

預設裝置：build 含 gfx1201 kernel 時用第一張 R9700，否則用第一張 build 有 kernel 的 GPU（例如只有 Radeon 8060S 的機器）；`--device` / `WHIRL_DEVICE` 可指定其他裝置（`whirl devices` 會標出預設）。

檔案切分：`kernels/common.hip`（型別、wave 基本操作、區塊解碼器）、`gemv_*.hip`、`gemm_prefill.hip` / `gemm_fp8*.hip` / `gemm_small.hip`、`attn*.hip`、`gdn_*.hip`（DeltaNet）、`moe.hip` / `moe_mxfp4.hip`、`fused_decode.hip`、`misc_*.hip`、`sample_*.hip`、`draft_d2.hip`，全部由 `kernels/whirl_kernels.hip` 引入；視覺編碼器的 kernel 是 `kernels/vision/` 下另一個 code object。gfx1151 的 kernel 集在 `kernels/gfx1151/`（kernel 名稱與參數 ABI 相同、自己的檔案切分，只在編譯 gfx1151 時由 `whirl_kernels.hip` 引入）；它缺少的家族列在 [kernels.md](kernels.md#gfx1151)。

### 2.5 前向傳遞 — `src/model/`

Prefill 與 decode 是不同的程式：

| | Prefill | Decode |
|---|---|---|
| 受限於 | 運算 | 記憶體頻寬 |
| 矩陣乘法 | WMMA GEMM（f16，或 MXFP4×fp8），依（型別、形狀、批次區間）自動調校 | 使用 `v_dot4` 的 int8 GEMV；3–16 列時用 int8 WMMA |
| Attention | WMMA flash attention，Sᵀ = K·Qᵀ | split-K flash decoding；WMMA 分組驗證 |
| DeltaNet | 分塊（64 token，WY 形式），f32 或 f16-WMMA | 每個 token 一個遞迴步驟，與 gated RMSNorm 融合 |
| 主機角色 | 以 ≤ 1024–2048 列為一塊排入佇列 | 提前排入數個步驟；argmax kernel 在 GPU 上寫入下一個 token 與位置 |

Decode 狀態存放在 GPU 上。Argmax kernel 把下一個 token id 與位置直接寫入裝置記憶體，因此主機每一步都不必讀回，可以排入數個步驟；一整個 MTP 草擬加驗證循環只需要一次主機同步。各 kernel 家族見 [kernels.md](kernels.md)。

### 2.6 排程器、取樣器與 server — `src/server/`、`src/tier/`

相容 OpenAI 的伺服器（server）負責：最多 16 個 slot（預設 4）的連續批次處理、共享的分頁 KV 池、用於前綴重用的 DeltaNet 檢查點、VRAM → pinned RAM → SSD 的 KV 層、每個 slot 的推測解碼（MTP + n-gram）、取樣、工具呼叫解析與 `reasoning_content`。見 [server.md](server.md) 與 [kv-and-caching.md](kv-and-caching.md)。

## 3. 為何特化 — 以量測為據

特化引擎只有在特化效果反映在通用引擎於同一硬體上達不到的數字時，才值得它的成本。以下在同一張 R9700、同一個 GGUF、greedy decoding、兩個引擎各自透過自家相容 OpenAI 的 server、一次只跑一個模型行程的條件下量測（完整流程見 [benchmarking.md](benchmarking.md)）：

| Qwen3.8-27B，tok/s | llama.cpp b11214 ROCm，Q4_K_M | WHIRL，Q4_K_M | WHIRL，MXFP4 |
|---|---|---|---|
| Prefill 8k token | 1,177 | 1,747–1,751 | 3,548–3,557 |
| Decode，預設模式（MTP + n-gram） | N/A | 145.4 | 159.1 |
| Decode，MTP | 56.9 | 110.4 | 120.3 |
| Decode，無 MTP | 30.9 | 34.3 | 37.6 |

Decode 各列：兩個引擎使用相同的統一 server 流程（19 個提示詞平均）。WHIRL 的 prefill 列是 CLI（KV f16）；WHIRL *server* 在 8k 的冷 prefill 為 1,560（Q4_K_M）/ 3,235（MXFP4）tok/s，仍大幅領先。細節與注意事項見 [benchmarking.md](benchmarking.md#current-numbers)。

差距的來源，依大小排序：

1. **圍繞精確性設計的推測解碼。** WHIRL 的多列 GEMV 產生的結果與 1 列 GEMV 逐位元相同，因此驗證 4 個草稿只比 decode 1 個 token 多約 15% 成本，而且 MTP 輸出與單純 greedy 輸出完全相同。這讓自適應策略最多可使用 8 個 MTP 草稿（加上 n-gram 草稿時為 15 個），而 llama.cpp 使用固定上限 3 個。
2. **為單一檔案量身打造的 kernel。** unsloth UD-Q4_K_M 檔案混用九種格式（Q5_K、IQ4_XS、Q4_K、Q6_K、IQ4_NL、Q3_K、IQ3_S、Q8_0、F32）。每種都有自己的解碼器與調校過的多列 kernel；最慢的兩種（Q3_K、IQ3_S）是在把每種型別與純讀取上限比對量測後才修好的。
3. **為硬體挑選的格式。** MXFP4 的 2 的冪次區塊縮放可以精確地併入 fp8 指數，因此 RDNA 4 的 fp8 WMMA 能以單一 epilogue、約 240 TOPS 執行 prefill GEMM。這就是「速度模式」（[quantization.md](quantization.md)）。

**單純 decode 有硬性上限。** 沒有推測時，每個 token 都必須把所有權重串流一次：Q4_K_M 檔案是 14.33 GB。WHIRL 的 1-token GEMV 已跑到實測純讀取上限（604–626 GB/s）的 97–100%，完整的 decode 步驟耗時 27.74 ms（Q4_K_M）/ 25.15 ms（MXFP4）。因此 27B 4-bit 模型在 640 GB/s 顯示卡上的無 MTP 上限約為 **38 tok/s**；kernel 優化只能再挖回幾個百分點。要更快，就得在每次權重傳遞中驗證多個 token — 這就是預設模式是 MTP + n-gram 的原因，也是本文件中 decode 數字以它為首的原因。

## 4. 正確性模型

兩種模式，依模型檔案選擇：

- **精確模式（Q4_K_M 及其他 K/IQ 量化）。** 凡前一版逐位元相同之處，每項優化都必須維持輸出逐位元相同。有損的變更（例如 8-bit KV）必須通過相對於實測路徑雜訊的 KL 散度檢查、成對問答測試（350 題，McNemar 精確檢定）以及長上下文 needle 測試。
- **速度模式（MXFP4）。** 接受 prefill 中的 fp8 activation（啟動值）；其影響會被量測並記錄（KL、成對 QA、perplexity），但不是阻擋性的關卡（gate）。

兩種模式下，以下不變式都由每次變更必跑的常設 gate 強制執行：

- MTP greedy == 單純 greedy（草稿數 1–10 全部，含與不含 n-gram 草稿）。
- 多列 GEMV == 1 列 GEMV，逐位元相同，涵蓋每種型別、列數 2–16 與每種 kernel 變體。
- 並行請求 == 同一請求單獨執行（greedy），包含分段 prefill。
- 在**沒有**任何環境變數、**沒有**任何選項下啟動的 server，產生與 CLI 相同的 greedy 文字（在一個只出現在預設路徑上的 bug 之後加入）。
- 從 RAM 或 SSD 還原的 KV == 從未離開 VRAM 的 KV，逐位元相同。
- 重用快取系統提示詞的新 session（工作階段）== 同一請求冷啟動執行。

[benchmarking.md](benchmarking.md) 列出這些 gate 及其執行方式。

## 5. 接下來讀什麼

| 主題 | 文件 |
|---|---|
| Windows 上的 HIP：驅動程式堆疊、記憶體、計時、行程紀律 | [windows-hip.md](windows-hip.md) |
| 每個 kernel 家族及其數字 | [kernels.md](kernels.md) |
| MTP、n-gram 共同草擬、精確性 | [speculative-decoding.md](speculative-decoding.md) |
| 分頁 KV、KV 格式、檢查點、分層快取 | [kv-and-caching.md](kv-and-caching.md) |
| OpenAI server 行為與限制 | [server.md](server.md) |
| 我們如何量測；目前數字 | [benchmarking.md](benchmarking.md) |
| 精確模式 vs 速度模式、MXFP4 量化配方 | [quantization.md](quantization.md) |
| 影像輸入 | [vision.md](vision.md) |
| 我們踩過的每個坑 | [pitfalls.md](pitfalls.md) |
