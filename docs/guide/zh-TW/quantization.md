[English](../en/quantization.md) | **繁體中文**

# 量化：精確模式、速度模式，以及為 WHIRL 製作 GGUF

> **狀態。** 本文描述的 kernel 支援已收錄在 WHIRL 0.1.0（gfx1201，R9700）。文中列出的 GGUF 檔案由我們製作並量測；
> Swift-1.5 MXFP4 檔案已發布於 Hugging Face，Ornith-1.5-35B-A3B MXFP4 檔案會在第一個發行版之後發布。

**對誰有幫助：** 為 RDNA 上的混合式 Qwen3.x 模型挑選量化格式的人、用 llama.cpp 的 `llama-quantize` 量化這類模型的人（下面有幾個不明顯的陷阱），以及想知道 MXFP4 在品質上實際付出多少代價的人。

## 1. 兩種模式

WHIRL 會以兩種方式之一看待一個模型檔案：

| | 精確模式 | 速度模式 |
|---|---|---|
| 典型檔案 | unsloth UD-Q4_K_M（K-quants + IQ quants） | MXFP4 |
| Prefill activation（啟動值） | f16 | fp8（e4m3），每個 token 一個 scale |
| 有損變更的規則 | 必須通過 KL 對路徑雜訊 + 成對 QA（McNemar）+ needles | 接受；量測並記錄影響 |
| 各項優化之間的輸出 | 前一版逐位元相同之處維持逐位元相同 | 速度路徑改變時可能改變 |
| MTP == 純 greedy、並行 == 單獨執行 | 是 | 是 |

當檔案包含 MXFP4 tensor 時，速度模式會自動套用：fp8 prefill（提示詞預填）GEMM、f16-WMMA DeltaNet chunk，以及供逐元素運算使用者讀取的 f16 GEMM 輸出。在 R9700 上，後兩項（balance 項目 `gdnwmma`、`h16`）也用在 Q4_K_M 等其他格式的 dense 模型（KG-1：Qwen3.8-27B Q4_K_M 8k–128k prefill 快 7–10%；128k 對 precise 的 KL 變化在 ±4e-5 內，為 2.7e-4／4.1e-4）；這些模型的 GEMM 仍用 f16 activation。`WHIRL_FP8=0` 會讓 MXFP4 prefill 回到 f16 activation（此時 prefill 只比 Q4_K_M 快 2–4%——速度來自 fp8，而非格式本身）。兩種模式下 decode（逐 token 生成）都是精確的（int8 GEMV，每個單元的算術相同）。

| R9700 上的 Qwen3.8-27B | Q4_K_M（精確） | MXFP4（速度） |
|---|---|---|
| 權重 | 15.3 GiB | 13.9 GiB（12.89 GiB MXFP4 + 0.97 GiB Q6_K head） |
| 預設 server KV 池（q8v，4 slots） | 199,936 tokens | 大約多 16%（加入系統檢查點後為 228,096） |
| CLI prefill 2k / 8k / 32k / 128k | 1,722 / 1,749 / 1,532 / 1,052 tok/s | 3,770 / 3,552 / 2,783 / 1,531 tok/s |
| Decode 無 MTP / MTP / MTP + n-gram（server） | 34.3 / 110.4 / 145.4 | 37.6 / 120.3 / 159.1 |
| Perplexity，中英文程式碼語料（llama.cpp） | 3.640 | 3.795（+4.3%） |
| 成對 QA，350 題（WHIRL，greedy） | 265（f16 KV 執行） | 256（fp8 prefill 執行） |

兩個 QA 數字來自不同的執行（並非成對比較）。在 MXFP4 內部，fp8 對 f16 prefill 有做成對比較：256 對 253，McNemar p = 0.58。

**選擇精確模式**：當輸出必須與 greedy 參考一致，且每項優化都必須通過精確度關卡（gate）時。**選擇速度模式**：用於長提示詞和會讀取大量檔案的 agent，這時 prefill 佔主導；decode 也快 8–10%（與權重位元組數成比例）。

## 2. WHIRL 接受的 tensor 類型

以下類型有 decode 與 prefill kernel：F32 和 F16（小型 tensor）、Q8_0、Q3_K、Q4_K、Q5_K、Q6_K、IQ3_S、IQ4_NL、IQ4_XS、MXFP4；視覺 projector 則有 BF16。任何其他類型（例如 NVFP4）會中止載入——沒有通用的反量化到 f16 的退路。MXFP4 已為 dense `qwen35` 與 MoE 專家（`qwen35moe`，[第 6 節](#ornith-mxfp4)）實作。在 gfx1151（8060S，預覽）上 MXFP4 走同樣的 int8 decode GEMV，prefill 先反量化成 f16（gfx1151 沒有 fp8 WMMA，所以那裡的 MXFP4 prefill 用 f16 activation；MoE 專家走通用路徑）。

### 2.1 unsloth UD-Q4_K_M 檔案包含什麼

一個檔案混用九種格式，這正是 WHIRL 逐檔調校的原因。每個 token 讀取的 decode 權重（共 14.33 GB）：

| 格式 | 位元組 | 1-token 時間 | 實際達到的頻寬 | 備註 |
|---|---|---|---|---|
| Q5_K | 4.834 GB | 8.33 ms | 580 GB/s | |
| IQ4_XS | 4.760 GB | 8.06 ms | 590 GB/s | |
| Q4_K | 3.489 GB | 6.12 ms | 570 GB/s | |
| Q6_K | 0.430 GB | 0.76 ms | 566 GB/s | 各層；輸出 head（1,043 MB）另計 |
| IQ4_NL | 0.330 GB | 0.57 ms | 584 GB/s | |
| Q3_K | 0.268 GB | 0.54 ms | 493 GB/s | 後來修正到約 580 GB/s |
| IQ3_S | 0.153 GB | 0.33 ms | 460 GB/s | 後來修正到約 580 GB/s |
| Q8_0 | 0.070 GB | 0.69 ms | 101 GB/s | 很小的 48×5120 DeltaNet beta/alpha 矩陣；已融合 |
| F32 | — | — | — | norm、conv 權重、bias |

（早期量測；token_embd Q4_K 715 MB 每個 token 只讀一列。）

## <a id="mxfp4"></a>3. WHIRL 中的 MXFP4

- **格式。** 每個 block 有 32 個 e2m1 值 + 一個 E8M0（2 的冪次）指數，17 位元組。
- **載入。** 每一列的 block 會重組為 256 值的 super-block `e[8] | qs[8][16]`（136 位元組——與 IQ4_XS 相同的大小與對齊，因此位元組數不變），並為 fp8 路徑儲存每一列的參考指數（最大的 block 指數）。載入器會計算指數比該列參考值低超過 8 的 block（它們在 fp8 折疊中會損失精度）並記錄在 log：整個 Qwen3.8 MXFP4 模型中有 56 個 block（在 7 個抽樣 tensor 上的指數差直方圖：差 0：1.7e7 個 block，1：3.3e7，2：1.4e6，…，7：120，8：1）。
- **Decode。** 與 IQ4_XS 相同的單元結構與浮點運算式，scale 為 `2^(e−128)`，並用 `v_perm` 查 e2m1 表。所有 GEMV 變體（1-token、多 token dp4、多列、int8 WMMA）都已具備，因此 MTP 維持逐位元相同。
- **Prefill。** MXFP4 × fp8 WMMA，E8M0 指數折疊進 fp8 權重（[kernels.md](kernels.md#mxfp4-gemm)）；每個 token 的 fp8 activation。

**fp8 prefill 的精確度**（最後一個 token 的 KL，MXFP4 f16-prefill 對 fp8-prefill，f16 KV）：arch1k 5.9e-5、zh-short 2.5e-4、p4k 3.6e-3、p12k 4.0e-5、p24k 3.5e-5、code16k 2.9e-2、code32k 5.6e-2（top-1 在接近平手處翻轉，p 0.265）。作為比較尺度：f16 prefill 用 chunk 1024 對 4096 的 KL 為 0（逐位元相同）；「整個提示詞逐 token decode（int8 activation）對 f16 prefill」為 2.4e-6 / 9.3e-5 / 6.4e-4（MXFP4）與 1.7e-6 / 5.7e-5 / 4.8e-3（Q4_K_M）；格式差異 Q4_K_M 對 MXFP4（兩者皆 f16）為 2.6e-4 … 1.4e-1。因此 fp8 增加的量約為 decode 路徑雜訊的 3–25 倍、格式差異的 5–45%。fp8 activation 也會放大任何其他擾動：同一個 f16 WMMA DeltaNet 變更在 Q4_K_M 上是 4.5e-8 … 1.5e-4，在 fp8 下變成 1.2e-4 … 7.0e-2。

`WHIRL_FP8_MASK`（位元：1 = attention/DeltaNet/MTP projection，2 = FFN gate/up，4 = FFN down）是為了研究哪一類 GEMM 貢獻誤差而存在；該研究尚未完成。

## <a id="ppl"></a>4. Perplexity

llama.cpp `llama-perplexity`（b11214 ROCm），中英文混合程式碼語料（219 KB），51 個 chunk × 2048 tokens：

| 檔案 | PPL |
|---|---|
| Qwen3.8-27B UD-Q4_K_M | 3.6403 ± 0.0355 |
| Qwen3.8-27B MXFP4（FreedomAISVR） | 3.7952 ± 0.0384（+4.3%；51/51 個 chunk 都較差；ΔNLL +0.0417 ± 0.0027 nats/token） |
| Swift-1.5 MXFP4 A（Q6_K head） | 3.8271 ± 0.0394 |
| Swift-1.5 MXFP4 B（Q8_0 head） | 3.8289 ± 0.0395 |
| Swift-1.5 MXFP4 C（Q4_K head） | 3.8386 ± 0.0396 |

Swift-1.5 是不同的微調模型，因此它與 Qwen MXFP4 的差距（A 約 +0.8%）並非全部來自量化。UD-Q4_K_M 與 MXFP4 之間的 PPL 差距在預期之內：參考用的 MXFP4 檔案幾乎全部是 4.25 bpw 且沒有 imatrix，而 UD-Q4_K_M 平均位元數較多（Q5_K 4.5、IQ4_XS 4.43、Q4_K 3.92、Q6_K 1.69 GiB，…）。（兩次 Qwen PPL 執行的共享 GPU 記憶體峰值為 306 MiB，我們的 VRAM 規則會把這標記為影響*速度*；數值不受影響。）

## <a id="swift"></a>5. 我們如何把 Swift-1.5-Qwen3.8-27B 量化為 MXFP4

已發布：[tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF)，附 F16 mmproj。

### 5.1 做法

1. **轉換為 BF16 GGUF，保留 MTP：** `convert_hf_to_gguf.py <BF16 dir> --outtype bf16` →
   866 個 tensor（54.6 GB），`qwen35.nextn_predict_layers = 1`。
2. **量化**，使用 llama.cpp b11214 的 `llama-quantize`。這個版本沒有 dense MXFP4 檔案類型，
   因此 ftype 用 `MXFP4_MOE`，搭配一個 tensor-type 檔案（第一條符合的規則勝出）以及明確的
   輸出類型：

   ```
   llama-quantize --imatrix <imatrix.gguf> --tensor-type-file tt_common.txt \
       --output-tensor-type q6_k swift-1.5-27b-bf16.gguf \
       Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf MXFP4_MOE 16
   ```

   `tt_common.txt`：

   ```
   blk\.64\.=q8_0
   ssm_alpha=q8_0
   ssm_beta=q8_0
   token_embd=q8_0
   ```

   其餘所有部分（attn_gate、attn_qkv、attn_q/k/v、attn output、ffn_gate/up/down、ssm_out）
   交給 MXFP4 ftype。變體 B 使用 `--output-tensor-type q8_0`，C 使用 `q4_k`。
3. **驗證 tensor 類型**：把結果 dump 出來檢查：MXFP4 = 64 層中每個大型矩陣；
   Q8_0 = token_embd、ssm_alpha/beta（48 × 2）、整個 MTP 層 `blk.64`（attention q/k/v/o、FFN、
   `nextn.eh_proj`）；F32 = norm、`ssm_a`、`ssm_conv1d`、`ssm_dt.bias`；output = Q6_K / Q8_0 / Q4_K。

| 變體 | 檔案 | 位元組 |
|---|---|---|
| A（建議） | `Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf` | 15,815,469,280 |
| B | `…-B-outQ8_0.gguf` | 16,123,386,080 |
| C | `…-C-outQ4_K.gguf` | 15,487,686,880 |

### 5.2 這個做法中的陷阱

- **imatrix 在這裡幾乎沒有作用。** 主線 `llama-quantize` 對 MXFP4（`GGML_UNUSED(quant_weights)`）和 Q8_0 會忽略重要性矩陣。只有 K-quant 輸出 head（A 的 Q6_K、C 的 Q4_K）會用到它。
- **參考用的 MXFP4 檔案全部都是 MXFP4**——包括 token_embd、ssm_alpha/beta 和 MTP 層——搭配 Q6_K head。我們把 embedding、MTP 和 alpha/beta 提高到 Q8_0。無論哪種做法，WHIRL 的速度模式都會套用。
- <a id="f32-alpha-beta"></a>**不要把 `ssm_alpha` / `ssm_beta` 存成 F32。** 第一版依照一份把它們保留為 F32 的規格：WHIRL 的 MTP smoke test 在三個變體上全部失敗（MTP 輸出 ≠ 純 greedy），草稿數被限制為 1。原因：WHIRL 沒有針對 F32 權重的 int8 GEMV，因此 n = 1 使用 f32-activation GEMV，而 n ≥ 2（驗證）使用 f16 GEMM——數值計算不同，破壞了 MTP 的逐位元相同。改用 Q8_0（它有融合的精確 kernel，unsloth 的 Q4_K_M 也是這樣用）後，所有變體都通過，草稿數回到 8。（自 0.1.3 起，引擎讓 F32／F16 小矩陣在所有 n 下都走同一個 GEMV 家族，這類檔案因此逐位元正確；但因融合的 DeltaNet decode kernel `gdn_ab` 只支援 Q8_0／MXFP4 的 α/β，草稿數仍上限為 1——見 [speculative-decoding.md](speculative-decoding.md#exact)。仍建議使用 Q8_0。）

### 5.3 結果（WHIRL server 預設值，統一流程，2 輪）

| | A（Q6_K head） | B（Q8_0 head） | C（Q4_K head） | Qwen3.8 MXFP4 參考 |
|---|---|---|---|---|
| Decode，MTP + n-gram（預設） | 157.2 | 143.6 | **157.5** | 156.9 |
| … zh think0 / think1 / edit | 106.0 / 87.9 / 325.7 | 90.6 / 80.4 / 306.3 | 101.1 / 90.0 / 330.8 | 107.0 / 83.3 / 329.5 |
| Decode，MTP | **117.0** | 98.0 | 112.0 | 118.1 |
| Decode，無 MTP | 36.8 | 36.1 | **37.5** | 36.9 |
| MTP 接受率（MTP / MTP + n-gram） | 65.4% / 66.3% | 73.2% / 72.9% | 68.3% / 69.1% | 60.6% / 62.2% |
| Prefill 2k / 8k / 32k | 3164 / 3302 / 2570 | 3181 / 3291 / 2563 | 3138 / 3307 / 2572 | 3223 / 3338 / 2602 |

- **B 最慢**（MTP 比 A 慢 16%）：每個草稿步驟都要跑輸出 head，而 Q8_0 head（1.3 GB）最昂貴；它較高的接受率無法彌補。
- **在 MTP 下 A 比 C 快 4.5%**，儘管沒有 MTP 時 C 快 1.9%：WHIRL 對 Q6_K head 有專用路徑（2-bit 草稿 head 是從它重新量化而來；驗證使用 Q6_K 專用的多列表）。Q4_K head 兩者都沒有。加上 n-gram 時兩者打平（157.2 對 157.5）。
- Swift 對 Qwen MXFP4 參考：decode 速度相同（Q8_0 MTP 層讓草擬稍微變貴；高出 5 個百分點的接受率彌補了這點）；prefill 慢 1–2%（Q8_0 embedding、alpha/beta、MTP 層）。
- **建議：A。** 在 WHIRL 中 MTP 最快，加上 n-gram 時與 C 打平，head 比 C 更精確，且 head 類型與參考相同，因此所有 Q6_K-head 路徑都適用。對 llama.cpp 使用者而言 C 是合理的選擇（在那裡稍快）；不建議 B。

llama.cpp b11214 也能載入這三個檔案（唯一的警告是已知的 `unknown type mxfp4`；在一般模式下 4 個 nextn tensor 會被回報為未使用，這是正常的），並能執行 `--spec-type draft-mtp`：一般模式 33.3 / 32.8 / 33.8 tok/s，MTP（n-max 3）58.1 / 55.9 / 60.7，接受率 73.6 / 72.2 / 75.0%，prefill 2k 約 1206–1215。在 llama.cpp 中，英文的 MTP 輸出與一般模式一致；中文輸出在中途分歧（它的批次數值計算不是逐位元相同）。

**思考長度**（官方取樣 temp 1.0 / top_p 0.95 / top_k 20 / min_p 0，開啟思考，上限 16,384 tokens，9 個提示詞 × 2 個 seed）：Swift A 129,299 個思考 token / 1,859 s / 5 次撞到上限；Qwen MXFP4 169,510 / 2,328 s / 8 次撞到上限。在四個中文寫程式提示詞上，Swift 的思考少了 44%（54,948 對 97,496 tokens），完成時間早了 38%（858 對 1,395 s）。我們檢查了一次撞到上限的 Swift 執行是否有迴圈（重複 32-gram 佔比前半為 0.5%、後半為 1.3%）：是真正的推敲，而非退化性的重複。

## <a id="ornith-mxfp4"></a>6. Ornith-1.5-35B-A3B MXFP4

由我們從官方 BF16 權重量化（尚未發佈）。WHIRL 以速度模式執行：prefill 時 MXFP4 專家搭配 fp8 activation，decode 時用整塊（whole-block）int8 kernel。

### 6.1 配方

1. **轉換**：llama.cpp 的 `convert_hf_to_gguf.py --outtype bf16`（保留 MTP 層：`nextn_predict_layers` 1，tensor 名稱與官方 Q4_K_M 相同）。
2. **量化**：llama.cpp b11214 的 `llama-quantize`，**不用 imatrix**（主線對 MXFP4 與 Q8_0 本來就不使用），搭配 tensor-type 檔：

   ```
   blk\.40\.ffn_gate_inp=f32
   blk\.40\.=q8_0
   ssm_alpha=q8_0
   ssm_beta=q8_0
   token_embd=q8_0
   attn_gate=mxfp4
   attn_qkv=mxfp4
   attn_q\.=mxfp4
   attn_k\.=mxfp4
   attn_v\.=mxfp4
   attn_output=mxfp4
   ffn_gate_exps=mxfp4
   ffn_up_exps=mxfp4
   ffn_down_exps=mxfp4
   ffn_gate_shexp=mxfp4
   ffn_up_shexp=mxfp4
   ffn_down_shexp=mxfp4
   ssm_out=mxfp4
   ```

   ```
   llama-quantize --tensor-type-file tt_orn.txt --output-tensor-type q6_k \
       ornith-1.5-35b-bf16.gguf Ornith-1.5-35B-A3B-MXFP4.gguf MXFP4_MOE 16
   ```

   結果：experts、shared expert 與所有 dense 矩陣為 MXFP4；`ssm_alpha` / `ssm_beta`、embedding 與整個 MTP 層（`blk.40`）為 Q8_0；router（`ffn_gate_inp`，含 `blk.40`）、`ffn_gate_inp_shexp`、norm 與小的 SSM tensor 為 F32；輸出 head 為 Q6_K。19,819,767,136 位元組（4.46 bpw，載入 18.45 GiB）。
3. **視覺投影器**：`convert_hf_to_gguf.py --mmproj --outtype f16`（899,283,296 位元組）。

### 6.2 結果（R9700）

兩個引擎用同一個 GGUF；WHIRL 為 greedy，MTP 輸出與 plain greedy 輸出相同（也與研究原型逐 token 相同）；llama.cpp b11214 ROCm、`-fa on`，取 `-ub 512 / 2048` 較快者（每個 prefill 長度都是 2048 較快）。

| | WHIRL MXFP4 | WHIRL Q4_K_M | llama.cpp MXFP4 | llama.cpp Q4_K_M |
|---|---|---|---|---|
| Prefill 88 / 2k / 8k / 32k tokens（tok/s） | 2,542 / 11,729 / 10,853 / 7,978 | 1,338 / 5,862 / 5,992 / 4,980 | 1,820 / 4,820 / 4,760 / 3,936 | 1,725 / 4,383 / 4,328 / 3,639 |
| Decode，MTP + n-gram，7 個中英程式提示（中位數，800 tokens） | 253 tok/s | 224 tok/s | — | — |
| Decode，MTP + n-gram，編輯類提示（bugfix / refactor-js） | 272 / 253 tok/s | 224 / 241 tok/s | — | — |
| Decode，無 MTP（同提示；llama.cpp 為 `tg256`） | 194 tok/s | 180 tok/s | 120 tok/s | 103 tok/s |
| Server，單一請求，MTP + n-gram（llama.cpp：無 MTP） | 246 tok/s | 203 tok/s | 113 tok/s | 96 tok/s |
| Server，4 個並發請求，總吞吐 | 435 tok/s（MTP + n-gram）、355（無 MTP） | 347、321 | 227（無 MTP）、138（MTP） | 219、84 |

Server 列兩個引擎用同一個 client（7 個 bench 提示、800 tokens、thinking 關、top_k 1）；llama.cpp 的 MTP（`--spec-type draft-mtp`，3 drafts）在這個模型上比它自己的 plain decode 慢，所以公平比較看它的 plain 數字。

fp8 專家 prefill（預設）在 2k 比 f16-activation 專家 GEMM（`WHIRL_MOE_FP8=0`：1,920 / 8,591 / 8,532 / 6,636 tok/s）快 37%。第一版反而比 f16 路徑慢（2k 5,249 tok/s），因為 down projection 的 f32 輸出每個 lane 寫 8 個分散的 4-byte；每個 lane 持有同一個 token 的 8 個連續 row，改成兩個 16-byte store 寫出相同的值（位元相同）。2k-token prefill profile 的 GEMM 時間從 187 ms（f16 專家）降到 122 ms（fp8 專家）。

## <a id="rules"></a>7. 製作在 WHIRL 上跑得好的 GGUF 檢查清單

1. 保留 `general.architecture` `qwen35` / `qwen35moe`，並**保留 MTP 層**（`nextn`）——
   沒有它，WHIRL 只能以無 MTP 的上限 decode（27B 4-bit 模型約 38 tok/s）。
2. 不要對 `ssm_alpha` / `ssm_beta` 使用 F32；使用 Q8_0（或 MXFP4）。
3. 優先選擇 **Q6_K 輸出 head**：它供給 2-bit 草稿 head，且有自己的多列調校。
   避免 Q8_0 head（每個草稿步驟都要為它付出代價）。
4. token_embd 和 MTP 層使用 Q8_0 沒問題（速度代價小，接受率略高）。
5. 只使用支援的類型（§2）；其他任何類型在載入時都會被拒絕。
6. 用 MTP 精確性測試檢查：純 greedy、MTP、MTP + n-gram 必須產生完全相同的文字。

## <a id="mmproj"></a>8. 視覺 projector（mmproj）

| 檔案 | 類型 | 大小 |
|---|---|---|
| `mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf` | F16，334 個 tensor | 927.6 MB |
| Ornith-1.5 mmproj | F16 | 899,283,296 位元組 |
| Qwen3.8 mmproj（參考） | F16 | 已在我們的 GGUF 一致性測試中解析 |
| Ornith mmproj（參考） | BF16 | 已支援（BF16 載入路徑） |

Swift mmproj 已用 llama.cpp 的 `mtmd` 驗證（影像編碼正確）。WHIRL 自己的圖片輸入見 [vision.md](vision.md)。
