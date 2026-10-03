[English](../en/kernels.md) | **繁體中文**

# Kernels

> **狀態。** 以下每一個 kernel 家族都已在 `kernels/*.hip` 中實作（gfx1201，R9700）；gfx1151（8060S）的 kernel
> 在規劃中。除非標註 8060S（gfx1151；這些數字來自研究建置），數字皆為 R9700（gfx1201）。「逐位元相同」一律指輸出
> 位元完全相同，並由常設檢查驗證，而不是「在容許誤差內」。

**對誰有幫助：** 為 RDNA 3.5/4 撰寫 HIP kernel 的人（WMMA 佈局、int8 內積、暫存器壓力）、在 AMD 上處理量化
GEMV/GEMM 的 llama.cpp/ggml 貢獻者，以及任何需要讓批次與非批次程式路徑產生相同位元的人。

## <a id="ceilings"></a>0. 我們拿來比較的上限

我們從不對著規格表做最佳化。以下每個上限都是在這張卡上用探測 kernel 實測的，本文件中的每個結果也都以它為基準來引用。

| 上限（R9700，gfx1201） | 實測 | 方法 |
|---|---|---|
| WMMA f16 → f32（`v_wmma_f32_16x16x16_f16`） | 180–184 TFLOPS | 只用暫存器的迴圈 |
| WMMA f16 → f16 累加 | ~177 TFLOPS | 未使用：幾乎沒有更快，精度差很多 |
| WMMA iu8（`v_wmma_i32_16x16x16_iu8`） | 343–374 TOPS | 只用暫存器的迴圈 |
| iu8 WMMA + 每個元素一個 `v_mad_i32_i24` | ~205 TOPS | 精確 Q4_K int8 收尾（epilogue）的上限（§5.5） |
| fp8 WMMA | 冷時脈下 ~344 TFLOPS；在 prefill 期間觀察到的 ~2.4 GHz 下持續 ~260–285 | 取自 r9700-stack 公開的微基準測試筆記 |
| 串流讀取，每個 wave 一列的 GEMV 模式，16 位元組載入 | 604–626 GB/s（規格 640） | 純讀取 kernel，> 512 MB 輪替以避開 64 MB Infinity Cache |

| 上限（8060S，gfx1151） | 實測 |
|---|---|
| WMMA f16 | 43 TFLOPS |
| WMMA int8 | 45 TOPS（與 f16 相同） |
| `v_dot4` int8 | 2.7 T lane-op/s（R9700：9.4） |
| 串流讀取 | `hipMalloc` 238 GB/s，pinned 232 GB/s |
| 啟動開銷 | 1.3–3.3 µs（R9700 2.2–2.5） |

形塑了每個 kernel 的硬體事實：wave 是 32 個 lane（Vulkan 回報 64——那是驅動程式的另一種分組）；gfx12 WMMA
處理 16×16×16 的 tile，A/B fragment（片段）放在暫存器中；LDS（共享記憶體）每個 workgroup 有 64 KB，但在我們的一個
kernel 中使用超過 ~41 KB 就讓佔用率（occupancy）從 16 降到 6 waves/SIMD；**暫存器才是真正的限制**——一旦 kernel
超出它的 VGPR 預算，編譯器就會溢出到 scratch 記憶體，效能隨之崩潰。

## <a id="gemv"></a>1. Decode GEMV：int8 activation、`v_dot4`、`v_perm` 查找表

Decode（逐 token 生成）是純粹的頻寬問題：每個 token 都要把所有權重串流一次（UD-Q4_K_M 檔案為 14.33 GB，其中混用了九種格式）。設計如下：

- **Activation（啟動值）量化為 int8**，以 32 個為一組，每組一個 scale，由產生它們的 kernel 完成（RMSNorm + 量化、
  SiLU·mul + 量化、attention combine + 量化都是融合的）。
- **權重在暫存器中解碼為 int8**，由各格式的解碼器（`QD<T>::dot`）負責，內積使用 `v_dot4`
  （`__builtin_amdgcn_sudot4`，有號 × 無號 int8 四路內積）。
- **寬單元。** Q4_K、Q5_K 與 IQ4_XS 以 64 個權重為一個單元處理（兩次 16 位元組載入、一次 header 解碼），使每個權重的浮點工作量降到
  16 值設計的四分之一。
- **透過 `v_perm_b32` 實作非線性 4-bit 表。** IQ4_NL/IQ4_XS（以及後來的 MXFP4）把 4-bit 代碼經由 16 項的表映射。兩次
  `v_perm_b32` 位元組選取加上依 bit 3 的一次混合（blend），完全在暫存器中完成查找，不存取記憶體。
- **小矩陣融合。** DeltaNet 的 beta/alpha 投影非常小（48 × 5120，在 unsloth 檔案中為 Q8_0）：單獨執行只能達到
  101 GB/s，因為啟動成本佔主導，所以把它們與 causal conv + L2 norm 融合成一個 grid。

第一版完整設計的結果：matmul 每 token 25.3 ms（566 GB/s，約規格的 88%），完整一步 28.8 ms = 34.7 tok/s。
經過下面後續的工作後，一個 decode 步驟為 **27.74 ms（Q4_K_M）/ 25.15 ms（MXFP4）**。

### 1.1 各格式距離上限多近

每個 1-token GEMV 都與使用相同存取模式的純讀取 kernel 比較（真實權重，> 512 MB 輪替）：

| 型別 / 形狀 | 正式版 1-token | 純讀取上限 | 判定 |
|---|---|---|---|
| q4_k 17408×5120 | 587–606 GB/s | 604–611 | 達頻寬（0–3%） |
| q5_k 10240×5120 | 584–591 | 587–599 | 達頻寬 |
| iq4_xs 5120×17408 | 573–593 | 598–612 | 達頻寬（2–4%） |
| mxfp4 up / down / qkv | 589–600 / 584–593 / 567–575 | 603–611 / 605 / 587–590 | 達頻寬（2–3.5%） |
| q6_k 5120×6144 / output head 248320×5120 | 576 / 626 | 578 / 626 | = 上限 |
| iq4_nl 5120×17408 | 591 | 601 | 達頻寬 |
| q8_0 17408×5120 / 5120×17408 | 599 / 594 | 622 / 615 | 達頻寬 |
| **q3_k** 17408×5120 / 5120×17408 | **491 / 466** | 593 / 598 | 受 ALU/延遲限制 |
| **iq3_s** 17408×5120 / 5120×17408 | **502 / 482** | 595 / 597 | 受 ALU/延遲限制 |

對已達頻寬的格式**沒有**帶來收益的做法（全部在 ±2% 內，全部逐位元相同）：activation 放進 LDS（每個 block 或常駐）、
常駐 grid（256/512/1024 個 block）、每個 wave 2 或 4 列共用 activation、計算前每個 lane 先載入 2–4 個單元、non-temporal
載入（較慢）、block 大小 128/512/1024。有公開報告指出在這張卡上用 `v_perm` 解包可 +15%，但那是相對於 ggml 基準；WHIRL
本來就使用 `v_perm` 表。

**Q3_K / IQ3_S 的修正。** ISA 顯示 Q3_K 的 scale 查找
`kk < 8 ? sb8[kk] & 0xF : sb8[kk-8] >> 4` 被編譯成 wave 分歧的分支：載入 `d`、分支、載入 scale 位元組、
`s_wait_loadcnt 0`，然後才發出權重載入——每個 super-block 有兩段序列化的記憶體延遲，而每個 wave 步驟只處理一個
110 位元組的 super-block（記憶體層級平行度太低）。修正保持 lane l 擁有每個 super-block 的單元 l，浮點運算式也完全相同，但以無分支方式讀取
scale nibble（固定位元組 `96 + (kk & 7)`，位移 0 或 4），在計算前先發出 4 個 super-block 的所有載入，並使用 2 位元組對齊的
u16/u32 載入。逐位元相同；q3_k 491 → 588–590、466 → 581–583 GB/s，iq3_s 502 → 583–586、
482 → 577–579 GB/s（+16…+25%，≈ 上限的 98%）。

### 1.2 Decode 中的小 kernel 融合

| 融合 | 取代的啟動次數 |
|---|---|
| `attn_prep`：q-norm、k-norm、q 與 k 的 RoPE、KV 快取寫入 | 每個 attention 層 5 → 1 |
| `attn_combine_q8`：split-K combine 直接寫出 o-projection GEMV 的 int8 輸入 | 省掉一次量化啟動 |
| `gdn_abconv`：beta/alpha 投影與 conv + L2 norm 在同一個 grid（閒置執行緒加 0，所以樹狀順序不變） | 每個 DeltaNet 層 2 → 1 |
| recurrent step + gated RMSNorm + int8 量化 | 3 → 1 |

每個 decode 步驟的啟動次數從 868 → 740，decode 步驟 28.37 → 28.17 ms（−0.7%）。剩下的啟動間空隙約每步 1 ms；之後我們融合了相同輸入的
GEMV（§4）。

## <a id="multirow"></a>2. 與 1 列 GEMV 逐位元相同的多列 GEMV

推測解碼以 k+1 列跑過模型來驗證 k 個草稿。如果驗證 kernel 計算某一列時的浮點順序與 1-token decode kernel 不同，logits
會在最後幾個位元不同，接近平手的 token 會翻轉，MTP 輸出就不再與一般 greedy 輸出一致。我們要求位元相等，所以算術是共用的：

- **標準順序。** 每個單元 u 依 u 遞增累加到 slice `u % 32`，使用相同的浮點運算式；32 個 slice 以固定的 xor 樹（16、8、4、2、1）合併。
  1-token kernel 使用相同的程式路徑（`QDN<T>::dot<1>`），所以「1 列」只是 n = 1 的情形。
- 2–16 列的**多 token dp4 kernel** 由巨集產生；從 6 列起，每個 wave 處理 2 列並共用已載入的 activation。Matmul 時間：T=1
  25.35 → 25.0 ms、T=8 42 → 36、T=11 64 → 41、T=16 100 → 53 ms（在下面的 WMMA 版本之前）。
- **常設檢查 `checkGemvBitwise`：** 每種型別 × 列數 2..16 × 每個 kernel 變體（以及 output head）都必須與 1-token 結果逐位元相等。

在這些批次大小下 WMMA GEMM 太慢了（16 列一步約 158 ms），所以 ≤ 16 列一律使用 GEMV 式的 kernel。

## <a id="wmma-gemv"></a>3. int8 WMMA 中批次 GEMV（3–16 列，gfx1201）

`v_wmma_i32_16x16x16_iu8`，A = activation（16 個 token 列），B = 解碼後的權重（16 個權重列）。WMMA 內部的整數內積是精確的，所以只有浮點部分需要重現標準的
1-token 順序：

- 每個單元（Q4_K/Q5_K/IQ4_XS 為 64 個權重；Q6_K/Q3_K 為各自型別的單元）依 u 遞增累加到 slice `u % 32`，使用 1-token
  的浮點運算式；wave w 擁有 slice w、w+8、w+16、w+24，所以 xor-16 與 xor-8 在暫存器中完成，xor-4/2/1 透過 LDS——同一棵樹；
- Q4_K/Q5_K 的 `dmin·m` 項需要每個 sub-block 的 Σx：多一次 B = 全 1 的 WMMA，直接把它放進 D 佈局；
- Q6_K 使用有號（q − 32）權重，對其 8 值 scale 群組用半遮罩 WMMA；Q3_K 與 IQ3_S 把 8 值單元配對；IQ4_NL 每個單元一次 WMMA。

讓它變快的暫存器修正：用 `readfirstlane` 讓 wave 索引成為 uniform（scale 分支變成純量）、一次 16 位元組 header 載入搭配無分支 scale 解碼，以及在每個單元的累加器上用
`asm volatile` 釘住——沒有這些的話，編譯器會交錯四個單元的 WMMA，用掉 256 個 VGPR 外加數百位元組的溢出。

結果：matmul T=8 36.5 → 30.5 ms、T=16 53.2 → 32.5 ms（T=1 不變）；驗證 n=16 68.3 → 42.8 ms；單一使用者 MTP +6–8%；4–16
個並行使用者總吞吐量 +16–25%。

### 3.1 第二代：管線化的 `gw2` / `gw8`

一個只重現記憶體模式（相同的 16 列 × 32 B 權重 + header + activation）的探測達到 608–616 GB/s，所以存取模式不是瓶頸。正式版的
ISA 每個單元都有一個 `s_wait_loadcnt 0`：一個 `j < 2` 的 scale 分支把 header 載入切成條件式載入，使 activation → header →
權重序列化。改寫保留完全相同的 lane/單元/slot 對應、浮點運算式與歸約（與 1-token 逐位元相同），並且：

- 以 u = w + 8k 迭代，讓 scale 類別 `j = u & 3` 在每個 wave 內固定 → 迴圈依類別特化，沒有分支；16 位元組 header 載入變成無條件；
- 在計算目前單元前，先發出下一個單元（或下 3 個單元）的所有載入（2 或 4 的環形緩衝）；
- **主迴圈中沒有 `break`**（尾端另外處理）：有 `break` 時，編譯器每一步都把等待計數器歸零，預取完全沒作用；
- activation 以每個 lane 一個 dword 載入（token = lane % 16）再加 shuffle，取代每個單元八次 64-bit 載入；
- 以 `(b << 8) - b` 在 asm 屏障後混合 LUT nibble，取代 `v_mul_lo_u32`（否則編譯器會把它折回乘法）。

變體：v5 = 1 tile / 深度 1、v6 = 1 tile / 深度 3、v7 = 2 tiles / 深度 1、v8 = 2 tiles / 深度 2。`gw8` 處理 8 值單元的型別（Q3_K、IQ3_S），每個
lane 解碼自己的四個單元 scale，IQ3_S grid 表放在 LDS。

預設選擇（列數 → 變體），依型別實測：

| 型別 | 選擇 |
|---|---|
| Q4_K | 3 列用 v1，4–7 用 v6，8 起用 v5 |
| Q5_K | 3–5 用 v6，6 起用 v5 |
| IQ4_XS | 3–5 用 v1，6–11 用 v6，12 起用 v8 |
| MXFP4 | 3 用 v1，4–13 用 v6，14 起用 v8 |
| Q6_K（各層） | 6–7 用 v5，8 起用 v7 |
| Q6_K output head（248,320 列） | 6 起用 v2（有自己的表；管線化版本較差：2.16–2.30 vs 2.07 ms） |
| Q3_K | 4 列以下用新的 dp4，5 起用 v5 |
| IQ3_S | 12 列以下用 dp4，12 起用 v5 |
| IQ4_NL、Q8_0 | 不變（dp4 / 較早的 WMMA 變體） |

結果：驗證 n=16 41.61 → 39.62 ms（Q4_K_M），MXFP4 T=16 matmul 27.86 → 25.94 ms；四個並行使用者的穩態吞吐量 241.1 → 252.9 tok/s（Q4_K_M）與
276.7 → 289.8（MXFP4）；單一使用者程式設計基準測試 +3.8%、檔案編輯 +4.4%（Q4_K_M）。16 列相對 1 列的剩餘成本：q4_k 1.09×、q5_k 1.11×、iq4_xs
1.12×、mxfp4 1.16×、q6_k 各層 1.46×、head 1.25×、q3_k 1.9×、iq3_s 2.5×——其餘是每個（列、token、單元）的標準浮點運算式，逐位元相同的要求不允許省略。較差的做法：每個
block 3–4 個 tile（溢出，−7…−40%）。

### 3.2 gfx1151（RDNA 3.5）的差異

- 在 8060S 上 `v_dot4` 的速率只有 int8 WMMA 的一半，而且每個驗證列都要重讀 activation，所以 Q4_K、Q5_K、IQ4_XS 與 Q6_K 的
  2–16 列 kernel 使用 int8 WMMA；1-token dp4 kernel 共用相同的切片與加總順序（MTP 維持逐位元相同）。相對 1 列的驗證成本：n=4 1.36 →
  1.23×、n=8 1.91 → 1.50×。
- **gfx11 WMMA 需要兩個 half-wave 都有 B fragment。** 原本每個 half-wave 解碼整個 64 值單元（IQ4_XS 的 LUT 解碼很貴）。現在每一半只解碼自己的 32 值半邊，再用
  `v_permlanex16` 交換兩半。整數內積不變 → 逐位元相同；IQ4_XS T=3 27.68 → 25.37 ms（−8…−9%）。
- **gfx1151 decode MoE kernel** 原本使用 16 值單元，每個 lane 8 位元組載入（166–180 GB/s）；改用 64 值寬單元後，在 MoE 模型上 decode
  −2.3%、9 列驗證 −14%。

## <a id="grouped"></a>4. 相同輸入 GEMV 的群組啟動

啟動的頭尾成本約 2.8 µs。讀取相同輸入的矩陣——attention q/k/v、FFN gate/up、DeltaNet qkv/z、MoE 共用專家的 gate/up（以及 MTP
層中相同的區塊）——只要型別與欄數相同，就只啟動一次：

- 1-token 入口接收一個 `GvArgs` struct，最多三個段落 `(W, y, row_bytes, nrows)`；block 範圍 `[0,b1)`、`[b1,b2)`、`[b2,…)`
  對應到各段落，block 索引在每個段落內重新起算。每一列的 kernel 與浮點運算式不變 → 逐位元相同。
- 一個標記 kernel 告訴 host 該 code object 是否具備群組 ABI（沒有它的 gfx1151 object 會自動使用舊 ABI）。

**Codegen 陷阱。** 讓*每個* GEMV 入口都使用群組 ABI 改變了多 token 的程式碼生成：只要權重指標不是直接來自 kernel 參數（一個
`select`，或 `W + offset`），VGPR 就增加 30–90%，部分變體出現溢出（q6_k v7 在 T=8：0.92 → 1.97 ms）。`readfirstlane`、
`__builtin_assume` 與整數 offset 都沒幫助；三個 inline 分支更糟。解法：單一矩陣保留原本的入口；群組使用獨立的「雙胞胎」入口，共用的實作是
`__forceinline__`（有兩個呼叫者時它就不再 inline，達到 248 VGPR + scratch）。雙胞胎用到 11 列（MXFP4 為 13），超過就變差。

| | Q4_K_M 之前 | 之後 | MXFP4 之前 | 之後 |
|---|---|---|---|---|
| 每 token 啟動次數 | 740 | 674 | 740 | 596 |
| Decode 步驟（無 MTP） | 28.09 ms | 27.74（−1.2%） | 25.66 | 25.15（−2.0%） |
| 驗證 n=3 | 30.99 | 30.22（−2.5%） | 27.70 | 27.26（−1.6%） |
| 驗證 n=5 | 32.38 | 31.71（−2.1%） | 29.00 | 28.44（−1.9%） |

代價：kernel 建置時間 5 → 13 分鐘。混合型別的群組（在 unsloth 檔案中 q/k/v 為 q4_k/q5_k/q5_k）仍然分開啟動。

## <a id="prefill-gemm"></a>5. Prefill GEMM

### 5.1 基礎設計

- 一個樣板化的 WMMA GEMM `gemm3_impl<T, BM, BN, BK, WAVES_M, MODE, NTH, TOK_X>`。
- 快速 f16 解碼器 `dec_group_h<T>` 把量化權重轉成 WMMA fragment。
- 兩條路徑，依（型別、形狀）選擇：**融合**（邊解碼邊計算）與**先反量化為 f16，再做純 f16 GEMM**（適合大批次，反量化成本被攤提）。
- **`TOK_X` grid 順序：** grid 為 `{ceil(n/BN), ceil(rows/BM), 1}`——token 在 x——所以連續的 workgroup 從 L2 而非主記憶體重用同一個權重
  tile。這救回了第一版快速反量化，並讓效能更進一步。
- 對每個 tile 設定依（型別、形狀、批次 bucket）**自動調校**，結果快取在磁碟上。

這個設計中的坑（細節見 [pitfalls.md](pitfalls.md#kern)）：

- 快速反量化提高了暫存器壓力；最常用的 tile 每個 lane 溢出 552 位元組，prefill 嚴重退步。一個無溢出搜尋編譯了每個候選、讀取資源用量，只保留
  24 個沒有溢出的設定。
- 每次重複後都同步的自動調校量到的是啟動開銷，並偏好融合路徑。改成對一批重複計時、最後只同步一次就修好了。（冷快取調校沒有幫助。）32-bit
  的候選遮罩在 48 個候選時溢位（未定義的位移）→ 改為 64-bit。

第一輪的結果：模型所有 matmul 在 T=4096 跑到 107–108 TFLOPS（約 180 的 59%），最好的純 f16 GEMM 為 119 TFLOPS（約 66%）；prefill
在 1.1k / 14k / 24k token 為 1,296 / 1,430 / 1,355 tok/s。

### 5.2 批次 bucket

| 變更 | 原因 | 效果 |
|---|---|---|
| 2 個 bucket（在 512 與 4096 調校） | 第一版設計 | 85 token 的提示詞被補齊到 256 token 的 tile |
| + bucket n ≤ 128，在 128 調校 | 85 token 的 FFN-up 用 512 調校的 tile 要 0.791 ms，用既有的 64×128 tile 只要 0.271 ms | 85–89 token 的 prefill 184–188 → 132–133 ms（−29%）；先試過 n ≤ 256，讓 152/231 token 的提示詞慢了 3–4% |
| 11 個 bucket：≤32、48、64、96、128、192、256、384、≤768（在 512 調校）、769–1024（在 1024 調校）、> 1024（在 4096 調校） | server 以 1024 列為一個 chunk 執行，落在 512 調校的 bucket 中 | 只改調校檔案，server 8k 冷 prefill 就從 1318 → 1441 tok/s（+9.3%） |

Tile 的選擇只改變 M/N 的分塊；K 方向的 WMMA 順序是固定的，所以每種 tile 設定都產生相同的位元。一個常設的不變性檢查會執行每個 GEMM 選項（目前集合中有
61 個）、每個 MoE token tile 以及 40 列子集，並要求位元相等——這就是讓批次多請求 prefill 等同單獨 prefill 的關鍵。

### 5.3 `gemmh_f16`：f16 權重的 fragment 順序 LDS

兩個運算元都經過 LDS，但以 WMMA fragment 順序儲存，所以每次 fragment 讀取都是無衝突的 `ds_load_b128`。設定：128×256 tile、BK 32、8 個
wave（TM 2 × TN 8）、單緩衝、以 4 個 token block 為一組做光柵化（rasterization）、16 位元組的收尾儲存。比先前的 f16 kernel 快
+19.6…+30.2%（約 126–130 TFLOPS），逐位元相同，其他地方的資料佈局都不需改變。用於 n ≥ 512。被否決的變體：activation 直接以 tiled
形式讀取（+25–31%，但每個 f16 產生者都要改——相對上面的方案不值得）、權重直接以 tiled 形式讀取（+22–28%）、兩者都直接讀（+12–22%）、雙緩衝 /
512 執行緒 / BK 64（較差）、不做光柵化分組（−9…−49%），以及**把反量化融合進 GEMM 的權重讀取**（逐位元相同但 −10…−13%：Q4_K/Q5_K/Q6_K
解碼的 VALU 工作落在 GEMM 的關鍵路徑上）。

Q4_K_M CLI prefill 在 2k–32k 提升 +9.6…+14.0%，在 128k 提升 +8.1…+8.3%。

### 5.4 小批次：`gemms` / `gemmsd`

對一個 35 token 的提示詞，prefill 有 92% 是 matmul，權重串流只有 120 GB/s：大部分 WMMA 工作都是補齊。兩個新的 tile 家族成為**自動調校選項**（調校器依型別、形狀與
bucket 挑選；沒有任何寫死）：

- `gemms`（BN 16/32/48/64）：每個 256-k 階段把**原始量化位元組**以 16 位元組對齊複製進 LDS，並保留每個 super-block 在全域記憶體中的 mod-16
  對齊（使 `dec_group_h` 讀到相同的位元組），activation slab 也放進 LDS，下一個階段預取到暫存器。Token tile 16–64，不補齊到 128/256。
- `gemmsd`（64×64、64×96、64×128）：同樣的原始階段，但整個 block 協同把一個 128-k 半階段一次解碼成 f16 A tile，再從 LDS 執行 WMMA（用於 n
  ≈ 96–512）。

數值與 `gemm3` 相同（A fragment 來自 `dec_group_h`，B 來自相同的 f16 X，K 以 16 為步長遞增）。探測：n = 17–64 時比既有最佳設定快
+57…+175%。消融（ablation）：光是載入 + LDS 儲存 + 屏障就花 0.105 ms（~440 GB/s），而 0.08 ms 的計算幾乎沒有重疊——單階段預取、每個 WGP
兩個 block（activation slab 佔滿 LDS）是目前的上限，而 gfx12 沒有 16 位元組的非同步 global→LDS 載入。較差的做法：更深的暫存器預取（溢出）、activation
直接從 global 讀取（L2 流量太大）、半個 activation slab、不經 LDS 解碼權重（−20…−66%）。

CLI prefill（Q4_K_M）：88 token 692.7–697.5 → 848.5–867.3 tok/s（+23.4%，TTFT 127 → 103 ms），209 token +9.5%，≥ 493 token 不變。

### <a id="int8-prefill"></a>5.5 負面結果：gfx1201 上的 int8（W8A8 / 精確 W4A8）prefill

紙面上 int8 WMMA 是 f16 的 2 倍。只用暫存器時，iu8 WMMA 加上每 32-k 一次的浮點收尾（轉換 + scale + fma）跑到 356 TOPS——gfx12 上
VALU 與 WMMA 會重疊。但在由記憶體餵資料的 GEMM 中，它從未贏過 f16：

| 嘗試（FFN-up 17408×5120，4096 token） | 最佳 |
|---|---|
| 目前的 f16 路徑（反量化 + f16 GEMM） | 108–116 TFLOPS（6.3–6.8 ms） |
| W8A8，精確的每 32-k 權重 scale，三代共 15 個變體 | 96.5 TOPS（7.57 ms） |
| W8A8，有損的每 128-k 權重 scale | 108.8 TOPS（6.71 ms） |
| 精確 W4A8（Q4_K/Q5_K/IQ4_XS 權重不動，int8 activation）：v1 從 LDS 原始 block | 52.5 TOPS |
| … HIP `int4` struct 陣列改用 ext_vector 型別（預取不再落到 scratch） | 95.0 |
| … header 在載入時解碼 + 透過 inline asm 使用 `v_mad_i32_i24` | 118.2 |
| … 權重在 LDS 中依 activation 排列解包為 int8，128×128，512 執行緒 | **126.1（Q4_K），117–120（IQ4_XS、Q5_K）** |
| … 再加上 64-k chunk 與雙緩衝 LDS | 110.2 |

把 activation 量化也算進去後，最好的精確 W4A8 在 FFN-up 上只比 f16 快 +6.6…+10.4%，在 down-projection 上則是 −4…−7%。上限解釋了原因：兩個串接的
iu8 WMMA 加上每個元素一個 `v_mad_i32_i24`（精確 Q4_K 無法避免的 sub-block 整數 scale）最高只到 ~205 TOPS；gfx1201 每個 WMMA 只能免費重疊約 4–5
個 VALU 指令（每 WMMA 4 個 VALU：217 TOPS，8 個：102 TOPS）。解包、定址與 LDS 吃掉其餘部分，所以實際落在 120–130。精度也較差：每 256 一組的 int8
activation 造成 1.32e-2 的 GEMM 相對誤差，f16 路徑為 3.8e-4（35×）；每 32 一組的 activation 則需要每個 sub-block 的浮點收尾（74–84 TOPS）。獨立佐證：llama.cpp
的 Vulkan int8 coopmat 路徑（PR #27952）在這張同樣的卡上對 q4_K/q5_K/q4_1/q5_1/nvfp4 停用 int8，因為它比 fp16 慢。在 gfx1151 上 int8 WMMA 與 f16
同速，而且 gfx11 WMMA 與 VALU 共用，所以我們沒有嘗試。

留作教訓：*沒有* sub-block scale 的 int8 GEMM 在 RDNA 4 上可達峰值的 ~55%（一個公開的 W8A8 kernel：在 9070 XT 上 ~210 TOPS）；殺死它的是 K-quant 的
sub-block 收尾。MXFP4 完全避開了這個問題（§6）。

## <a id="mxfp4-gemm"></a>6. MXFP4 × fp8 WMMA prefill（速度模式）

**構想（歸功於 [r9700-stack](https://github.com/bkvargyas/r9700-stack)，Apache-2.0）：** 一個 MXFP4 block 是 32 個 e2m1 值加上一個
E8M0 指數——純粹的 2 的冪次。2 的冪次 scale 可以精確地折進 fp8（e4m3）的指數：令 `ref` = 該列最大的 block 指數，只要 `ref − e ≤ 8`（在 e4m3 的次正規範圍內），
`w8 = e2m1 × 2^-(ref − e)` 就是精確的。在 `v_wmma_f32_16x16x16_fp8_fp8` 之後只剩一個收尾：
`y = acc × sx[t] × 2^(ref − 127)`，每個 token 一個 fp8 activation scale（`amax / 448`）。權重解包在寫入 LDS 時完成（每 32 個值：一次 16 位元組載入、兩次
`v_perm` 查表再加上符號位元），平均每個 WMMA 不到一個 VALU 指令——沒有 sub-block 整數收尾，所以 §5.5 的上限都不適用。在抽樣的 7 個張量中，每列的
「最大指數 − block 指數」處處 ≤ 8；在整個 Qwen3.8 MXFP4 模型中只有 56 個非零 block 超過 8 而被捨入（載入器會計數並記錄）。

第一次探測（4096 token，真實權重）：

| 形狀 | f16 路徑 | fp8 GEMM | 含 activation 量化，相對 f16 |
|---|---|---|---|
| FFN up 17408×5120 | 6.40–6.68 ms | 3.67–3.77 ms（194–199 TOPS） | +65–70% |
| down 5120×17408 | 7.01–7.11 | 3.47–3.57（204–210） | +69–71% |
| qkv 10240×5120 | 3.93–4.03 | 2.17–2.19（196–198） | +59–61% |

準確度：相對於使用相同 fp8 activation 的 f64 參考為 4e-8（kernel 是精確的）；相對於 f32 activation 為 2.57e-2，而 f16 路徑為 2.0e-4。這個誤差來自
e4m3 本身的 3-bit 尾數——不管選每 token 或每 block 的 scale 都改變不了。這就是為什麼 MXFP4 是*速度*模式（[quantization.md](quantization.md)）。

### 6.1 `gemm8t`：fragment 分塊的 activation

fp8 activation 的產生者（activation 量化、RMSNorm、SiLU·mul、gated norm）直接以 WMMA fragment 順序寫出：一個 16 token × 16-k 的
fragment 是 256 個連續位元組，lane l 持有 token `l % 16`、k `(l / 16) × 8 … +8`——正好是 fp8 WMMA 的 B 運算元。每個 wave 用一次合併的
`global_load_b64` 把它的 B fragment 直接載入 WMMA 暫存器；activation 從不經過 LDS。權重（經指數折疊解包）以 fragment 順序寫入 LDS（無衝突的
`ds_load_b64`），以 64-k slab 為單位，雙緩衝，每個 slab 一個屏障。每個 block 的 LDS 從 55 KB 降到 16 KB，所以每個 WGP 可容納多個 block。每個累加器的 WMMA
順序與收尾 scale（`st × 2^(ref−127)` == `ldexpf(st, ref−127)`，精確）不變，所以輸出與先前的 fp8 GEMM 逐位元相同。（fragment 分塊 activation
技術以及只針對 LDS 的屏障序列 `s_wait_dscnt 0; s_barrier_signal -1;
s_barrier_wait -1` 也取自 r9700-stack；kernel 結構與權重佈局是 WHIRL 自己的。）

調校掃描（所有變體都與基礎版逐位元組比較）：最佳為 128 列 × 256 token、8 個 wave 排成 4×2（各 TM 2 × TN 8）、BK 64。以每組 4 個 token block
光柵化（接著是列 block）讓 qkv +3–4%、attn_q +8–9%（attn_q 的 48 KB 輸出步距有衝突）。改寫收尾（把每列的 `2^(ref−127)` 提出、每個 tile 兩次 16 位元組儲存）又多了
7–19 個百分點。較差的做法：BK 32/128、256×256、256×128、64×256、512/128 執行緒、`s_setprio`、在計算期間預取下一個 slab 的 B（碰到 256 VGPR，−40%）、pair
佈局的 16 位元組 B 載入（±2%）。小批次：n ≤ 64 → 128×64、≤ 192 → 128×128、其餘 128×256（之後也加入 n ≤ 32 用 128×32、n ≤ 48 用 128×48）。更早嘗試過把權重直接從
global 餵進 A fragment（183–189 TOPS）或對舊 kernel 做雙緩衝（158–173），都比單緩衝 + 暫存器預取慢。

各形狀相對先前 fp8 GEMM 的提升：up +42.7%、down +27.9%、qkv +27.1%、z +32.9%、ssm_out +29.6%、attn_q +73.7%、attn_k +11.6%——約
**238–248 TOPS**（原本 170–190）。MXFP4 CLI prefill 在 2k–8k 提升 +15.3…+16.4%。

### <a id="act-fusion"></a>6.2 Prefill 的 activation 融合

`rmsnorm_x8/x16`（RMSNorm 直接寫出下一個 GEMM 的輸入，fp8 + 每 token scale 或 f16）、`silu_mul_x8/x16`（SiLU·mul 寫出 down-projection
的輸入）、`gated_norm_x8/x16`（DeltaNet gated RMSNorm 寫出 out-projection 的輸入）以及 `gdn_conv_l2n`（causal conv + q/k L2 norm，每個 block
128 個 channel × 8 個 token，每個輸入列只讀一次）。只有在張量的每個消費者都使用相同 GEMM 輸入格式時才啟用；否則執行未融合的 kernel。全部逐位元相同——這需要
asm 屏障，因為編譯器把 `x*x` 收縮進第一個 butterfly 加法，並把一個乘法與 f16 轉換合併成 `v_fma_mix`，每個都造成 1-ulp 差異（§12）。8k 下的效果：misc
191 → 135 ms、norm 73 → 52、DeltaNet norm 53 → 36、DeltaNet prep 86 → 50。

僅限速度模式（非逐位元相同，MXFP4 預設）：FFN gate/up 與 DeltaNet qkv/z GEMM 為其逐元素消費者寫出 f16 而非 f32（+5.4…+5.8% 與 +0.6%）。

## <a id="flash"></a>7. Prefill flash attention

- **Sᵀ = K·Qᵀ 技巧。** 計算 S = Q·Kᵀ 會讓 softmax 機率 P 落在 D 佈局，但下一個 WMMA（O += P·V）需要 P 作為 A 運算元，這意味著要繞一趟 LDS。改算轉置
  Sᵀ = K·Qᵀ，則 Pᵀ 會留在暫存器中，佈局正好是 Oᵀ += Vᵀ·Pᵀ 的 B 運算元。P 不需要經過 LDS 來回。
- V 在暫存器中轉置（每個 block 128 個 query 列，256 執行緒）；KV 以 f16 儲存。
- **延遲 rescale。** 每處理完一個 key tile，累計輸出通常要乘上 `alpha = exp(m_old − m_new)`。當整個 wave 的
  `__builtin_amdgcn_ballot_w32(alpha != 1.f)` 為零時（累計最大值沒有改變——在長上下文深處幾乎總是如此），就跳過這 128 次乘法。乘以 1 是恆等運算，所以結果逐位元相同。每層
  −4.5%；CLI prefill 在 64k +1.1…+1.3%、在 128k +2.0%。
- **`attn_kx`。** 在 120k 上下文的分析：kernel 達到 61 TFLOPS；拿掉 K/V 載入會快 70%；PV 那一半最貴，因為 256 VGPR 仍溢出 40 個（o 累加器 128 + Qᵀ
  fragment 64），而且每個 WMMA 都從 LDS 讀取 512 位元組的 fragment。`attn_kx` 每 128 個 query 使用 16 個 wave；兩個相鄰的 wave 共用 16 個 query，各自用原本的
  WMMA 鏈為 16 個 key 計算 Sᵀ，透過 LDS 交換，兩者計算相同的 softmax，各保留 Oᵀ 的一半（64 VGPR，無溢出）；下一個 K/V tile 預取到暫存器。逐位元相同。120k：f16
  KV 218 → 185 ms（+17.8%），q8v KV 247 → 198 ms（+25.0%）；32k +11%；≤ 8k 不變。
- 超過 128k 上下文時，啟動依 head 範圍切分（結果不變；成本約 2%；≤ 128k 從不套用），這是在一次 256k prefill 期間出現無法解釋的 `HipFailed` 之後採取的預防措施
  （[pitfalls.md](pitfalls.md#hip-256k)）。
- 較差的做法：在非對角 tile 上跳過 causal mask（30k −4%，但 120k +1.3%）、只跳過 mask（+8%，排程變差）、依維度切分的 wave 對（1.5× WMMA，沒有收益）、只做預取（`attn_pf`，已被取代）、釘住
  PV 迴圈（scratch 252 → 28 B/lane，但 f16 慢 5%）。這個 kernel 對程式碼佈局非常敏感：每個變更都要量測。fp8 attention 沒有嘗試，因為它會改變長上下文的精度。

## <a id="decode-attn"></a>8. Decode attention：split-K 與 WMMA 群組驗證

- **Split-K flash decoding**，split 數量固定，每個 split 在自己範圍內的 64 位置 chunk 上執行 online softmax。Split 範圍在 GPU 上根據常駐裝置端的位置計算，所以
  decode 步驟仍可被 graph 捕捉。評分是每個 lane 一個位置：每個 wave 在一個 64 維切片上為 32 個位置評分，q 從 LDS 廣播，4 個切片的部分和透過 LDS 相加——沒有
  shuffle。24k 下的 split 數掃描：64 / 128 / 256 個 split = 35.5 / 33.9 / 34.7 ms/token。
- **`attn_wsplit1`（WMMA 群組驗證）。** 一個 block（KV head × split）以 WMMA 處理一*群* query 欄：一個 KV head 的 q heads（GQA）× 同一序列的連續驗證列。Sᵀ =
  K·Qᵀ 且 Oᵀ += Vᵀ·Pᵀ；K fragment 直接從快取讀取；V 由前 128 個執行緒轉置進 LDS；Qᵀ 放在 LDS 以避免溢出。群組只由同一序列、位置連續且 split 大小相同的列組成，≤ 16
  欄（27B：每個 KV head 6 個 q head → 每群 2 列）。每一欄的算術都等同單欄情形（`#pragma clang fp contract(off)` + 明確的 `__builtin_fmaf`），所以群組驗證的一列與單
  token decode 的一列逐位元相同，MTP 維持精確。常設檢查 `checkAttnGroups`（在位置 1000/1023 的隨機 Q/K/V）。
- 最大 split 數 128 → 64（每個 split 現在更有效率；位置 24000 的 combine：34.7 → 17.4 µs）。位置 24000 的 attention kernel：298 → ~170 µs。

| Decode tok/s（MTP / 一般） | 之前 | 之後 |
|---|---|---|
| 27B，12k 上下文 | 55.3 / 31.3 | 64.5 / 33.1 |
| 27B，24k | 45.5 / 29.8 | 61.5 / 31.9 |
| Ornith，12k | 165 / 137 | 190 / 158 |
| Ornith，24k | 137 / 129 | 185 / 150 |

- 較差的做法：64 欄變體（速度相同；在 gfx1151 上非逐位元相同）；一個對所有 query 只讀一次 KV 的版本（24k MTP 39.5 → 35.6 tok/s——瓶頸是歸約而不是 KV 讀取，而且它
  41 KB 的 LDS 降低了佔用率）。
- **`attn_wsplit2`（每群最多 32 欄）。** 每群 16 欄時，一個序列的驗證列兩列一組進 attention（27B：每個 KV head 6 個 GQA head），所以每個 split 的 K/V 範圍每 2 列就讀一次，
  而 block 的 4 個 wave 中有 3 個只幫忙搬 Vᵀ。`attn_wsplit2` 是同一個 template 的兩個欄群版本：wave 0 與 wave 1 在同一塊已搬好的 Vᵀ tile 上各算 16 欄（最多 5 列共用一次
  K/V 讀取；LDS 37 KB，低於約 41 KB 的佔用率斷崖）。每一欄走完全相同的程式路徑，所以 partials 與單獨執行逐位元相同（`checkAttnGroups` 與 kernel test 都拿 5 列群組和逐
  query launch 比對）。主機端以 32 欄上限分群，只有某群因此超過 2 列時才用 `attn_wsplit2`（單列 decode 仍用 `attn_wsplit1`；`WHIRL_ATTN_WIDE=0` 可關閉）。每個 attention
  層的 kernel 時間，q8v KV，R9700：

  | 上下文 | 列數 | `attn_wsplit1` | `attn_wsplit2` |
  |---|---|---|---|
  | 16k | 1 序列 × 5（一位使用者、4 個草稿） | 0.286 ms | 0.141 ms |
  | 16k | 1 × 9（8 個草稿） | 0.477 ms | 0.277 ms |
  | 16k | 4 × 4（四位使用者、各 3 個草稿） | 0.853 ms | 0.607 ms |
  | 32k | 4 × 4 | 1.784 ms | 1.162 ms |
  | 32k | 1 × 9 | 1.115 ms | 0.633 ms |
- **gfx1151 的限制：** 當 P = 0 的項乘上另一個 query 的真實 V 列時，gfx11 WMMA 並不精確，所以在 8060S 上每個群組只放一個 query（kernel 仍會執行，只是不共用 K/V）。
- 這項工作之後，一般 decode 時間每 1k token 上下文約增加 0.11 ms——正好是以 ~600 GB/s 每 1k token 多讀 64 MiB KV 的時間。Decode attention 已達頻寬；剩下的槓桿是減少 KV
  位元組，而反量化 K 的成本比它省下的位元組還多（[kv-and-caching.md](kv-and-caching.md#formats)）。

## <a id="deltanet"></a>9. Gated DeltaNet

### 9.1 分塊 prefill（f32，精確模式）

逐 token 的遞迴對 prefill 來說太慢了。分塊形式：64 token 的 chunk，每個 chunk 內使用 WY 表示法搭配三角求解，chunk 之間做遞迴掃描。第一個分塊版本比循序版本還慢；經過暫存器分塊、逐欄三角求解與管線化掃描後才變快。狀態欄
`S[128]` 一直溢出，直到把每一欄切到 4 個 lane 並用 DPP 歸約；另外一個以參考方式捕捉陣列的 lambda 迫使陣列進入 scratch（改用巨集取代）。Causal conv 有自己的平行 kernel；狀態另外處理。

Chunk 邊界從每個 prefill 段落的開頭起算，而 prefill 段落都從 1024 的倍數開始，所以單獨與批次 prefill 看到的 chunk 相同。

### 9.2 f16 WMMA chunk（速度模式預設）

- **Prep**（每個 chunk × v-head 一個 block）：WMMA 以 10 個下三角 16×16 tile 計算 K·Kᵀ 與 Q·Kᵀ；T = (I + B)⁻¹ 以 f32 前向代入求得（每欄 4 個 lane，DPP quad
  加總）；T 與 M 直接寫成 f16 WMMA A fragment（每個 chunk 與 head 10 KB，取代 80 KB 的 f32 W/U/M）。
- **Scan**（每個 head 一個 block，每個 wave 擁有 16 個狀態欄）：狀態 S（128×16）以 C fragment 形式留在暫存器中，而且**在 gfx12 上 C 佈局等於 B 佈局**，所以 K·S、Q·S、T·X、M·U
  與 Kᵀ·U 都不需要搬動 S 的資料。每個 wave 每個 chunk 116 次 WMMA。
- 探測（8192 token，一層）：13.4 ms → 1.9–2.0 ms（6.7×）。相對 f64 逐 token 遞迴的誤差：7e-7（f32 路徑）→ 4.2e-4（f16 輸入）。在 Q4_K_M 搭配 f16 prefill 時，最後
  token 的 KL 為 4.5e-8 … 1.5e-4（方法本身是精確的）；在 fp8 prefill 下升到 1.2e-4 … 7.0e-2，因為 fp8 activation 會放大任何擾動。僅為 MXFP4 的預設；一個需主動開啟的「relaxed」旗標可讓
  Q4_K_M 使用它。8k MXFP4 prefill 2212 → 2639 tok/s（+19.3%）。
- 坑：700 多個溢出的 VGPR 來自編譯器把 32 次 V 讀取與 32 次輸出寫入的 64-bit 位址當作迴圈不變量提到迴圈外。讓 lane offset 在每次迭代都不透明（`asm volatile("" : "+v"(off))`），並把位址拆成
  uniform 的列基底加上 32-bit lane offset，溢出從 700 降到 20。LDS 列有補齊（每列 17 / 9 個 chunk）以避免 bank 衝突。把下一個 chunk 預取到暫存器（S 64 + X/O 64 + V 32 + 預取 64
  VGPR）超過 256 而溢出——未採用。

### 9.3 Decode 步驟

`gdn_step_norm` 在一個 kernel 中執行遞迴步驟、gated RMSNorm 與 int8 量化。它的改寫把每一列的 q/k/v/decay/beta 預取進 LDS，相依鏈中每列只留一個屏障（雙緩衝歸約：第 t 列的輸出歸約延後到第
t+1 列的屏障之後），並在最後一次做 norm + 量化（每列一個 wave，加法順序與之前相同）。逐位元相同——但要加 asm 屏障，因為第一版讓編譯器把 `y*y` 收縮進 shuffle-add 的 fma。驗證 −0.2…−1%。

草稿被拒絕時遞迴狀態如何回滾（快照，然後重播）見 [speculative-decoding.md](speculative-decoding.md#replay)。

## <a id="moe"></a>10. 混合專家（MoE）（`qwen35moe`）

- **路由器（router）：** f32 softmax → 256 選 top-8 → 重新正規化使總和為 1（語意同 llama.cpp 的 `build_moe_ffn`）；共用專家以 `sigmoid(gate_inp_shexp · h)` 縮放。
- **Decode：** 被選中專家的 gate 與 up 在一個 int8 GEMV 中完成，只讀取 8 個被選中專家的列；down-projection、加權合併、共用專家與殘差在一個 kernel 中完成；沒有 host 同步。
- **Prefill：** （token, slot）配對在 GPU 上依專家分組，收集成 f16，經過群組 WMMA GEMM 再散回。專家平均少於 48 個 token 時 token tile 為 32，否則為 64。1.1k token 的 prefill
  ~3300 → ~5000 tok/s。
- 融合的 DeltaNet beta/alpha kernel 被一般化了，因為 Ornith 以 Q4_K 儲存 beta/alpha（27B：Q8_0）；沒有它，3 草稿驗證就無法執行。MTP 層的 FFN 也是 MoE。
- **路由的接近平手無所不在。** 在任兩條同樣有效的 prefill 路徑之間（WMMA 快速版、純量 naive attention、循序 DeltaNet），17–28% 的（token, 層）配對會選到不同的專家集合，從第 0–4
  層就開始；最後 token 的 KL 範圍為 1.2e-4 … 3.4e-2，而 45 個配對的 top-1 全部一致。稠密的 27B 在相同檢查下：KL ~1e-7。因此 MoE 的 KL 門檻必須依實測的路徑雜訊設定（MoE
  長上下文檢查用 1e-2，稠密模型用 1e-3）。

## <a id="resources"></a>11. 暫存器與資源紀律

上面每個 kernel 都經過同一個循環：建置時輸出資源用量、讀取 VGPR / scratch / 溢出、修正、重新量測。有效的技巧：

| 問題 | 技巧 |
|---|---|
| wave-uniform 的值卻按 lane 保存，純量分支變成向量分支 | 用 `__builtin_amdgcn_readfirstlane` 讓索引成為 uniform |
| 編譯器交錯獨立的累加鏈（4 個單元的 WMMA），暫存器爆掉 | 在每個 WMMA 之後用 `asm volatile("" : "+v"(acc))` 釘住各累加器 |
| 排程器預先載入全部 16 個 V fragment（256 VGPR + 13 個溢出，56 B scratch） | 在每個 dt WMMA 之後做同樣的釘住 → 253 VGPR，0 溢出（8060S 上 prefill +1.1%） |
| 迴圈不變的 64-bit 位址被提到迴圈外（700+ 溢出） | 每次迭代不透明的 lane offset；uniform 基底 + 32-bit offset |
| HIP `int4`（一個 struct）陣列被迫進入 scratch | `ext_vector_type` 向量 |
| 以參考方式捕捉陣列的 device lambda 迫使陣列進入 scratch | 巨集 |
| 有兩個呼叫者的實作不再 inline（248 VGPR + scratch） | `__forceinline__` |
| 權重指標不是直接來自 kernel 參數，改變了 codegen（VGPR +30–90%） | 依呼叫形狀使用不同的入口 |
| 管線化迴圈中的 `break` 讓等待計數器每步歸零 | 無 `break` 的主迴圈 + 另外的尾端 |
| wave 分歧的 scale 分支使記憶體延遲序列化 | 無分支的索引算術 |
| 沒有選用 `v_mad_i32_i24`（編譯器用了 `mul_lo_u32` 或 `mul24 + add3`） | inline asm |
| LDS bank 衝突 | fragment 順序佈局、補齊 |
| 會溢出的 tile 設定 | 編譯所有候選，只保留無溢出者 |

務必用 code object 的中繼資料確認重構後 VGPR/SGPR/scratch 數量不變（關於建置旗標一致，見 [windows-hip.md](windows-hip.md#build)）。

## <a id="fma"></a>12. 浮點收縮與逐位元相同

融合或重構 kernel 可能在數學不變的情況下改變結果，因為 clang 可以自由地把 `a*b + c` 收縮成 `fma`，而且程式碼移動時可能選擇不同的收縮方式。我們遇過的案例：

| 變更 | 編譯器做了什麼 | 損害 |
|---|---|---|
| 融合時把 conv `h0*w0 + h1*w1 + h2*w2 + x*w3` 移進 inline 函式 | 選擇了不同的 fma 收縮順序 | 最後 token 的 logits 差異最多 0.17 |
| RMSNorm 融合進 GEMM 輸入產生者 | 把 `x*x` 收縮進第一個 butterfly 加法；把乘法與 f16 轉換合併成 `v_fma_mix` | 每個元素 1 ulp |
| 遞迴步驟 kernel 改寫 | 把 `y*y` 收縮進 shuffle-add 的 fma | 非逐位元相同 |

我們遵守的規則：

1. 在位元很重要的 kernel 中，把標準運算式寫成明確的 `__builtin_fmaf` 鏈（等同參考 kernel 編譯出來的結果），並／或使用
   `#pragma clang fp contract(off)`。
2. 在平方與最終乘積之後放 `asm volatile("" : "+v"(v))`，阻止跨越這個邊界的收縮。
3. 以相同的 lane 排列（`block_sum`、xor 樹）重現歸約，而不是用「數學上相等」的另一種。
4. 整數 WMMA 是精確的；f16 WMMA 只有在輸入與累加順序都相同時才精確——而在 gfx11 上有一種情況連這樣都不行（P = 0 乘以另一個 query 的 V，§8）。
5. 乘以恰好 1（延遲 rescale）以及改變 M/N 方向的 tile 形狀（GEMM bucket）是安全的；改變 K 的順序則不安全。
6. 端到端驗證：對多個提示詞跨執行檔逐位元比較最後 token 的 logits，再加上常設的各 kernel 檢查。

完整的 kernel 與數值相關坑清單見 [pitfalls.md](pitfalls.md#kern)。
