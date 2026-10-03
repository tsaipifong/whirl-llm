[English](../en/index.md) | **繁體中文**

# WHIRL 指南

WHIRL（*Windows HIP Inference for RDNA LLMs*）是以 C++ 與 HIP 撰寫、授權為 Apache-2.0 的 LLM 推論引擎，
針對 AMD RDNA GPU、原生在 Windows 上執行。它走「專門化」路線——一次加入一種模型架構，並為每一種架構、
每一張 GPU 調校所有 kernel——依循 [NInfer](https://github.com/Neroued/ninfer) 的理念。WHIRL 不含 NInfer
的任何程式碼或文字。

這些文件說明 WHIRL 如何運作、每個設計為什麼這樣選、我們量到了什麼、哪些替代方案輸了，以及——寫得最詳細的
——我們踩過的每一個坑。其中許多坑是 Windows 上的 HIP 與 RDNA 3.5/4 特有的，其他地方查不到；我們把它們寫下來，
讓其他專案可以直接拿去用。

> **狀態（WHIRL 0.1.0，2026-10-03）。** 這裡描述的一切都已在本儲存庫的 C++ 引擎中實作（R9700，gfx1201）：
> kernel、載入器、forward pass、推測解碼、含分層快取的伺服器，以及圖片輸入。gfx1151（Radeon 8060S）版本在
> 規劃中。這些文件中有部分數字是在 C++ 引擎之前的研究建置上量測的；C++ 引擎的輸出相同，速度也相同（差距 ±1% 內）。
> 發行版量測見 [benchmarks.md](benchmarks.md)。

## 文件

| 文件 | 內容 |
|---|---|
| [usage.md](usage.md) | 每個指令、選項、環境變數、伺服器端點與結束代碼的參考 |
| [architecture.md](architecture.md) | 整體流程（GGUF → 載入器 → kernel → forward → server）、每種 GPU 一個 code object、為什麼要專門化、正確性模型 |
| [windows-hip.md](windows-hip.md) | Windows 上的 HIP：PAL 與 ROCr 的差異、沒有效能分析工具時如何計時、記憶體（大配置、mmap、VMM、WDDM 降級、pinned 記憶體算成共享使用量）、stream、裝置編號、eGPU、shell 陷阱 |
| [kernels.md](kernels.md) | 每一個 kernel 家族：`v_dot4`／`v_perm` 的 int8 GEMV、逐位元相同的多列 GEMV、int8 WMMA 中批次 GEMV、合併 launch、prefill GEMM、MXFP4×fp8 WMMA、flash attention、split-K decode attention、DeltaNet、MoE、暫存器紀律、fma 收縮 |
| [speculative-decoding.md](speculative-decoding.md) | MTP 草擬、逐位元相同的驗證要求、DeltaNet 快照 → replay、自動草稿數、2-bit 草擬頭、n-gram 共同草稿、EOS 截斷、與 llama.cpp 的比較 |
| [kv-and-caching.md](kv-and-caching.md) | 分頁 KV 池、KV 格式（f16／q8／q8h／q8v）與自動策略、128k + 64k 下限規則、DeltaNet 檢查點、系統提示檢查點、forward 合併、VRAM → RAM → SSD 分層 |
| [server.md](server.md) | OpenAI 相容 API、取樣、工具呼叫、`reasoning_content`、continuous batching、log、限制 |
| [benchmarking.md](benchmarking.md) | 量測方法（交錯 A/B、min–max、與 llama.cpp 的統一協定、快取與散熱陷阱、只用專用 VRAM 規則）、正確性關卡、**目前的數字** |
| [benchmarks.md](benchmarks.md) | WHIRL 0.1.0 對 llama.cpp 的發行版量測（prefill、decode、並發、快取 TTFT、VRAM、視覺） |
| [quantization.md](quantization.md) | 精確模式（Q4_K_M）與速度模式（MXFP4）、MXFP4 的處理、困惑度、Swift-1.5 MXFP4 量化步驟、讓 GGUF 跑得好的檢查清單、mmproj |
| [vision.md](vision.md) | 按需載入的圖片輸入：設計、數值、找到的 bug |
| [pitfalls.md](pitfalls.md) | **125 個坑**，每個都有 條件 → 症狀 → 根本原因 → 修正 → 關卡，外加可通用的教訓 |

## 閱讀路線

- **「我想跑起來。」** [README](../../README_zh-TW.md) → [快速上手](../../quickstart_zh-TW.md) →
  [quantization.md](quantization.md#rules)（選哪個 GGUF）→ [使用參考](usage.md) → [server.md](server.md)。
- **「我在 Windows 上寫 HIP。」** [windows-hip.md](windows-hip.md) → [pitfalls.md](pitfalls.md#hip)。
- **「我寫 RDNA kernel。」** [kernels.md](kernels.md) → [pitfalls.md](pitfalls.md#kern)。
- **「我在做 llama.cpp。」** [pitfalls.md](pitfalls.md#tok-1)（tokenizer）、[pitfalls.md](pitfalls.md#hip-2) 與
  [#hip-3](pitfalls.md#hip-3)（Windows 上的 ROCm）、[speculative-decoding.md](speculative-decoding.md#vs-llamacpp)、
  [quantization.md](quantization.md#swift)。
- **「我在服務混合架構模型。」** [kv-and-caching.md](kv-and-caching.md) →
  [speculative-decoding.md](speculative-decoding.md#replay)。
- **「我在量測 AMD GPU。」** [benchmarking.md](benchmarking.md)。

## 慣例

- **硬體。** R9700 = AMD Radeon AI PRO R9700（RDNA 4、gfx1201、32 GB），主要目標。在開發機上它是
  **USB4 eGPU**（host 連結約 3.8 GB/s）；這條連結造成的現象一律標為*環境限制*。8060S = AMD Radeon 8060S
  （RDNA 3.5、gfx1151），裝在散熱受限的筆電裡。作業系統：Windows 11 build 26200；HIP SDK 7.2。
- **數字**都是實測，附單位與條件；範圍是交錯多輪的 min–max。decode 數字先報 WHIRL 的預設模式
  （MTP + n-gram）；「不開 MTP」的數字會明確標示——在這張卡上，27B 4-bit 模型的一般 decode 受記憶體頻寬
  限制，上限約 38 tok/s。
- **逐位元相同**指的是位元完全相同，並由常設檢查把關。
- **名稱。** 命令列旗標與 `WHIRL_*` 環境變數都是 WHIRL 0.1.0 的名稱，全部列在[使用參考](usage.md)。
- **未知**的數字標為 TODO，不做推估。

## 致謝

理念：[NInfer](https://github.com/Neroued/ninfer)（專門化引擎；Apache-2.0）。改寫自第三方的程式碼：
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)（MIT：tokenizer 預切詞與 BPE 合併迴圈、IQ 查表）。
技術：[r9700-stack](https://github.com/bkvargyas/r9700-stack)（Apache-2.0：E8M0 → fp8 指數折疊、fragment
排列的 fp8 activation、只等 LDS 的 barrier 指令序列）。Strix Halo 的量測方法：[gufo](https://github.com/gufo-org/gufo)。
分層快取排程的想法（delay hit、命中即預取）：Strata 論文（arXiv 2508.18572）。詳見
[THIRD_PARTY_NOTICES.md](../../../THIRD_PARTY_NOTICES.md) 與 [PROVENANCE.md](../../../PROVENANCE.md)。
