[English](../README.md) | **繁體中文**

# WHIRL

**Windows HIP Inference for RDNA LLMs**

WHIRL 是為 AMD Radeon AI PRO R9700 打造、原生在 Windows 上執行的 LLM 推論引擎，以純 C++ 與 HIP 寫成：不用
WSL、不用 Linux 虛擬機，底下也沒有 llama.cpp runtime。它有兩根支柱：

- **快速的 decode。** 為 RDNA 4 手工調校的 kernel（int8 GEMV 達到實測記憶體頻寬上限、WMMA prefill、MXFP4 × fp8
  矩陣路徑），再加上 **MTP + n-gram 推測解碼**，輸出與一般 greedy decode 逐位元相同。
- **為長時間、多輪工作設計的伺服器。** OpenAI 相容伺服器，具備 continuous batching，以及**前綴快取與多層 KV 快取
  （VRAM → pinned 主記憶體 → SSD）**，回到舊對話或共用的系統提示時都不必重新 prefill。

WHIRL 走專門化路線，而不是泛用引擎：一次加入一種模型架構，並為每一張支援的 GPU 調校。目前支援的架構是
**`qwen35`（dense）**與 **`qwen35moe`（混合專家，MoE）**，之後會逐一加入更多架構。WHIRL 依循
[NInfer](https://github.com/Neroued/ninfer)（Apache-2.0）的設計理念——為選定的模型與選定的 GPU 打造，盡可能逼近
硬體極限；[zynfer](https://github.com/thanos/zynfer) 把同樣的想法帶到了 RDNA 4。

## 需求

| | |
|---|---|
| GPU | **AMD Radeon AI PRO R9700**（RDNA 4，gfx1201）——必要；只支援單張 GPU。**Radeon 8060S（Ryzen AI Max+ 395，gfx1151）支援開發中**，見 [`gfx1151` 分支](https://github.com/tsaipifong/whirl-llm/tree/gfx1151) |
| 作業系統 | **Windows 11**，64 位元 |
| 驅動程式 | **AMD Software: Adrenalin Edition 26.8.1**（驅動程式 32.0.31041.1004）或更新版 |
| 執行 | 只需要驅動程式（它會安裝 `amdhip64_7.dll` 與 `amd_comgr_3.dll`）。不用 HIP SDK、不用 ROCm、不用 Visual C++ runtime |
| 從原始碼建置 | HIP SDK 7.2 + Visual Studio 2022 Build Tools（MSVC）+ CMake + Ninja——見[從原始碼建置](building_zh-TW.md) |
| 記憶體 / 磁碟 | 預設設定下，伺服器會為主記憶體快取層 pin 住約 8～9 GiB 主記憶體，SSD 快取層最多使用 64 GiB；兩者都可調整或關閉（`--kv-ram-mb`、`--kv-ssd-gb`） |

我們的測試機以 USB4 / Thunderbolt eGPU 連接 R9700，可以正常使用。插在主機板 PCIe 插槽上的顯示卡，數字可能略有
不同（主要是資料要經過主機連結的部分，例如載入模型、主記憶體 / SSD 快取層）。

## 快速上手

1. 從 [Releases](https://github.com/tsaipifong/whirl-llm/releases) 下載 `whirl-0.1.0-windows-x64.zip`，以及一個
   [支援的模型](#models)。
2. 在 PowerShell 執行：

```powershell
Unblock-File .\whirl-0.1.0-windows-x64.zip
Expand-Archive .\whirl-0.1.0-windows-x64.zip -DestinationPath C:\whirl
cd C:\whirl\whirl-0.1.0-windows-x64
.\whirl.exe devices                                    # 應該會列出 AMD Radeon AI PRO R9700 (gfx1201)
.\whirl.exe chat C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf "用兩句話說明 TCP 慢啟動（slow start）。" --max-tokens 400
.\whirl-server.exe C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf --port 8080
```

伺服器在 `http://127.0.0.1:8080/v1` 提供 OpenAI API（`/v1/chat/completions`、`/v1/completions`、`/v1/models`、
`/health`），任何 OpenAI 相容的用戶端都能用。按 Ctrl+C 停止；快取的 session 會先寫入 SSD 層。

第一次使用某個模型時會先調校一次 GPU kernel——27B Q4_K_M 約 1.5 分鐘，MXFP4 檔只要幾秒——結果快取在
`%LOCALAPPDATA%\whirl`。完整步驟：[快速上手](quickstart_zh-TW.md)。所有選項：[使用參考](guide/zh-TW/usage.md)。

## <a id="models"></a>支援的模型

WHIRL 讀取 GGUF 的 `general.architecture`，只接受 `qwen35` 與 `qwen35moe`（Gated DeltaNet + gated attention 的
混合架構，可帶 MTP 層），以及保留這個架構的微調模型。其他架構（Llama、Gemma、Mistral、DeepSeek、`qwen3`、
`qwen2` 等）會在載入時被拒絕並顯示清楚的訊息；WHIRL 沒有 kernel 的 tensor 型別（例如 NVFP4）也一樣，不會退回
某條很慢的泛用路徑。

> **推薦：我們自己量化的 MXFP4 檔效能最好。**
> - Dense 27B：[**Swift-1.5-Qwen3.8-27b-MXFP4-GGUF**](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF)，
>   A 版（`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`）。Swift-1.5 的思考也比原模型精簡得多（在我們的中英混合
>   程式提示上，思考 token 少 44%）。
> - MoE 35B（約 3B 啟用）：**Ornith-1.5-35B-A3B MXFP4**：[tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF)。

測試過的 GGUF 檔（可載入、greedy 輸出已檢查、MTP 輸出與一般 greedy 相同，並在 R9700 上量測過）：

| 模型檔（Hugging Face） | 類型 | 架構 | 量化 | 模式 |
|---|---|---|---|---|
| [tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF) `…-A-outQ6_K.gguf`（推薦）、`…-B-outQ8_0.gguf`、`…-C-outQ4_K.gguf` | dense 27B | qwen35 | MXFP4（+ Q8_0 MTP / embedding） | 速度模式；我們為 WHIRL 量化 |
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `Qwen3.8-27B-UD-Q4_K_M.gguf` | dense 27B | qwen35 | UD-Q4_K_M | 精確模式（各項最佳化前後輸出逐位元相同） |
| [FreedomAISVR/Qwen3.8-27B-MXFP4-GGUF](https://huggingface.co/FreedomAISVR/Qwen3.8-27B-MXFP4-GGUF) `qwen3.8-27b-mxfp4.gguf` | dense 27B | qwen35 | MXFP4 | 速度模式 |
| [`Ornith-1.5-35B-A3B-MXFP4.gguf`](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF)（我們量化） | MoE 35B，約 3B 啟用 | qwen35moe | MXFP4 專家與 dense（+ Q8_0 MTP / embedding、Q6_K head） | 速度模式（專家 prefill 走 fp8） |
| [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) `Ornith-1.5-35B-Q4_K_M.gguf` | MoE 35B，約 3B 啟用 | qwen35moe | Q4_K_M | 精確模式 |

圖片輸入搭配 Qwen3-VL 形式的 mmproj（F16 / BF16），以 `--mmproj` 指定；已用上表的 27B 檔測試。其他 `qwen35` /
`qwen35moe` GGUF 應該可以載入，但未經測試。如何選擇與製作 GGUF：[quantization.md](guide/zh-TW/quantization.md)。

## 效能

WHIRL 0.1.0 對 llama.cpp b11214（ROCm，每一列都用我們找到最快的旗標），R9700，greedy 解碼，兩個引擎用相同的
提示。decode 數字先報 WHIRL 的預設模式 **MTP + n-gram**；標示 **no MTP** 的列比較的是不用推測解碼的一般 decode。

<p>
<img src="images/bench_decode_zh.png" width="49%" alt="Decode 速度：中文寫程式提示">
<img src="images/bench_server_concurrency.png" width="49%" alt="伺服器 1、2、4 人並發的總吞吐">
<img src="images/bench_prefill.png" width="49%" alt="Prefill 速度與提示長度">
<img src="images/bench_ttft_cache.png" width="49%" alt="前綴已快取與 session 還原時的首 token 時間">
</p>

| R9700，greedy——WHIRL 對 llama.cpp（倍數） | Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用） | Swift-1.5 27B MXFP4-A（dense） | Qwen3.8-27B UD-Q4_K_M（dense） |
|---|---|---|---|
| Decode tok/s，7 個中文寫程式提示（800 token）——WHIRL MTP+n-gram 對 llama.cpp 最快¹ | 244.5 vs 118.8（**2.06×**） | 107.5 vs 60.8（**1.77×**） | 98.5 vs 56.0（**1.76×**） |
| 16k token context 之後的 decode tok/s——WHIRL MTP+n-gram 對 llama.cpp 最快¹ | 301.5 vs 107.5（**2.80×**） | 69.5 vs 63.5（1.09×） | 74.3 vs 59.8（1.24×） |
| Decode tok/s，同樣 7 個提示——WHIRL **no MTP** 對 llama.cpp **no MTP** | 166.7 vs 118.8（1.40×） | 37.5 vs 33.4（1.13×） | 34.4 vs 30.9（1.11×） |
| 伺服器 4 人並發總吞吐 tok/s（含 prefill）——WHIRL MTP+n-gram 對 llama.cpp 最快¹ | 396.7 vs 191.9（**2.07×**） | 198.6 vs 65.8（**3.02×**） | 134.1 vs 58.5（**2.29×**） |
| Prefill tok/s，8k token 提示（各自的 CLI bench 工具） | 10,858 vs 4,637（**2.34×**） | 3,278 vs 1,338（**2.45×**） | 1,689 vs 1,223（1.38×） |
| Prefill tok/s，32k token 提示 | 7,978 vs 3,778（**2.11×**） | 2,595 vs 1,174（**2.21×**） | 1,479 vs 1,086（1.36×） |
| 首 token 時間：新對話重用 26k token 的系統提示（秒，越低越好） | 0.073 vs 0.290（**4.0×**） | 0.122 vs 0.538（**4.4×**） | 0.136 vs 0.546（**4.0×**） |
| 首 token 時間：伺服器重啟後的 26k token session（從 SSD 層還原） | 0.298 s vs N/A² | 0.729 s vs N/A² | 0.739 s vs N/A² |

¹ 該列 llama.cpp 在 plain / MTP / MTP + n-gram 中最快者（MoE 模型上它的 plain decode 最快）。² llama.cpp 沒有自動的
持久化 KV 快取。

WHIRL 領先**不多**的地方：dense 模型的一般 decode（no MTP）兩邊都受記憶體頻寬限制（1.1～1.2×）；Swift MXFP4-A
在 16k context 之後只有 llama.cpp 最佳設定的 1.09×；Q4_K_M 的 88 token 提示 prefill 幾乎打平（1.03×）；同樣 context
下，dense 模型的 WHIRL CLI 比 llama-bench 多用 3～5 GiB VRAM。我們的 R9700 是 USB4 eGPU；prefill 與 decode 都在 VRAM 內，
但還原與載入模型會經過主機連結。

完整方法與所有數字：[benchmarks.md](guide/zh-TW/benchmarks.md)。

## Windows 安全性提示

`whirl.exe` 與 `whirl-server.exe` 沒有程式碼簽章。剛下載的檔案可能讓 Windows SmartScreen 顯示「Windows 已保護您
的電腦」：按 **其他資訊 → 仍要執行**，或在解壓縮前先對 zip 執行 `Unblock-File`（只對從官方發布頁下載、且已核對
SHA-256 的 zip 這樣做）。**智慧型應用程式控制**設為*開啟*時，Windows 可能直接封鎖程式，而且沒有「仍要執行」
選項。詳見 [Windows 安全性提示](windows_security_zh-TW.md)。

## 文件

| | 繁體中文 | English |
|---|---|---|
| 快速上手 | [quickstart_zh-TW.md](quickstart_zh-TW.md) | [quickstart.md](quickstart.md) |
| 使用參考（所有指令、選項、環境變數、結束代碼） | [usage.md](guide/zh-TW/usage.md) | [usage.md](usage.md) |
| 伺服器（API、取樣、工具呼叫、批次、log） | [server.md](guide/zh-TW/server.md) | [server.md](guide/en/server.md) |
| 與 llama.cpp 的效能比較 | [benchmarks.md](guide/zh-TW/benchmarks.md) | [benchmarks.md](benchmarks.md) |
| Windows 安全性提示 | [windows_security_zh-TW.md](windows_security_zh-TW.md) | [windows_security.md](windows_security.md) |
| 從原始碼建置 | [building_zh-TW.md](building_zh-TW.md) | [building.md](building.md) |
| 指南首頁 | [index.md](guide/zh-TW/index.md) | [index.md](guide/en/index.md) |
| 架構 | [architecture.md](guide/zh-TW/architecture.md) | [architecture.md](guide/en/architecture.md) |
| Kernel | [kernels.md](guide/zh-TW/kernels.md) | [kernels.md](guide/en/kernels.md) |
| 推測解碼（MTP + n-gram） | [speculative-decoding.md](guide/zh-TW/speculative-decoding.md) | [speculative-decoding.md](guide/en/speculative-decoding.md) |
| KV 快取、前綴快取、分層快取 | [kv-and-caching.md](guide/zh-TW/kv-and-caching.md) | [kv-and-caching.md](guide/en/kv-and-caching.md) |
| 量化與模型檔 | [quantization.md](guide/zh-TW/quantization.md) | [quantization.md](guide/en/quantization.md) |
| 圖片輸入 | [vision.md](guide/zh-TW/vision.md) | [vision.md](guide/en/vision.md) |
| Windows 上的 HIP | [windows-hip.md](guide/zh-TW/windows-hip.md) | [windows-hip.md](guide/en/windows-hip.md) |
| 量測方法 | [benchmarking.md](guide/zh-TW/benchmarking.md) | [benchmarking.md](guide/en/benchmarking.md) |
| **踩坑全集**（125 條，每條都有 症狀 → 根本原因 → 修正 → 檢查） | [pitfalls.md](guide/zh-TW/pitfalls.md) | [pitfalls.md](guide/en/pitfalls.md) |

## 開發與 AI 協助

WHIRL 是在專案擁有者（[@tsaipifong](https://github.com/tsaipifong)）的指導下，由 Anthropic 的 AI 模型
**Claude Opus 5.5** 參與設計、實作、最佳化與量測。Anthropic 與本專案沒有隸屬關係，也不為本專案背書。

## 授權與致謝

Apache License 2.0；見 [LICENSE](../LICENSE) 與 [NOTICE](../NOTICE)。第三方素材列在
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)，每個原始檔的來源列在 [PROVENANCE.md](../PROVENANCE.md)。

- [NInfer](https://github.com/Neroued/ninfer)——WHIRL 依循的設計理念（只借理念）。
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)——我們量測時的比較基準，也是 GGUF 與量化格式相容性的
  參考；少數改寫的部分（tokenizer 預切詞 / BPE 迴圈、查表、圖片前處理）採 MIT 授權，列在 THIRD_PARTY_NOTICES.md。
- [gufo](https://github.com/gufo-org/gufo) 與 [r9700-stack](https://github.com/bkvargyas/r9700-stack)——量測方法與
  最佳化想法。
- 模型作者：Qwen 團隊（Qwen3.8）、[unsloth](https://huggingface.co/unsloth)、
  [FreedomAISVR](https://huggingface.co/FreedomAISVR)、[ornith-ai](https://huggingface.co/ornith-ai)（Ornith-1.5），
  以及 [ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)（Swift-1.5，我們的 MXFP4 檔就是由它量化而來）。
