[English](../README.md) | **繁體中文**

# WHIRL

**Windows HIP Inference for RDNA LLMs**

> ### 0.2.0 新增：支援 AMD Ryzen AI Max+ 395（Radeon 8060S）
> WHIRL 現在可以在 **AMD Ryzen AI Max+ 395** 的內建 GPU **Radeon 8060S** 上執行：kernel 針對它調校過，三種數值模式都能用，提示最長約 250k token。在 8060S 上，我們量過的每一種提示長度都比原廠 llama.cpp 的 ROCm 和 Vulkan 版本快（見[下方 8060S 表格](#whirl-對-llamacpp一眼看)；最大功率（80 W），ASUS ROG Flow Z13）。

WHIRL 是 Windows 上 AMD GPU 的開源（Apache-2.0）LLM 推論引擎：原生 C++／HIP，每一支 kernel 都公開，不需要 WSL、Docker，也不用安裝 ROCm。**下載 zip、解壓縮、執行，只需要 AMD 驅動。** 0.2.0 支援 **AMD Ryzen AI Max+ 395** 的 **Radeon 8060S**（0.2.0 新增）與 **AMD Radeon AI PRO R9700**（RDNA 4，32 GB），搭配我們的兩個 MXFP4 模型：**Swift-1.5 27B MXFP4**（稠密）與 **Ornith-1.5-35B-A3B MXFP4**（MoE）。兩張卡上都**比 llama.cpp 快**，ROCm 與 Vulkan 版皆然，三種模式、我們量過的每個提示長度（2k 到 ~250k token）都是如此。

## WHIRL 對 llama.cpp：一眼看

WHIRL 0.2.0 預設模式（balance）對 ggml-org 官方 llama.cpp **b11214**，**ROCm** 與 **Vulkan** 兩個版本分開列。同一個 GGUF 檔、同一組提示、greedy 解碼、一個使用者。

![Radeon 8060S（Ryzen AI Max+ 395）prefill 速度：WHIRL 0.2.0 對 llama.cpp ROCm 與 Vulkan](images/bench020_prefill_s8.png)

![Radeon 8060S（Ryzen AI Max+ 395）decode 速度：WHIRL 0.2.0 對 llama.cpp ROCm 與 Vulkan](images/bench020_decode_s8.png)

![Radeon AI PRO R9700 prefill 速度：WHIRL 0.2.0 對 llama.cpp ROCm 與 Vulkan](images/bench020_prefill_r97.png)

![Radeon AI PRO R9700 decode 速度：WHIRL 0.2.0 對 llama.cpp ROCm 與 Vulkan](images/bench020_decode_r97.png)

![寫程式 agent 工作階段耗時：WHIRL 0.2.0 對 llama.cpp ROCm 與 Vulkan](images/bench020_agent.png)

![WHIRL 0.2.0 三種模式（precise、balance、fast）的 32k prefill](images/bench020_modes_pp.png)

![WHIRL 0.2.0 三種模式在 32k 文件之後的 decode](images/bench020_modes_tg.png)

<!-- gen:lead.zh -->
- **Radeon 8060S**（最大功率（80 W），ASUS ROG Flow Z13）：8k prefill Ornith 2,244 tok/s（+75%／+199%）、Swift 470.8（+53%／+101%）；128k token 文件之後的 decode Ornith 58.4（+54%／+57%）、Swift 23.8（+46%／+105%）。
- **R9700 prefill**：Ornith 35B-A3B 讀 2k token 提示 **11,550 tok/s**（llama.cpp ROCm 4,014、Vulkan 3,551：WHIRL +188%／+225%），~250k token 提示 2,664 tok/s（1,519／1,218：+75%／+119%）。Swift 27B：32k 2,805 tok/s（+163%／+256%），128k 1,715（+139%／+208%）。
- **R9700 decode**：Ornith 回答 2k token 文件的問題 238.4 tok/s（+107%／+106%）；Swift 在 32k token 文件之後 79.2 tok/s（+80%／+84%）。
- **真實的 coding agent 工作階段**（Swift 27B，冷啟動）：8060S 95.7 秒，ROCm 145.6 秒、Vulkan 264.4 秒；R9700 17.0 秒，ROCm 81.0 秒、Vulkan 69.4 秒。
<!-- /gen -->

**Radeon 8060S**（gfx1151，Ryzen AI Max+ 395），**最大功率（80 W），ASUS ROG Flow Z13**。llama.cpp 每一格括號裡的百分比，是 WHIRL 領先該格的幅度。

<!-- gen:glance.s8.zh -->
| tok/s，越高越好 | Swift 27B：WHIRL | llama.cpp ROCm | llama.cpp Vulkan | Ornith 35B-A3B：WHIRL | llama.cpp ROCm | llama.cpp Vulkan |
|---|---|---|---|---|---|---|
| Prefill，2k token 提示 | **498.4** | 305.7 (WHIRL +63%) | 162.8ᶜ (WHIRL +206%) | **2,411** | 1,259 (WHIRL +91%) | 447.7ᶜ (WHIRL +439%) |
| Prefill，32k token 提示 | **426.7** | 266.2 (WHIRL +60%) | 221.6 (WHIRL +93%) | **1,829** | 1,049 (WHIRL +74%) | 756.2 (WHIRL +142%) |
| Prefill，128k token 提示 | **298.9** | 180.8 (WHIRL +65%) | 127.8 (WHIRL +134%) | **1,055** | 636.6 (WHIRL +66%) | 428.4 (WHIRL +146%) |
| Prefill，~250k token 提示 | **221.3** | 135.1 (WHIRL +64%) | 78.1 (WHIRL +183%) | **685.0** | 418.3 (WHIRL +64%) | 278.9 (WHIRL +146%) |
| Decode，2k token 文件問答 | **31.2** | 24.4 (WHIRL +28%) | 25.1 (WHIRL +24%) | **101.4** | 70.0 (WHIRL +45%) | 70.7 (WHIRL +43%) |
| Decode，32k token 文件問答 | **29.5** | 22.7 (WHIRL +30%) | 17.0 (WHIRL +74%) | **83.8** | 57.9 (WHIRL +45%) | 57.1 (WHIRL +47%) |
| Decode，128k token 文件問答 | **23.8** | 16.2 (WHIRL +46%) | 11.6 (WHIRL +105%) | **58.4** | 38.0 (WHIRL +54%) | 37.3 (WHIRL +57%) |
| Decode，寫程式對話（全新撰寫） | **34.2** | 27.6 (WHIRL +24%) | 28.3 (WHIRL +21%) | **106.0** | 70.6 (WHIRL +50%) | 72.9 (WHIRL +45%) |
<!-- /gen -->

**Radeon AI PRO R9700**（gfx1201，32 GB）。

<!-- gen:glance.r97.zh -->
| tok/s，越高越好 | Swift 27B：WHIRL | llama.cpp ROCm | llama.cpp Vulkan | Ornith 35B-A3B：WHIRL | llama.cpp ROCm | llama.cpp Vulkan |
|---|---|---|---|---|---|---|
| Prefill，2k token 提示 | **3,273** | 1,124 (WHIRL +191%) | 829.4 (WHIRL +295%) | **11,550** | 4,014 (WHIRL +188%) | 3,551 (WHIRL +225%) |
| Prefill，32k token 提示 | **2,805** | 1,065 (WHIRL +163%) | 787.5 (WHIRL +256%) | **8,469** | 3,872 (WHIRL +119%) | 3,395 (WHIRL +149%) |
| Prefill，128k token 提示 | **1,715** | 718.7 (WHIRL +139%) | 557.2 (WHIRL +208%) | **4,275** | 2,307 (WHIRL +85%) | 1,922 (WHIRL +122%) |
| Prefill，~250k token 提示 | **1,153** | 342.0ᵃ (WHIRL +237%) | 259.9ᵃ (WHIRL +344%) | **2,664** | 1,519 (WHIRL +75%) | 1,218 (WHIRL +119%) |
| Decode，2k token 文件問答 | **87.6** | 50.5 (WHIRL +74%) | 50.4 (WHIRL +74%) | **238.4** | 115.4 (WHIRL +107%) | 115.9 (WHIRL +106%) |
| Decode，32k token 文件問答 | **79.2** | 44.0 (WHIRL +80%) | 43.1 (WHIRL +84%) | **199.1** | 101.4 (WHIRL +96%) | 100.6 (WHIRL +98%) |
| Decode，128k token 文件問答 | **53.9** | 33.2 (WHIRL +62%) | 34.8 (WHIRL +55%) | **131.3** | 75.3 (WHIRL +74%) | 73.8 (WHIRL +78%) |
| Decode，寫程式對話（全新撰寫） | **109.6** | 57.8 (WHIRL +90%) | 57.6 (WHIRL +90%) | **241.5** | 118.9 (WHIRL +103%) | 115.1 (WHIRL +110%) |
<!-- /gen -->

<!-- gen:foot.glance.zh -->
ᵃ R9700＋Swift 27B 的 ~250k：官方 llama.cpp 用 f16 KV 在這個長度放不進 32 GB（記錄檔："failed to fit params to free device memory"），溢到共用記憶體，decode 因此崩落。表中照實列出量到的數字，但這一格不拿來當任何領先的依據。

ᶜ 8060S 的 2k prefill、llama.cpp Vulkan：這張卡上 Vulkan 處理短提示的 prefill 特別慢（遠低於它在 8k 以上的趨勢），所以對 Vulkan 的 2k 領先不當主打；對 Vulkan 的主打領先一律用 8k 以上。
<!-- /gen -->

各列量的是什麼（確切提示與指令：[benchmarks.md](guide/zh-TW/benchmarks.md#prompts)）：

- **Prefill**：處理該長度長文件提示的速度。「~250k」是 249,510 token 的提示。
- **Decode，文件問答**：每個長度問同一個問題（說明文件開頭附近引用的一個函式，並寫出修正版，256 token）。部分回答可以照抄提示裡的程式碼，這對每個使用投機草稿的引擎都有幫助。
- **Decode，寫程式對話（全新撰寫）**：一個寫程式問題，回答（說明與程式碼）全部是新寫的，提示裡沒有可照抄的內容，對投機草稿是困難的情況。512 token。
- **各引擎怎麼解碼**：WHIRL 用自己的 MTP（+ n-gram）草稿。llama.cpp 在 Swift 27B 用 MTP + n-gram（`--spec-type draft-mtp,ngram-mod`），是它在這個模型上最快的模式；在 Ornith **不用**投機解碼，因為我們量到它的 MTP 在這個 MoE 模型上比一般解碼慢。所以 Ornith 的 decode 列是兩邊**各自出廠設定**的對比，不是同條件對比。
- R9700 每格都是單次量測（WHIRL 的寫程式對話除外：3 次中位數），差距小於約 5% 視為雜訊。三種 WHIRL 模式的完整表格：[效能](#效能)。

### 真實的 coding agent 工作階段

Swift 27B，WHIRL balance。一段工作階段＝約 20k token 的系統＋工具提示（coding agent 指示加上 skills／MCP 工具手冊）、一次 `read_file` 工具呼叫回傳約 3.6k token 的 C++ 原始檔，接著四輪使用者提問（說明這個檔案、找出最複雜的函式、改寫該函式、審查使用者送來的修改版），每次呼叫都重送完整歷史，每個回答最多 384 token。測試資料在 repo 裡（[benchmarks.md §6](guide/zh-TW/benchmarks.md#agent-session)）。

**Radeon 8060S**（最大功率（80 W），ASUS ROG Flow Z13）：

<!-- gen:agent.s8.zh -->
| 工作階段（總時間） | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 比 ROCm／Vulkan 省下的時間 | 模型呼叫次數 WHIRL／ROCm／Vulkan |
|---|---|---|---|---|---|
| 冷啟動：20k token 的系統＋工具提示尚未快取 | **95.7 秒** | 145.6 秒 | 264.4 秒 | 34% / 64% | 6/6/6 |
| 第二段：共用同一個前綴，處理另一個檔案 | **44.9 秒** | 66.2 秒 | 196.8 秒 | 32% / 77% | 5/5/6 |
<!-- /gen -->

**R9700：**

<!-- gen:agent.r97.zh -->
| 工作階段（總時間） | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 比 ROCm／Vulkan 省下的時間 | 模型呼叫次數 WHIRL／ROCm／Vulkan |
|---|---|---|---|---|---|
| 冷啟動：20k token 的系統＋工具提示尚未快取 | **17.0 秒** | 81.0 秒 | 69.4 秒 | 79% / 76% | 5/6/6 |
| 第二段：共用同一個前綴，處理另一個檔案 | **11.8 秒** | 60.9 秒 | 38.0 秒 | 81% / 69% | 6/5/6 |
<!-- /gen -->

每個引擎寫出自己的回答，所以模型呼叫次數不同（最後一欄：重讀一次檔案的引擎會多一次呼叫）。每格單次量測。

## 為什麼 WHIRL 跑這些 MXFP4 檔特別快

我們的 MXFP4 檔是為 WHIRL 量化的，WHIRL 也有專為它們寫的 kernel：

<!-- gen:why.zh -->
- **R9700 上的 fp8 prefill**（balance 與 fast）：prefill 的 MXFP4 矩陣乘法跑在 RDNA 4 的 fp8 矩陣單元上（Swift 27B 2k：3,273 tok/s，precise 不用 fp8 為 1,702）。8060S 沒有 fp8 單元，走一般路徑。
- **`ssm_alpha`／`ssm_beta` 用 Q8_0**（而不是 F32）：單 token decode 與多 token 的 MTP 驗證因此走同一個融合 kernel，MTP 輸出與單純 greedy 解碼逐位元相同，草稿也維持夠長。
- **Q6_K 輸出頭**：WHIRL 草稿步驟最快的輸出頭（每個草稿 token 都要經過 24.8 萬列的輸出頭）。
<!-- /gen -->

每一支 kernel 都在這個 repo 開源，沒有封閉元件，執行時也不會另外下載。

## 需求

| | |
|---|---|
| **下載就能跑** | 從 [Releases](https://github.com/tsaipifong/whirl-llm/releases) 下載 **`whirl-0.2.3-windows-x64.zip`**，解壓縮、執行。**只需要 AMD 驅動**：不需要 HIP SDK、ROCm，也不用另外安裝 Visual C++ 可轉散發套件（ROCm 10.1 驅動需要的幾個執行階段 DLL 已附在 zip 裡） |
| 驅動 | **AMD Software: Adrenalin Edition 26.8.1（驅動 32.0.31041.1004）或更新版**，兩張卡都一樣。更舊的版本未測試。使用 Adrenalin Edition for ROCm 10.1 驅動時，請讓 zip 裡的 `rocm_kpack.dll` 留在 `whirl.exe` 旁邊（該驅動的 GPU runtime 需要它，但驅動沒有安裝它），連同它用到的 Visual C++ 執行階段 DLL（`msvcp140.dll`、`vcruntime140.dll`、`vcruntime140_1.dll`，同樣附在 zip 裡） |
| GPU | **Radeon 8060S**（RDNA 3.5，gfx1151，AMD Ryzen AI Max+ 395 內顯）或 **AMD Radeon AI PRO R9700**（RDNA 4，gfx1201，32 GB）；一個行程用一張 GPU |
| Radeon 8060S：GPU 記憶體 | **先在 Armoury Crate 或 BIOS 設定 GPU 記憶體（UMA frame buffer 大小）**（其他筆電：AMD Software 的 *Variable Graphics Memory*），再重新開機。WHIRL 把模型、緩衝區和 KV 快取放在這塊專用記憶體；Windows 的「共用 GPU 記憶體」不拿來放這些（token 嵌入表和 RAM 快取層放在鎖定的系統 RAM，工作管理員會把它顯示成共用 GPU 記憶體）。**Swift-1.5 27B MXFP4 至少設 40 GB**（precise 模式搭配預設 KV 快取要 48 GB），**Ornith-1.5-35B-A3B MXFP4 至少 32 GB**（128 GB 的機器設 64 GB，任一模型、任一模式都能跑）。以 server 預設值量測（4 個 slot、262,144 token 的 KV 快取）：Swift 在 balance 用 29.9 GiB、fast 31.0 GiB、precise 38.2 GiB；Ornith 在 balance 與 precise 用 27.3 GiB、fast 23.4 GiB。設得比這少時，server 會縮小 KV 快取，或停下來並告訴你要設多大。`whirl devices` 會以「dedicated」顯示目前的大小 |
| 作業系統 | Windows 11 64 位元 |
| 模型 | Hugging Face 上的 [Swift-1.5 27B MXFP4](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF) 或 [Ornith-1.5-35B-A3B MXFP4](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF)（[支援的模型](#支援的模型)） |
| 記憶體／磁碟 | server 會鎖定一部分主記憶體作 RAM 快取層（Radeon 8060S：Windows 看得到的 RAM 的 1/8，64 GB 機器為 8 GiB，低於 32 GB 則關閉；R9700 主機：約 RAM 的 1/4，64 GB 電腦為 16 GiB），SSD 層最多用 64 GiB，只在 server 停止時寫入；兩者都可調整或關閉（`--kv-ram-mb`、`--kv-ssd-gb`） |
| 進階：從原始碼建置 | 給想自己修改、調校 WHIRL 的使用者：HIP SDK 7.2 + Visual Studio 2022 Build Tools（MSVC）+ CMake + Ninja；見 [building_zh-TW.md](building_zh-TW.md) |

支援範圍：WHIRL 正式支援 Radeon 8060S 與 R9700，目標是 **16 GB 以上**的卡。其他 AMD GPU 未測試（RX 9070 系列和 R9700 同晶片、16 GB，未測試）。16 GB 以下的卡不在正式支援範圍，歡迎 fork 原始碼自行修改。

我們的 R9700 是用 USB4／Thunderbolt 外接，可以正常運作。直接插在 PCIe 插槽的卡，在資料要經過主機連線的地方（載入模型、RAM／SSD 層還原）數字可能略有不同，prefill 與 decode 不受影響。

## 快速上手

1. 從 [Releases](https://github.com/tsaipifong/whirl-llm/releases) 下載 `whirl-0.2.3-windows-x64.zip`，以及一個模型，例如 [`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF)，存到 `C:\models\`。
2. 解壓縮後執行，在 PowerShell：

```powershell
Unblock-File .\whirl-0.2.3-windows-x64.zip
Expand-Archive .\whirl-0.2.3-windows-x64.zip -DestinationPath C:\whirl
cd C:\whirl\whirl-0.2.3-windows-x64
.\whirl.exe devices                                    # 應列出 Radeon 8060S（gfx1151）和／或 R9700（gfx1201）
.\whirl.exe chat C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf "用兩句話解釋 TCP slow start。" --max-tokens 400
.\whirl-server.exe C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf --port 8080      # 預設模式：balance
```

server 在 `http://127.0.0.1:8080/v1` 提供 OpenAI API（`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/health`），任何相容 OpenAI 的用戶端都能用。加 `--precise` 得到逐位元無損輸出，加 `--fast` 追求最高速度。用 Ctrl+C 停止 server，會先把重複使用過的工作階段寫進 SSD 層。

兩張卡都在時預設用 R9700；`--device 8060s`（或 `WHIRL_DEVICE=8060s`）選 8060S。長文件可加大每個 slot 的上下文，例如 `--ctx-per-slot 262144`；見 [recipes_zh-TW.md](recipes_zh-TW.md#long-context)。

每個模型第一次執行時會調校一次 GPU kernel，結果快取在 `%LOCALAPPDATA%\whirl`（R9700 上我們的 MXFP4 檔只要幾秒；8060S 較久）。完整步驟：[快速上手](quickstart_zh-TW.md)。所有選項：[使用參考](guide/zh-TW/usage.md)。

**進階：從原始碼建置**——給想自己修改、調校 WHIRL 的使用者：[building_zh-TW.md](building_zh-TW.md)。

## 三種模式：速度與精度

WHIRL 0.2.0 有三種數值模式，每個行程選一種（`--precise`、`--balance`、`--fast`）；不加旗標時是 **balance**。

| 模式 | 是什麼 | 輸出保證 |
|---|---|---|
| **precise** | **逐位元無損**：除了 GGUF 本身的權重之外，不再量化任何東西；f16 KV 快取，激活值全程 f16／f32 | 輸出與單純 greedy 解碼**逐位元相同**：有沒有投機草稿、單獨跑或和其他使用者合批、命中前綴快取或還原工作階段，位元都相同 |
| **balance**（預設） | 近乎無損，預設 | MTP／n-gram 草稿、合批，輸出的 token 都和 balance 模式下單純 greedy 解碼相同，還原的工作階段與留在 VRAM 的同一工作階段 token 相同；對話的下一回合從上一段回覆結尾的快取接續，與冷算數值等價，但非逐位元相同，回覆可能不同 |
| **fast**（自選） | 在通過關卡的可接受損失內求最高速度：balance 再加 4-bit KV 快取與放寬的 MTP 接受規則（Radeon 8060S 上的稠密模型保留 int8 KV 快取，在那張卡上 decode 較快） | 有損，但在公布的 KL 門檻內；文字可能隨草稿與其他使用者而不同 |

實際上代表：

- **precise** 是有逐位元保證的模式：逐位元無損，輸出與單純 greedy 解碼完全相同，兩張卡都一樣。
- 在 **precise 與 balance**，每一項加速輸出的 token 都和該模式下單純 greedy 解碼相同：MTP 與 n-gram 投機解碼、多人合批、共用系統提示的前綴快取命中、以及在新對話中（不含第一個回答）對同一份長文件提出的第二個問題；從 RAM 或 SSD 層還原的工作階段，與留在 VRAM 的同一工作階段 token 相同。常駐測試把關（`whirl selftest`、kernel 測試、server 關卡、兩張卡的 golden 輸出）。兩者有一處不同：balance 下，對話的下一回合從上一段回覆結尾的快取接續，那段歷史是 decode kernel 算出來的，所以回覆可能和把整段對話送給一個新啟動的 server 得到的不同（數值等價，KL 最多約 2e-4；還原的工作階段仍與從未離開 VRAM 的同一工作階段逐位元相同）。precise 則改從上一個提示詞最後一個 chunk 的起點重新 prefill，與冷算逐位元相同。
- **precise 與 balance 絕不使用 4-bit KV、3-bit 權重或 fp8 attention。** 只有 fast 可能用到 4-bit KV，而且只在你要求時。
- **權重**就是你選的 GGUF，WHIRL 照檔案原樣計算。在 balance 與 fast，接續對話時沿用上一輪算好的狀態，數值上等價，但和把整段歷史當成一個 prompt 重新處理並非逐位元相同；precise 則逐位元相同（[kv-and-caching.md](guide/zh-TW/kv-and-caching.md)）。

<!-- gen:precise.zh -->
precise 會犧牲速度。R9700 上代價最大的是 MXFP4 檔的 prefill，因為 precise 不用 fp8（Swift 27B 2k：1,702 對 balance 3,273 tok/s；Ornith：7,575 對 11,550）；寫程式對話 Swift decode 83.4 對 109.6 tok/s。8060S 沒有 fp8 矩陣單元，所以 precise 與 balance 的 prefill 很接近（Swift 2k：479.1 對 498.4）。precise 在每個量過的格子仍領先兩個 llama.cpp 版本。
<!-- /gen -->

<!-- gen:fast.zh -->
**fast 模式還沒有完全優化，之後的 0.2.x 小版本會再改進。** 目前實測的效果：主要加快稠密 Swift 27B 的 decode（R9700：2k token 文件之後 115.5 對 balance 87.6 tok/s，寫程式對話 130.4 對 109.6；8060S（最大功率 80 W，ASUS ROG Flow Z13）8k：37.7 對 32.5），以及 Ornith MoE 的長上下文 decode（R9700 ~250k：125.3 對 101.7）。在 Ornith 上，**約 64k token 以下 fast 比 balance 慢**（R9700：2k 208.3 對 238.4、32k 183.5 對 199.1；64k 174.9 對 169.1；8060S 2k：93.8 對 101.4），因為 4-bit KV 快取每個 token 有固定的額外成本，要到長上下文才划算。以上是 0.2.0 的量測；0.2.2 讓 Radeon 8060S 上 Ornith 的 fast 快了 3-5%（文件問答 2k、96k、128k 與寫程式對話；32k 在雜訊範圍內），在 2k、32k 與寫程式對話中與 balance 持平（見 CHANGELOG）。fast 目前不會加快 prefill（R9700 Swift 32k：2,790 對 2,805 tok/s）。
<!-- /gen -->

各模式的完整說明：[usage.md 的數值模式](guide/zh-TW/usage.md#modes)。

## 支援的模型

> **我們自己的 MXFP4 量化，專為 WHIRL 製作**（公布的數字都用這兩個檔）：
> - 稠密 27B：[**Swift-1.5-Qwen3.8-27b-MXFP4-GGUF**](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF)，變體 A（`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`）。Swift-1.5 的思考也比基底模型短很多（見模型卡）。
> - MoE 35B（約 3B 啟用）：[**Ornith-1.5-35B-A3B-MXFP4-GGUF**](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF)。

| 模型檔（Hugging Face） | 類型 | 架構 | 量化 |
|---|---|---|---|
| [tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF) `…-A-outQ6_K.gguf`（推薦）、`…-B-outQ8_0.gguf`、`…-C-outQ4_K.gguf` | 稠密 27B | qwen35 | MXFP4（MTP／embedding／`ssm_alpha`／`ssm_beta` 為 Q8_0）；我們為 WHIRL 量化 |
| [tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF) `Ornith-1.5-35B-A3B-MXFP4.gguf` | MoE 35B，約 3B 啟用 | qwen35moe | 專家與稠密矩陣 MXFP4（MTP／embedding／`ssm_alpha`／`ssm_beta` 為 Q8_0，輸出頭 Q6_K）；我們為 WHIRL 量化 |

兩個檔都在兩張卡上載入並檢查過（greedy 輸出；precise 與 balance 下 MTP 輸出與單純 greedy 相同）。WHIRL 讀取 GGUF 的 `general.architecture`，接受 `qwen35` 與 `qwen35moe`（Gated DeltaNet + gated attention 混合架構，可附 MTP head），所以這兩種架構的其他 GGUF 也能載入，但不是我們量測與調校的對象。其他架構（Llama、Gemma、Mistral、DeepSeek、`qwen3`、`qwen2`……）載入時會被拒絕並給出清楚訊息；WHIRL 沒有 kernel 的張量型別（例如 NVFP4）也一樣；沒有慢速的通用退路。

圖片輸入透過 `--mmproj` 使用 Qwen3-VL 式的 mmproj（F16／BF16），兩張卡都可以；兩個模型 repo 都附有 mmproj。稠密 27B 另有自選的 DFlash2 草稿模型（`--draft dflash`，見[下方](#020-新功能)）。選擇與製作 GGUF：[quantization.md](guide/zh-TW/quantization.md)。

## 效能

WHIRL 0.2.0 對官方 llama.cpp b11214（ggml-org 的 Windows 執行檔），ROCm 與 Vulkan 版並列。同一個 GGUF 檔、同一組提示（文件問答提示，第一行每次不同，不會命中前綴快取；生成 256 token）、greedy 解碼、關閉思考、一個使用者，tok/s 取自各 server 自己的 timings。三種 WHIRL 模式都列出；領先幅度欄是 balance（預設）。**precise 與 fast 也在每一格領先兩個 llama.cpp 版本。** 方法、旗標與提示：[benchmarks.md](guide/zh-TW/benchmarks.md)。

<!-- gen:ranges.zh -->
| WHIRL balance 領先幅度（每個量過的長度） | Prefill 對 ROCm | Prefill 對 Vulkan | Decode 對 ROCm | Decode 對 Vulkan |
|---|---|---|---|---|
| 8060S · Swift 27B | +53% 到 +65% | +93% 到 +183% | +28% 到 +46% | +24% 到 +105% |
| 8060S · Ornith 35B-A3B | +64% 到 +91% | +142% 到 +199% | +41% 到 +54% | +43% 到 +57% |
| R9700 · Swift 27B | +139% 到 +191% | +208% 到 +295% | +56% 到 +80% | +38% 到 +84% |
| R9700 · Ornith 35B-A3B | +75% 到 +188% | +119% 到 +225% | +74% 到 +107% | +78% 到 +106% |

有註腳的格子（ᵃ ᵇ ᶜ）不計入這些範圍；那些格子 WHIRL 也領先。
<!-- /gen -->

### <a id="radeon-8060s"></a>Radeon 8060S，最大功率（80 W），ASUS ROG Flow Z13

Ryzen AI Max+ 395。WHIRL 每格單次量測（2026-10-17）；llama.cpp 為 1 到 3 次的中位數。

<!-- gen:s8.swift.pp.zh -->
#### Swift-1.5 27B MXFP4（稠密）：Prefill tok/s

| 提示長度 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 479.1 | **498.4** | 502.6 | 305.7 | 162.8ᶜ | +63% | +206%ᶜ |
| 8k | 446.5 | **470.8** | 471.8 | 307.5 | 234.4 | +53% | +101% |
| 32k | 401.4 | **426.7** | 428.9 | 266.2 | 221.6 | +60% | +93% |
| 64k | 353.4 | **374.4** | 375.9 | 230.9 | 177.3 | +62% | +111% |
| 96k | 315.7 | **332.9** | 334.1 | 203.3 | 149.0 | +64% | +123% |
| 128k | 282.9 | **298.9** | 297.4 | 180.8 | 127.8 | +65% | +134% |
| ~250k | 197.0 | **221.3** | 222.2 | 135.1 | 78.1 | +64% | +183% |
<!-- /gen -->

<!-- gen:s8.swift.tg.zh -->
#### Swift-1.5 27B MXFP4（稠密）：Decode tok/s，文件問答（256 token 回答：說明文件中引用的一個函式並改寫，部分回答可照抄提示）

| 提示長度 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 30.0 | **31.2** | 37.1 | 24.4 (MTP+n-gram) | 25.1 (MTP+n-gram) | +28% | +24% |
| 8k | 30.8 | **32.5** | 37.7 | 24.6 (MTP+n-gram) | 25.2 (MTP+n-gram) | +32% | +29% |
| 32k | 25.5 | **29.5** | 34.7 | 22.7 (MTP+n-gram) | 17.0 (MTP+n-gram) | +30% | +74% |
| 64k | 23.9 | **26.0** | 31.2 | 18.4 (MTP+n-gram) | 18.0 (MTP+n-gram) | +41% | +44% |
| 96k | 21.8 | **24.1** | 29.6 | 17.9 (MTP+n-gram) | 16.6 (MTP+n-gram) | +35% | +45% |
| 128k | 22.4 | **23.8** | 26.1 | 16.2 (MTP+n-gram) | 11.6 (MTP+n-gram) | +46% | +105% |
| ~250k | 14.6 | **16.8** | 20.7 | 12.2 (MTP+n-gram) | 10.9 (MTP+n-gram) | +38% | +54% |
<!-- /gen -->

<!-- gen:s8.ornith.pp.zh -->
#### Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用）：Prefill tok/s

| 提示長度 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 2,044 | **2,411** | 2,401 | 1,259 | 447.7ᶜ | +91% | +439%ᶜ |
| 8k | 1,867 | **2,244** | 2,232 | 1,282 | 749.2 | +75% | +199% |
| 32k | 1,577 | **1,829** | 1,830 | 1,049 | 756.2 | +74% | +142% |
| 64k | 1,299 | **1,463** | 1,466 | 848.9 | 603.6 | +72% | +142% |
| 96k | 1,108 | **1,227** | 1,229 | 710.4 | 500.5 | +73% | +145% |
| 128k | 966.3 | **1,055** | 1,059 | 636.6 | 428.4 | +66% | +146% |
| ~250k | 644.7 | **685.0** | 697.9 | 418.3 | 278.9 | +64% | +146% |
<!-- /gen -->

<!-- gen:s8.ornith.tg.zh -->
#### Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用）：Decode tok/s，文件問答（256 token 回答：說明文件中引用的一個函式並改寫，部分回答可照抄提示）

| 提示長度 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 91.6 | **101.4** | 93.8 | 70.0 (一般解碼) | 70.7 (一般解碼) | +45% | +43% |
| 8k | 87.5 | **94.9** | 93.6 | 64.3 (一般解碼) | 65.2 (一般解碼) | +48% | +46% |
| 32k | 76.3 | **83.8** | 86.1 | 57.9 (一般解碼) | 57.1 (一般解碼) | +45% | +47% |
| 64k | 67.9 | **69.2** | 76.2 | 49.0 (一般解碼) | 48.4 (一般解碼) | +41% | +43% |
| 96k | 58.5 | **63.9** | 67.0 | 42.7 (一般解碼) | 42.2 (一般解碼) | +50% | +51% |
| 128k | 51.1 | **58.4** | 62.7 | 38.0 (一般解碼) | 37.3 (一般解碼) | +54% | +57% |
| ~250k | 38.9 | **40.4** | 48.9 | 26.3 (一般解碼) | 25.8 (一般解碼) | +54% | +57% |
<!-- /gen -->

<!-- gen:foot.s8.zh -->
ᶜ 8060S 的 2k prefill、llama.cpp Vulkan：這張卡上 Vulkan 處理短提示的 prefill 特別慢（遠低於它在 8k 以上的趨勢），所以對 Vulkan 的 2k 領先不當主打；對 Vulkan 的主打領先一律用 8k 以上。
<!-- /gen -->

### Radeon AI PRO R9700

每格單次量測，WHIRL 於 2026-10-17 量測；差距小於約 5% 視為雜訊。Decode：WHIRL MTP（+ n-gram）；llama.cpp 在 Swift 27B 用 MTP + n-gram，在 Ornith 用一般解碼（各自出廠設定）。

<!-- gen:r97.swift.pp.zh -->
#### Swift-1.5 27B MXFP4（稠密）：Prefill tok/s

| 提示長度 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 1,702 | **3,273** | 3,180 | 1,124 | 829.4 | +191% | +295% |
| 8k | 1,764 | **3,296** | 3,269 | 867.5ᵇ | 705.5ᵇ | +280%ᵇ | +367%ᵇ |
| 32k | 1,626 | **2,805** | 2,790 | 1,065 | 787.5 | +163% | +256% |
| 64k | 1,444 | **2,332** | 2,304 | 913.6 | 691.9 | +155% | +237% |
| 96k | 1,301 | **1,971** | 1,952 | 805.2 | 618.1 | +145% | +219% |
| 128k | 1,185 | **1,715** | 1,697 | 718.7 | 557.2 | +139% | +208% |
| ~250k | 未量 ᵈ | **1,153** | 1,153 | 342.0ᵃ | 259.9ᵃ | +237%ᵃ | +344%ᵃ |
<!-- /gen -->

<!-- gen:r97.swift.tg.zh -->
#### Swift-1.5 27B MXFP4（稠密）：Decode tok/s，文件問答（256 token 回答：說明文件中引用的一個函式並改寫，部分回答可照抄提示）

| 提示長度 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 70.1 | **87.6** | 115.5 | 50.5 (MTP+n-gram) | 50.4 (MTP+n-gram) | +74% | +74% |
| 8k | 74.3 | **90.7** | 102.3 | 28.2ᵇ (MTP+n-gram) | 44.4ᵇ (MTP+n-gram) | +222%ᵇ | +104%ᵇ |
| 32k | 56.4 | **79.2** | 79.3 | 44.0 (MTP+n-gram) | 43.1 (MTP+n-gram) | +80% | +84% |
| 64k | 56.7 | **60.4** | 67.8 | 38.6 (MTP+n-gram) | 39.5 (MTP+n-gram) | +56% | +53% |
| 96k | 49.8 | **56.3** | 63.4 | 36.1 (MTP+n-gram) | 40.7 (MTP+n-gram) | +56% | +38% |
| 128k | 49.4 | **53.9** | 67.0 | 33.2 (MTP+n-gram) | 34.8 (MTP+n-gram) | +62% | +55% |
| ~250k | 未量 ᵈ | **37.9** | 38.8 | 2.3ᵃ (MTP+n-gram) | 6.4ᵃ (MTP+n-gram) | +1548%ᵃ | +492%ᵃ |
<!-- /gen -->

<!-- gen:r97.ornith.pp.zh -->
#### Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用）：Prefill tok/s

| 提示長度 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 7,575 | **11,550** | 11,801 | 4,014 | 3,551 | +188% | +225% |
| 8k | 6,846 | **10,601** | 10,501 | 4,487 | 3,912 | +136% | +171% |
| 32k | 5,959 | **8,469** | 8,450 | 3,872 | 3,395 | +119% | +149% |
| 64k | 4,873 | **6,451** | 6,420 | 3,173 | 2,735 | +103% | +136% |
| 96k | 4,098 | **5,156** | 5,143 | 2,667 | 2,280 | +93% | +126% |
| 128k | 3,520 | **4,275** | 4,270 | 2,307 | 1,922 | +85% | +122% |
| ~250k | 2,351 | **2,664** | 2,666 | 1,519 | 1,218 | +75% | +119% |
<!-- /gen -->

<!-- gen:r97.ornith.tg.zh -->
#### Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用）：Decode tok/s，文件問答（256 token 回答：說明文件中引用的一個函式並改寫，部分回答可照抄提示）

| 提示長度 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 181.8 | **238.4** | 208.3 | 115.4 (一般解碼) | 115.9 (一般解碼) | +107% | +106% |
| 8k | 174.6 | **212.9** | 191.3 | 112.3 (一般解碼) | 111.9 (一般解碼) | +90% | +90% |
| 32k | 154.2 | **199.1** | 183.5 | 101.4 (一般解碼) | 100.6 (一般解碼) | +96% | +98% |
| 64k | 138.6 | **169.1** | 174.9 | 90.7 (一般解碼) | 89.6 (一般解碼) | +86% | +89% |
| 96k | 126.4 | **148.1** | 156.2 | 81.9 (一般解碼) | 81.1 (一般解碼) | +81% | +83% |
| 128k | 121.3 | **131.3** | 152.4 | 75.3 (一般解碼) | 73.8 (一般解碼) | +74% | +78% |
| ~250k | 90.0 | **101.7** | 125.3 | 54.1 (一般解碼) | 53.6 (一般解碼) | +88% | +90% |
<!-- /gen -->

<!-- gen:foot.r97.zh -->
ᵃ R9700＋Swift 27B 的 ~250k：官方 llama.cpp 用 f16 KV 在這個長度放不進 32 GB（記錄檔："failed to fit params to free device memory"），溢到共用記憶體，decode 因此崩落。表中照實列出量到的數字，但這一格不拿來當任何領先的依據。

ᵇ R9700＋Swift 27B 的 8k：llama.cpp 的 8k 請求是新開 server 後的第一個長請求（冷的），數字比它自己的 2k 與 32k 還低。這一格不拿來當任何領先的依據。

ᵈ R9700＋Swift 27B 的 precise ~250k：未量。precise 的 KV 快取用 f16，這個長度的 f16 KV 加上權重放不進 32 GB（f16 池約可容納 19 萬 token）。
<!-- /gen -->

### 寫程式對話（全新撰寫）

512 token；回答（說明與程式碼）全部是新寫的，提示裡除了幾個選項名稱之外沒有可照抄的內容，對投機草稿是困難的情況。WHIRL：8060S 單次，R9700 為 3 次中位數。llama.cpp：只取第一次，因為它的 n-gram 草稿會學會重複的提示。

<!-- gen:chat.zh -->
| 顯示卡 · 模型 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 8060S · Swift 27B | 36.3 | **34.2** | 42.1 | 27.6 (MTP+n-gram) | 28.3 (MTP+n-gram) | +24% | +21% |
| 8060S · Ornith 35B-A3B | 94.5 | **106.0** | 102.8 | 70.6 (一般解碼) | 72.9 (一般解碼) | +50% | +45% |
| R9700 · Swift 27B | 83.4 | **109.6** | 130.4 | 57.8 (MTP+n-gram) | 57.6 (MTP+n-gram) | +90% | +90% |
| R9700 · Ornith 35B-A3B | 191.2 | **241.5** | 219.0 | 118.9 (一般解碼) | 115.1 (一般解碼) | +103% | +110% |

8060S 寫程式對話：WHIRL 每個模式單次量測。草稿被接受多少取決於各模式寫出的文字，所以這次 Swift 27B 的 precise（36.3）高於 balance（34.2）。
<!-- /gen -->

### WHIRL 領先不多的地方

<!-- gen:small.zh -->
- **領先最少：8060S 的稠密模型 decode。** Swift 27B 寫程式對話（全新撰寫）：34.2 對 llama.cpp Vulkan 28.3 tok/s（+21%），對 ROCm 27.6（+24%）；2k token 文件之後：31.2 對 llama.cpp Vulkan 25.1 tok/s（+24%）；31.2 對 llama.cpp ROCm 24.4 tok/s（+28%）。那裡的稠密 decode 受記憶體頻寬限制，領先來自投機解碼。
<!-- /gen -->
- **Ornith 的 decode 是各自出廠設定的對比**：WHIRL 用 MTP 草稿，llama.cpp 不用投機解碼（它的 MTP 在這個模型上較慢）。
- **不用投機的一般解碼**在每個引擎都受記憶體頻寬限制；WHIRL 大部分的 decode 領先來自 MTP + n-gram 投機解碼。
- **fast 不一定比 balance 快**（見[三種模式](#三種模式速度與精度)），**precise 要付出速度**，R9700 的 prefill 最明顯。

## 0.2.0 新功能

- **支援 Radeon 8060S（Ryzen AI Max+ 395）並已調校**：兩個模型在每個量過的提示長度都比 llama.cpp ROCm 與 Vulkan 快（見上表）。8060S 也能圖片輸入，RAM 與 SSD 快取層預設開啟（Windows 上 BIOS 的 VRAM carve-out 與作業系統 RAM 是兩個獨立的池，所以 128k token 的工作階段會從 RAM 或 SSD 還原，不必重新 prefill）。
- **三種數值模式**：precise（逐位元無損）、balance（近乎無損，預設）、fast（自選，損失受關卡管控）。見[上方](#三種模式速度與精度)。
- **我們的 MXFP4 模型**：Swift-1.5 27B MXFP4 與 Ornith-1.5-35B-A3B MXFP4，為 WHIRL 量化（[原因](#為什麼-whirl-跑這些-mxfp4-檔特別快)）。
- **DFlash2 草稿，自選**（`--draft dflash --draft-model Qwen3.8-27B-DFlash2-Q8_0.gguf`，草稿模型在 Hugging Face `incoai/Qwen3.8-27B-DFlash2-GGUF`，不隨附）：一個使用者、短或中等上下文、可預測的輸出時比 MTP 快；4 個使用者、長上下文與以中文為主的輸出時較慢；約佔 1.9 GiB VRAM。greedy 輸出與同模式的單純 greedy 解碼相同。
- **SSD 快取寫入大幅減少**：預設只在 server 停止時（或閒置約 5 分鐘、或 RAM 必須淘汰時）寫入被重複使用過的工作階段；只用一次的 prompt 永遠不寫。快取在磁碟上永遠不超過 `--kv-ssd-gb`（預設 64 GiB），7 天沒用的檔案會刪除。
- **對同一份長文件問第二個問題時沿用它**：server 保留尾端檢查點，只需 prefill 新問題（約 0.3k–1.3k token），不必重跑整份文件。
- **CLI 與 server 對同一個 prompt 輸出逐位元相同**（相同的 prefill 排程；有兩個記載的例外，見 [kv-and-caching.md §7](guide/zh-TW/kv-and-caching.md)）。

完整清單：[CHANGELOG.md](../CHANGELOG.md)。

## Windows 安全性提示

`whirl.exe` 與 `whirl-server.exe` 沒有程式碼簽章。剛下載的檔案可能讓 Windows SmartScreen 顯示「Windows 已保護您的電腦」：按 **其他資訊 → 仍要執行**，或在解壓縮前先對 zip 執行 `Unblock-File`（只對從官方發布頁下載、且已核對 SHA-256 的 zip 這樣做）。**智慧型應用程式控制**設為*開啟*時，Windows 可能直接封鎖程式，而且沒有「仍要執行」選項。詳見 [Windows 安全性提示](windows_security_zh-TW.md)。

## 文件

| | 繁體中文 | English |
|---|---|---|
| 快速上手 | [quickstart_zh-TW.md](quickstart_zh-TW.md) | [quickstart.md](quickstart.md) |
| 情境食譜：接上用戶端、長時間 agent 工作階段、長上下文、多使用者、圖片、看懂 log、疑難排解 | [recipes_zh-TW.md](recipes_zh-TW.md) | [recipes.md](recipes.md) |
| 使用參考（所有指令、選項、數值模式、環境變數、結束代碼） | [usage.md](guide/zh-TW/usage.md) | [usage.md](usage.md) |
| 伺服器（API、取樣、工具呼叫、批次、decode 保底、log） | [server.md](guide/zh-TW/server.md) | [server.md](guide/en/server.md) |
| 與 llama.cpp 的效能比較 | [benchmarks.md](guide/zh-TW/benchmarks.md) | [benchmarks.md](benchmarks.md) |
| Windows 安全性提示 | [windows_security_zh-TW.md](windows_security_zh-TW.md) | [windows_security.md](windows_security.md) |
| 進階：從原始碼建置 | [building_zh-TW.md](building_zh-TW.md) | [building.md](building.md) |
| 指南首頁 | [index.md](guide/zh-TW/index.md) | [index.md](guide/en/index.md) |
| 架構 | [architecture.md](guide/zh-TW/architecture.md) | [architecture.md](guide/en/architecture.md) |
| Kernel | [kernels.md](guide/zh-TW/kernels.md) | [kernels.md](guide/en/kernels.md) |
| 投機解碼（MTP + n-gram、DFlash2） | [speculative-decoding.md](guide/zh-TW/speculative-decoding.md) | [speculative-decoding.md](guide/en/speculative-decoding.md) |
| KV 快取、前綴快取、分層快取 | [kv-and-caching.md](guide/zh-TW/kv-and-caching.md) | [kv-and-caching.md](guide/en/kv-and-caching.md) |
| 量化與模型檔 | [quantization.md](guide/zh-TW/quantization.md) | [quantization.md](guide/en/quantization.md) |
| 圖片輸入 | [vision.md](guide/zh-TW/vision.md) | [vision.md](guide/en/vision.md) |
| Windows 上的 HIP | [windows-hip.md](guide/zh-TW/windows-hip.md) | [windows-hip.md](guide/en/windows-hip.md) |
| 量測方法 | [benchmarking.md](guide/zh-TW/benchmarking.md) | [benchmarking.md](guide/en/benchmarking.md) |
| **踩坑全集**（每條都有：症狀 → 根本原因 → 修正 → 檢查） | [pitfalls.md](guide/zh-TW/pitfalls.md) | [pitfalls.md](guide/en/pitfalls.md) |

## 設計理念

1. **專為 Windows 上的 AMD GPU 打造**：原生 Windows 的 C++／HIP 推論引擎，針對 RDNA 架構逐一手寫、調校 kernel。不需要 WSL、Docker 或 Linux 虛擬機，裝好 AMD 驅動就能跑。
2. **完整開源，包含每一支 kernel**：從 host 端排程到 GPU kernel，全部原始碼以 Apache-2.0 公開，沒有封閉的二進位元件。歡迎回報問題、提供量測數據，或直接貢獻程式碼。
3. **精度優先於速度**：在 precise 與 balance 模式下，所有加速手段（投機解碼、多人合批、前綴快取、快取還原）輸出的 token 都和該模式的單純 greedy 解碼相同（balance 下對話的下一回合與冷算數值等價、非逐位元，回覆可能不同），有常駐測試把關。任何損失更多的做法都是自選（`--fast`），並受 KL 與準確度門檻管控。
4. **少數模型，做到極致**：針對特定模型與硬體，把效能推到接近硬體極限，支援的架構一個一個加。目前支援 `qwen35`（稠密）與 `qwen35moe`（混合專家）。
5. **數字透明、可驗證**：所有數字都是實測，量測方法、指令和提示全部公開，包括 WHIRL 領先不多的地方。
6. **為真實工作流程設計**：以 coding agent 和長上下文為核心場景：長 prompt 快速處理、多人同時使用時主對話不卡頓、VRAM → RAM → SSD 三層快取，server 重啟後對話也能接回來。

WHIRL 依循 [NInfer](https://github.com/Neroued/ninfer)（Apache-2.0）的設計理念——為選定的模型與選定的 GPU 打造，盡可能逼近硬體極限；[zynfer](https://github.com/thanos/zynfer) 把同樣的想法帶到了 RDNA 4。

## 開發與 AI 協助

WHIRL 是在專案擁有者（[@tsaipifong](https://github.com/tsaipifong)）的指導下，由 Anthropic 的 AI 模型 **Claude Opus 5.5** 參與設計、實作、最佳化與量測。Anthropic 與本專案沒有隸屬關係，也不為本專案背書。

## 授權與致謝

Apache License 2.0；見 [LICENSE](../LICENSE) 與 [NOTICE](../NOTICE)。第三方素材列在 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)，每個原始檔的來源列在 [PROVENANCE.md](../PROVENANCE.md)。

- [NInfer](https://github.com/Neroued/ninfer)——WHIRL 依循的設計理念（只借理念）。
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)——我們量測時的比較基準，也是 GGUF 與量化格式相容性的參考；少數改寫的部分（tokenizer 預切詞 / BPE 迴圈、查表、圖片前處理）採 MIT 授權，列在 THIRD_PARTY_NOTICES.md。
- [gufo](https://github.com/gufo-org/gufo) 與 [r9700-stack](https://github.com/bkvargyas/r9700-stack)——量測方法與最佳化想法。
- 模型作者：Qwen 團隊（Qwen3.8）、[ornith-ai](https://huggingface.co/ornith-ai)（Ornith-1.5），以及 [ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)（Swift-1.5，我們的 MXFP4 檔就是由它量化而來）。
