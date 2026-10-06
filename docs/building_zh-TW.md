[English](building.md) | **繁體中文**

# 從原始碼建置

只有想修改 WHIRL 時才需要；發行版 zip 只要顯示卡驅動程式就能執行（[快速上手](quickstart_zh-TW.md)）。

## 需求

| 工具 | 我們使用的版本 |
|---|---|
| Visual Studio 2022 Build Tools（MSVC，C++20 工作負載） | 17.14（MSVC 19.44） |
| CMake | ≥ 3.24（我們用 4.4） |
| Ninja | 1.13 |
| AMD HIP SDK for Windows | 7.2（安裝程式會設定 `HIP_PATH`；預設 `C:\Program Files\AMD\ROCm\7.2`） |

建置、執行、測試都不需要 Python 或其他語言；儲存庫裡的工具與測試全部是 C++。

## 取得原始碼

```bat
git clone https://github.com/tsaipifong/whirl-llm.git
cd whirl-llm
```

## 建置

```bat
build.bat Release
```

`build.bat` 視需要呼叫 `vcvars64.bat`，以 Ninja 把 CMake 設定到 `build\Release`，並以低 CPU 優先權建置。它用 `vswhere` 尋找 `vcvars64.bat`，所以任何裝有 C++ x64 工具的 Visual Studio 2022 版本（Community、Professional、Enterprise 或 Build Tools）都可以；從 Developer Command Prompt 執行則略過尋找。尋找只接受 Visual Studio 2022（17.x）：HIP clang 搭配 Visual Studio 2026 的 STL 尚未驗證，所以兩者都裝的機器會使用 2022。
`build\Release\` 的產出：

| 檔案 | 用途 |
|---|---|
| `whirl.exe`、`whirl-server.exe` | 發行版的兩支程式 |
| `whirl-tool.exe` | GGUF 檢視、tokenizer、聊天樣板與裝置工具 |
| `whirl-tests.exe`、`whirl-model-tests.exe`、`whirl-server-tests.exe`、`whirl-tier-tests.exe`、`whirl-vision-tests.exe` | host 端測試（不用 GPU） |
| `whirl-kernel-test.exe` | GPU kernel 對 C++ 參考實作的比對（需要 R9700 與模型檔） |
| `whirl-server-gate.exe`、`whirl-parity.exe` | 伺服器端到端關卡，以及對外部參考的一致性檢查 |

host 端由 MSVC 以靜態 C++ runtime（`/MT`）編譯。device 端由 HIP SDK 的 clang 為每種 GPU 架構各編一個 code
object，經 `tools/bin2c` 轉成位元組陣列內嵌在執行檔裡。`amdhip64_7.dll` 採延遲載入，所以沒有安裝 HIP SDK 也
能啟動執行檔。預設會編兩套 kernel：gfx1201（R9700）與 gfx1151（Radeon 8060S）；執行時依所選 GPU 的
`gcnArchName` 載入對應的 code object。只編一套比較快（專心改某一張 GPU 的 kernel 時很方便）：

```bat
build.bat Release -DWHIRL_GPU_ARCHS=gfx1201
build.bat Release -DWHIRL_GPU_ARCHS=gfx1151
```

每個 code object 只依賴自己的原始檔（gfx1201 是 `kernels/*.hip`，gfx1151 是 `kernels/gfx1151/*.hip`）。
`whirl-kernel-test` 一次測一張 GPU：`--device 8060s`（或 `WHIRL_DEVICE=8060s`）測 gfx1151 那套；code object
沒有的 kernel，其檢查會列為 skipped。

在 Windows 上編譯與載入 device code 的更多細節：[windows-hip.md](guide/zh-TW/windows-hip.md#build)。

## 測試

```bat
build\Release\whirl-tests.exe
build\Release\whirl-model-tests.exe
build\Release\whirl-server-tests.exe --gguf MODEL.gguf
build\Release\whirl-tier-tests.exe
build\Release\whirl-kernel-test.exe
build\Release\whirl.exe selftest MODEL.gguf
build\Release\whirl.exe seqtest MODEL.gguf
```

測試程式裡沒有寫死任何特定機器的路徑；路徑一律來自命令列或下列環境變數：

| 變數 | 使用者 | 預設 |
|---|---|---|
| `WHIRL_TEST_GGUF` | `whirl-server-tests`（只用其 tokenizer，模型本身是 mock）、`whirl-kernel-test`（`--q4` 的備援） | 無：`whirl-server-tests` 需要 `--gguf` 或此變數 |
| `WHIRL_TEST_Q4`、`WHIRL_TEST_MX`、`WHIRL_TEST_MOE`、`WHIRL_TEST_MOEMX` | `whirl-kernel-test`（等同 `--q4` / `--mx` / `--moe` / `--moemx`：Qwen3.8-27B Q4_K_M、Qwen3.8-27B MXFP4、Ornith-1.5-35B-A3B Q4_K_M、Ornith-1.5-35B-A3B MXFP4） | 未設定：該家族改用合成資料，需要模型張量的檢查標為略過 |
| `WHIRL_TEST_TMP` | `whirl-server-tests`、`whirl-tier-tests`（SSD 層的暫存目錄） | `%TEMP%\whirl-tests` |
| `WHIRL_GATE_TEXT_ROOT` | `whirl-server-gate`（測試文字：llama.cpp 原始碼目錄的根，只當資料讀取） | 無（必填） |
| `WHIRL_GATE_WORK` | `whirl-server-gate`（等同 `--work`：記錄檔、SSD 目錄） | `%TEMP%\whirl-tests\gate` |
| `WHIRL_GATE_IMAGES` | `whirl-server-gate` 的 `vis` suite（等同 `--images`：`shapes.png`、`dialog.png`、`s1080.png`） | 無 |
| `WHIRL_GPU_LOCK` | `whirl-server-gate`（伺服器執行期間持有的鎖檔，避免兩個 gate 同時使用 GPU；`--no-lock` 可關閉） | `%TEMP%\whirl-gpu.lock` |

我們每次修改都會跑的正確性關卡見 [benchmarking.md](guide/zh-TW/benchmarking.md#gates)。

## 打包

```powershell
powershell -ExecutionPolicy Bypass -File tools\package_release.ps1
```

先檢查兩支執行檔回報的版本與專案版本相同，而且只匯入 Windows 系統 DLL 與驅動程式的 `amdhip64_7.dll`，再把
`whirl-<版本>-windows-x64.zip` 與其 SHA-256 寫到輸出目錄（`-OutDir`）。不會上傳任何東西。
