[English](quickstart.md) | **繁體中文**

# 快速上手

從下載到跑起 OpenAI 相容伺服器，只要幾分鐘。以下指令都在 PowerShell 執行。

## 1. 確認需求

| | |
|---|---|
| GPU | **AMD Radeon AI PRO R9700**（RDNA 4，gfx1201）。單張 GPU；此建置不支援其他 GPU |
| 作業系統 | Windows 11，64 位元 |
| 驅動程式 | **AMD Software: Adrenalin Edition 26.8.1**（驅動程式 32.0.31041.1004）或更新版 |
| 其他都不需要 | 不用 HIP SDK、不用 ROCm、不用 Visual C++ 可轉散發套件、不用 Python、不用 WSL |

驅動程式會安裝 WHIRL 用到的兩個 AMD runtime 檔（`amdhip64_7.dll`、`amd_comgr_3.dll`）。

## 2. 下載

1. 從 [WHIRL 發布頁](https://github.com/tsaipifong/whirl-llm/releases)下載 `whirl-0.1.0-windows-x64.zip`，並核對
   旁邊公布的 SHA-256：`Get-FileHash .\whirl-0.1.0-windows-x64.zip -Algorithm SHA256`。
2. 一個支援的 GGUF 模型（清單見 [README](README_zh-TW.md#支援的模型)）。最快的選擇是我們自己量化的 MXFP4
   [`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF)；
   以下指令假設它存在 `C:\models\`。

## 3. 解壓縮並執行

在「下載」資料夾開啟 PowerShell，貼上：

```powershell
Unblock-File .\whirl-0.1.0-windows-x64.zip
Expand-Archive .\whirl-0.1.0-windows-x64.zip -DestinationPath C:\whirl
cd C:\whirl\whirl-0.1.0-windows-x64
.\whirl.exe devices
.\whirl.exe chat C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf "用兩句話說明 TCP 慢啟動（slow start）。" --max-tokens 400
```

`whirl devices` 應該會列出 `AMD Radeon AI PRO R9700 (gfx1201)`。（`Unblock-File` 會移除「從網路下載」的標記，
Windows 就不會對每個檔案都詢問一次；見 [Windows 安全性提示](windows_security_zh-TW.md)。）

**第一次使用某個模型時，會先為 GPU 調校一次 kernel**：MXFP4 模型只要幾秒，27B Q4_K_M 模型約 1.5 分鐘
（畫面會有提示）。結果快取在 `%LOCALAPPDATA%\whirl`，之後執行會直接開始。

## 4. 啟動伺服器

```powershell
.\whirl-server.exe C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf --port 8080
```

在另一個 PowerShell 視窗：

```powershell
$body = '{"messages":[{"role":"user","content":"用繁體中文寫一首關於 GPU 的短詩。"}],"max_tokens":200}'
Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) |
  ForEach-Object { $_.choices[0].message.content }
```

任何 OpenAI 相容的用戶端（Open WebUI、Continue、Cline、`openai` Python 套件等）都能用：base URL 填
`http://127.0.0.1:8080/v1`，API key 隨意。除非加上 `--host 0.0.0.0`，伺服器只接受這台電腦的連線；它沒有
身分驗證。

按 **Ctrl+C** 停止：執行中的請求會跑完，快取的 session 會寫入 SSD 層（最多 10 秒），下次啟動就能還原。

## 5. 出問題時

WHIRL 會說明發生了什麼、該怎麼做，並以代碼結束：

| 代碼 | 意義 | 試試看 |
|---|---|---|
| 2 | 指令列有誤 | `.\whirl.exe chat --help` |
| 3 | GPU / 驅動程式問題 | 安裝 Adrenalin 26.8.1 或更新版；檢查 `.\whirl.exe devices` |
| 4 | 模型檔問題 | 確認檔案存在、是 GGUF，而且架構是 `qwen35` / `qwen35moe` |
| 5 | GPU 記憶體不足 | 較小的 `--ctx`，或 `$env:WHIRL_KV = "q8v"` |
| 6 | 伺服器 port 已被占用 | `--port 8081` |

若 Windows 顯示「Windows 已保護您的電腦」或程式被封鎖，請見 [Windows 安全性提示](windows_security_zh-TW.md)。

## 接下來

- 照做就會的情境食譜（接上 agent、長工作階段、長上下文、圖片、log 解讀）：[recipes_zh-TW.md](recipes_zh-TW.md)
- 每個選項與環境變數：[使用參考](guide/zh-TW/usage.md)
- 伺服器細節（API、取樣、工具呼叫、快取）：[server.md](guide/zh-TW/server.md)
- 該用哪個 GGUF：[quantization.md](guide/zh-TW/quantization.md#rules)
- 實測效能：[benchmarks.md](guide/zh-TW/benchmarks.md)
