[English](windows_security.md) | **繁體中文**

# Windows 安全性提示（SmartScreen、智慧型應用程式控制）

WHIRL 的 `whirl.exe` 與 `whirl-server.exe` 沒有程式碼簽章（本專案不購買程式碼簽章憑證）。
執行檔內含版本資訊與應用程式資訊清單，以目前使用者身分執行（不需要系統管理員權限、不會跳出
UAC），執行時只需要 AMD 顯示卡驅動程式。由於它們是新發布、未簽章的下載檔，Windows 仍可能
發出警告。

## SmartScreen（「Windows 已保護您的電腦」）

Windows 會標記從網路下載的檔案。從檔案總管啟動這類未簽章程式時，Microsoft Defender
SmartScreen 可能顯示 **「Windows 已保護您的電腦」**。若要照常執行：

1. 按 **其他資訊**。
2. 確認檔名是 `whirl.exe` / `whirl-server.exe`，再按 **仍要執行**。

若不想每個檔案都被詢問，請在**解壓縮之前**先解除封鎖 zip：在 zip 上按右鍵 → **內容** →
勾選 **解除封鎖** → **確定**；或在 PowerShell 執行：

```powershell
Unblock-File .\whirl-0.1.0-windows-x64.zip
```

只對從 WHIRL 官方發布頁下載的 zip 這樣做，並核對旁邊公布的 SHA-256：
`Get-FileHash .\whirl-0.1.0-windows-x64.zip -Algorithm SHA256`。

## 智慧型應用程式控制（Smart App Control）

Windows 11 的智慧型應用程式控制會封鎖微軟雲端服務尚未認識的未簽章程式。它設為 **開啟**
時，Windows 可能直接封鎖 `whirl.exe`，而且沒有「仍要執行」按鈕；解除封鎖 zip 也沒有用。

查看狀態：**Windows 安全性 → 應用程式與瀏覽器控制 → 智慧型應用程式控制設定**（開啟 / 評估 /
關閉），或在 PowerShell 執行：

```powershell
Get-ItemPropertyValue 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' -Name VerifiedAndReputablePolicyState
# 0 = 關閉, 1 = 開啟, 2 = 評估
```

- **關閉**：智慧型應用程式控制不影響 WHIRL。
- **評估**：Windows 還在判斷中，仍可能封鎖它認為不可信的程式。
- **開啟**：WHIRL 可能被封鎖。要執行它，必須在同一個設定頁把智慧型應用程式控制關閉。
  在許多 Windows 11 版本上，關閉後若不重設 Windows 就無法再開啟，請想清楚再決定，並查閱
  微軟針對你的 Windows 版本的最新說明。

## 執行檔會做與不會做的事

- 以目前使用者身分執行，不會要求系統管理員權限。
- 只寫入 `%LOCALAPPDATA%\whirl`（kernel 調校快取、伺服器 log、伺服器前綴快取的 SSD 層）；不會安裝任何系統
  層級的元件。
- 除非指定 `--host`，`whirl-server.exe` 只監聽 `127.0.0.1`（這台電腦）。使用 `--host 0.0.0.0` 時，Windows
  Defender 防火牆可能詢問是否允許網路存取；伺服器沒有身分驗證，請只在信任的網路上允許。
- 不會主動連線到網際網路（伺服器不會抓取 `http(s)` 圖片網址）。
