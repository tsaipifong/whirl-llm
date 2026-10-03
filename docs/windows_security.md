**English** | [繁體中文](windows_security_zh-TW.md)

# Windows security prompts (SmartScreen, Smart App Control)

WHIRL's `whirl.exe` and `whirl-server.exe` are not code-signed (the project
does not buy a code-signing certificate). They carry version information and
an application manifest, run as the current user (no administrator rights, no
UAC prompt), and need nothing but the AMD graphics driver. Windows may still
warn about them because they are new, unsigned downloads.

## SmartScreen ("Windows protected your PC")

Windows marks files downloaded from the internet. When such an unsigned
program is started from Explorer, Microsoft Defender SmartScreen can show
**"Windows protected your PC"**. To run it anyway:

1. Click **More info**.
2. Check that the file name is `whirl.exe` / `whirl-server.exe`, then click **Run anyway**.

To avoid the prompt for every file, unblock the zip **before** extracting it:
right-click the zip → **Properties** → tick **Unblock** → **OK**, or in PowerShell:

```powershell
Unblock-File .\whirl-0.1.0-windows-x64.zip
```

Only do this for a zip you downloaded from the official WHIRL release page.
Compare its SHA-256 with the value published next to it:
`Get-FileHash .\whirl-0.1.0-windows-x64.zip -Algorithm SHA256`.

## Smart App Control

Smart App Control (Windows 11) blocks unsigned programs that Microsoft's cloud
service does not yet know. When it is **On**, Windows may block `whirl.exe`
with no "Run anyway" button; unblocking the zip does not help.

Check the state: **Windows Security → App & browser control → Smart App Control
settings** (On / Evaluation / Off), or in PowerShell:

```powershell
Get-ItemPropertyValue 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' -Name VerifiedAndReputablePolicyState
# 0 = Off, 1 = On, 2 = Evaluation
```

- **Off**: Smart App Control does not affect WHIRL.
- **Evaluation**: Windows is still deciding; it can block apps it considers untrusted.
- **On**: WHIRL can be blocked. Running it requires turning Smart App Control off
  in the same settings page. On many Windows 11 versions it cannot be turned back
  on afterwards without resetting Windows, so decide deliberately; check
  Microsoft's current documentation for your Windows version.

## What the executables do and do not do

- They run as the current user and never ask for administrator rights.
- They write only to `%LOCALAPPDATA%\whirl` (kernel tuning cache, server log, SSD tier of the
  server's prefix cache); nothing is installed system-wide.
- `whirl-server.exe` listens on `127.0.0.1` (this computer only) unless you pass `--host`. With
  `--host 0.0.0.0` Windows Defender Firewall may ask whether to allow network access; allow it only
  for networks you trust, because the server has no authentication.
- They make no outgoing internet connections (the server never fetches `http(s)` image URLs).
