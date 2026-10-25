# Windows launcher: manual test (owner, before 0.2.6)

Branch feat026-launcher. These steps need a desktop session and one GPU; the CPU parts are covered by
`whirl-launcher-tests.exe` (decision logic) and `tests\launcher\scripted_start_check.ps1` (scripted starts on the real
whirl.exe). Use ONE GPU at a time (the other one idle). Before the test, rename an existing
`%LOCALAPPDATA%\whirl\launcher.ini` (if any) so the first-run setup appears.

Record for each step: pass / fail, what you saw, the time.

| # | Step | Expected |
|---|---|---|
| M1 | Double-click `whirl.exe` in Explorer (the release folder) | A console opens with "WHIRL first-start setup"; a GPU list with `pci:<bus>:<device>` and "(recommended: most GPU memory)" on the GPU `whirl devices` marks `[default]` |
| M2 | Press Enter at each question | GPU = recommended; model = the Swift-1.5 27B MXFP4-A file if it is in the folder / `models\` / `C:\models` / Downloads; mode = balance; context = the recommended value (the line shows the estimate and the maximum); port = 8080 (or the next free one) |
| M3 | Read the screen after the last answer | "Saved to %LOCALAPPDATA%\whirl\launcher.ini"; the equivalent command line `"...\whirl.exe" serve "...gguf" --device pci:... --balance --ctx-per-slot N --port 8080`; then the server's own startup log in the same window, ending with the listening line |
| M4 | Minimise the window | The window disappears from the taskbar; a WHIRL icon is in the notification area; one notification "WHIRL is still running" |
| M5 | Right-click the icon | Menu: Open chat, Show log, Settings..., Quit WHIRL |
| M6 | Show log (also: double-click the icon) | The console window comes back with the server log. **Windows Terminal as the default terminal:** note whether M4 hid the window; if it stayed on the taskbar, write down the Terminal version (Settings > About) |
| M7 | Open chat; type a Chinese question about code (主要場景), Enter | A second window "WHIRL chat"; the answer streams; `/new` starts over; `/exit` closes it. No second model load in Task Manager (one whirl.exe with GPU memory) |
| M8 | Settings... | A message names the settings file and it opens in the editor; changes apply at the next start |
| M9 | Quit WHIRL | The log window shows "stopping the server..." then the server's shutdown lines; the icon goes away; the window closes by itself |
| M10 | Double-click `whirl.exe` again | No questions: "starting with the saved settings"; the server starts at once (same command line as M3) |
| M11 | In a command prompt: `whirl --setup` | The questions again (the same recommendations as on the first start; the saved model's folder is also searched); then the server starts in that window |
| M12 | In a command prompt: `whirl` (no arguments) | The usage text as in 0.2.5, exit code 2, no questions (`echo %ERRORLEVEL%`) |
| M13 | Close the console window with X while the server runs | The server logs "shutdown requested (console closed)"; the tray icon disappears |
| M14 | Restart Explorer (Task Manager > Windows Explorer > Restart) while minimised | The WHIRL icon comes back |
| M15 | Edit launcher.ini: `model=` a file that does not exist; double-click | "the saved settings are incomplete or the model file is gone; asking again" and the setup |
| M16 | Model path with Chinese characters (e.g. `D:\模型\Swift...gguf`) typed or dragged into the setup | Accepted, saved, the server loads it |

Automated before M1 (CPU, any time):

```
build\Release\whirl-launcher-tests.exe
pwsh -NoProfile -File tests\launcher\scripted_start_check.ps1
C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe -I tests\launcher\chat_e2e_check.py build\Release\whirl.exe
```
