# The Windows launcher (`src/launcher`, 0.2.6 draft)

Status: on branch `feat026-launcher`, not released (feature freeze until the 0.2.5 modularization is done; owner
2026-10-26: 0.2.6 code starts on branches). Owner spec: internal_docs/PLAN_0.2.1.md "Windows launcher". Prior art and
the llama.cpp comparison: progress/feat026-launcher.PRIOR_ART.md.

## For users (draft for README / docs/cli.md at the 0.2.6 release)

> **Double-click `whirl.exe`.** The first time, WHIRL asks five questions, each with a recommended answer you take
> with Enter:
>
> 1. **GPU**: every AMD GPU with its PCI location (`pci:71:0` = Task Manager > Performance > GPU > Physical location
>    "PCI bus 71, device 0"). Recommended: the one with the most GPU memory (the same as `whirl devices` [default]).
> 2. **Model file**: `.gguf` files found next to whirl.exe, in its `models` folder, `C:\models` and Downloads, the
>    Swift-1.5 27B MXFP4 file first; or type / drag in a path.
> 3. **Mode**: balance (recommended), precise or fast.
> 4. **Context length**: the longest request in tokens (`64k` works). The recommendation is 80% of what the
>    estimate says fits next to the model (at most 131072); the maximum is shown. The server still lowers a value
>    that does not fit when it starts and says so in the log.
> 5. **Port**: 8080, or the next free one.
>
> The answers are saved in `%LOCALAPPDATA%\whirl\launcher.ini`, WHIRL prints the same server as a command line
> (for scripts) and starts it. The window shows the server log; minimise it and WHIRL keeps running in the
> notification area: right-click the WHIRL icon for **Open chat** (a chat window with the running server),
> **Show log**, **Settings...** and **Quit WHIRL**. Later double-clicks start at once with the saved settings;
> `whirl --setup` asks again.
>
> Scripts are not affected: `whirl` with arguments, or started from a command prompt, a script, a pipe or a
> service, never asks a question and never goes to the tray. `WHIRL_NO_LAUNCHER=1` turns the launcher off.

## When the launcher runs

`launcher::decideStart` (src/launcher/launcher.h), called first in `wmain` (src/cli/main.cpp):

| Start | Facts | Result |
|---|---|---|
| Explorer double-click, Start menu, shortcut, Win+R | no args, stdin + stdout = console, console has 1 process | launcher |
| `whirl` typed in cmd / PowerShell / Windows Terminal, a .bat or .ps1 | console has >= 2 processes | usage text, exit 2 (as 0.2.5) |
| `whirl < file`, `whirl \| more`, `whirl > x.txt` | stdin or stdout not a console | usage text, exit 2 |
| service, Task Scheduler without a window, DETACHED_PROCESS | no console (0 processes) | usage text, exit 2 |
| `WHIRL_NO_LAUNCHER=1` | env | usage text, exit 2 |
| any argument | n_args > 0 | the command, unchanged |
| `whirl --setup` | stdin + stdout = console (a command prompt is fine) | setup, then the launcher; otherwise exit 2 with a message (never a prompt) |
| `whirl --whirl-launcher-chat HOST:PORT` | the tray's chat window (hidden argument) | console chat client |

Pinned by `whirl-launcher-tests` (exhaustive over the facts: exactly one combination starts the launcher) and by
`tests/launcher/scripted_start_check.ps1` on the built whirl.exe (each scripted start must return within 20 s with
exit 2 and the usage text, also with Enter keys waiting on stdin).

## Shape

```
whirl.exe wmain ──(no args, double-click)──> launcher::launcherMain
   │                                            │ settings: %LOCALAPPDATA%\whirl\launcher.ini (launcher_logic: parse / serialize)
   │                                            │ setup: GPU list <- child `whirl.exe --whirl-gpu-probe` (gpusel::runChild, parseRecords)
   │                                            │        model scan (std::filesystem), context estimate (qwen35::estimateLoad,
   │                                            │        numerics::chooseKv; GGUF header only, no HIP), port check (app::requirePortFree)
   │                                            │ child: `whirl.exe serve MODEL --device pci:B:D --MODE [--ctx-per-slot N] --port P`
   │                                            │        same console (= the log panel), own process group
   │                                            └ tray (platform::shell::runTray): hidden top-level window + Shell_NotifyIcon,
   │                                              500 ms tick: child exited? console minimised -> hide
   └──(any argument)──> the existing commands, unchanged
```

- **No GPU context in the launcher.** Like llama.cpp's router (`tools/server/server.cpp:150`, "router server never
  loads a model and must not touch the GPU"), the launcher process never loads the HIP runtime: the GPU list comes
  from the existing probe child, the model runs in the `whirl serve` child. Two large processes on one GPU demote
  each other (docs/guide/en/windows-hip.md 4.4); the launcher holds no GPU memory at all.
- **The server is unchanged.** It is `whirl serve` with ordinary arguments; the printed command line reproduces it
  exactly. Quit = Ctrl+Break to its process group: the server's existing graceful shutdown (finish running requests,
  flush the SSD tier, server_main.cpp onConsoleCtrl). Killed only if it has not exited 180 s later.
- **Ctrl+C / closing the window** reach the server through the shared console as before; the launcher ignores
  Ctrl+C itself and waits up to 4.5 s for the child on a console close.
- **Open chat** starts `whirl.exe --whirl-launcher-chat 127.0.0.1:PORT` in a new console: a streaming
  /v1/chat/completions client (WinHTTP, platform/http_client). The WHIRL server has no web page; adding one would
  change `whirl serve`, so it is a separate later item (llama.cpp's built-in web UI is the better chat today).
- **DLLs.** whirl.exe delay-loads WINHTTP, SHELL32, ole32 and USER32 (cmake/launcher.cmake): every other command
  keeps the 0.2.5 static DLL set (dumpbin /dependents: WS2_32, dxgi, pdh, KERNEL32, WINMM, the same as a whirl.exe
  built from arch025-next 1a5a547), and the launcher's DLLs load only when the launcher runs.
- **OS calls** are in `src/platform/shell_win.cpp` (console facts and UTF-16 input, child processes, tray, message
  box, ShellExecute) and `src/platform/http_client_win.cpp` (WinHTTP); `src/launcher` has none. Nothing outside
  `whirl.exe` references them, so the linker leaves every other executable as it was.

## Known limits (draft)

- Windows Terminal as the default terminal: minimising hides the window only where Terminal forwards
  ShowWindow of its pseudo console window (Terminal PR #12515); otherwise the window stays on the taskbar and the
  tray still works (manual test M6).
- The context estimate uses the GGUF header and the server's own estimate functions with a margin (10% of the GPU
  memory, at least 1.5 GiB, plus the recurrent state of 4 slots); it is not a measurement. The server's startup fit
  is the authority.
- `whirl --help` does not list `--setup` yet (the help text is unchanged on this branch; add the line at release).
- The setup recommends; it does not remember the previous answers as defaults (owner spec: recommended values).

## Tests

| Test | What | Where |
|---|---|---|
| `whirl-launcher-tests` | decideStart / setupAllowed (exhaustive), settings round trip and errors (BOM, CRLF, non-ASCII, unknown keys), answers, GPU / model / mode / context / port recommendations, server arguments and Windows quoting, tray menu, chat request and SSE parsing | tests/launcher/launcher_tests.cpp |
| `scripted_start_check.ps1` | the built whirl.exe: no-args scripted starts, `--setup` from a script, `--help`, an unknown command | tests/launcher/ |
| `chat_e2e_check.py` | the built whirl.exe `--whirl-launcher-chat` against a mock OpenAI server on 127.0.0.1: request body, streamed answer, history, `/new`, `/exit`, an HTTP 500, Chinese text both ways | tests/launcher/ |
| manual M1-M16 | double-click, tray, chat, quit, Explorer restart, Windows Terminal | tests/launcher/manual_test.md |
