**English** | [繁體中文](quickstart_zh-TW.md)

# Quick start

From download to a running OpenAI-compatible server in a few minutes. All commands are for
PowerShell.

## 1. Check the requirements

| | |
|---|---|
| GPU | **AMD Radeon AI PRO R9700** (RDNA 4, gfx1201). One GPU; other GPUs are not supported by this build |
| OS | Windows 11, 64-bit |
| Driver | **AMD Software: Adrenalin Edition 26.8.1** (driver 32.0.31041.1004) or newer |
| Nothing else | no HIP SDK, no ROCm, no Visual C++ redistributable, no Python, no WSL |

The driver installs the two AMD runtime files WHIRL uses (`amdhip64_7.dll`, `amd_comgr_3.dll`).

## 2. Download

1. `whirl-0.1.0-windows-x64.zip` from the [WHIRL releases page](https://github.com/tsaipifong/whirl-llm/releases).
   Compare its SHA-256 with the value published next to it:
   `Get-FileHash .\whirl-0.1.0-windows-x64.zip -Algorithm SHA256`.
2. A supported GGUF model (list in the [README](../README.md#models)). The fastest choice is our own
   MXFP4 quantization
   [`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF);
   the commands below assume it is saved in `C:\models\`.

## 3. Unpack and run

Open PowerShell in your Downloads folder and paste:

```powershell
Unblock-File .\whirl-0.1.0-windows-x64.zip
Expand-Archive .\whirl-0.1.0-windows-x64.zip -DestinationPath C:\whirl
cd C:\whirl\whirl-0.1.0-windows-x64
.\whirl.exe devices
.\whirl.exe chat C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf "Explain TCP slow start in two sentences." --max-tokens 400
```

`whirl devices` should list `AMD Radeon AI PRO R9700 (gfx1201)`. (`Unblock-File` removes the
"downloaded from the internet" mark so that Windows does not ask about every file; see
[Windows security prompts](windows_security.md).)

**The first run with a model tunes the GPU kernels once**: a few seconds for MXFP4 models, about
1.5 minutes for a 27B Q4_K_M model (a message says so). The result is cached in
`%LOCALAPPDATA%\whirl`, so later runs start right away.

## 4. Start the server

```powershell
.\whirl-server.exe C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf --port 8080
```

From another PowerShell window:

```powershell
$body = '{"messages":[{"role":"user","content":"Write a haiku about GPUs."}],"max_tokens":200}'
Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body |
  ForEach-Object { $_.choices[0].message.content }
```

Any OpenAI-compatible client (Open WebUI, Continue, Cline, the `openai` Python package, …) works
with base URL `http://127.0.0.1:8080/v1` and any API key. The server listens only on this computer
unless you pass `--host 0.0.0.0`; it has no authentication.

Stop it with **Ctrl+C**: running requests finish and cached sessions are written to the SSD tier
(at most 10 seconds), so the next start can restore them.

## 5. If something goes wrong

WHIRL prints what happened and what to do, and exits with a code:

| Code | Meaning | Try |
|---|---|---|
| 2 | bad command line | `.\whirl.exe chat --help` |
| 3 | GPU / driver problem | install Adrenalin 26.8.1 or newer; check `.\whirl.exe devices` |
| 4 | model file problem | the file exists, is a GGUF, and its architecture is `qwen35` / `qwen35moe` |
| 5 | out of GPU memory | a smaller `--ctx`, or `$env:WHIRL_KV = "q8v"` |
| 6 | server port in use | `--port 8081` |

If Windows shows "Windows protected your PC", or the program is blocked, see
[Windows security prompts](windows_security.md).

## Next

- Every option and environment variable: [usage reference](usage.md)
- Server details (API, sampling, tool calls, caching): [server.md](guide/en/server.md)
- Which GGUF to use: [quantization.md](guide/en/quantization.md#rules)
- Measured performance: [benchmarks.md](benchmarks.md)
