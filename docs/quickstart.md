**English** | [繁體中文](quickstart_zh-TW.md)

# Quick start

From download to a running OpenAI-compatible server in a few minutes. All commands are for
PowerShell.

## 1. Check the requirements

| | |
|---|---|
| GPU | **AMD Radeon AI PRO R9700** (RDNA 4, gfx1201, 32 GB) or **Radeon 8060S** (RDNA 3.5, gfx1151, Ryzen AI Max+ 395). One GPU per process; other GPUs are not officially supported |
| OS | Windows 11, 64-bit |
| Driver | **AMD Software: Adrenalin Edition 26.8.1 (driver 32.0.31041.1004) or newer**, for both GPUs; older drivers are untested |
| Nothing else | no HIP SDK, no ROCm, no Visual C++ Redistributable to install, no Python, no WSL |

The driver installs the two AMD runtime files WHIRL uses (`amdhip64_7.dll`, `amd_comgr_3.dll`). The zip also carries AMD's `rocm_kpack.dll`: the GPU runtime of the Adrenalin Edition for ROCm 10.1 driver needs it and that driver does not install it, so keep it next to `whirl.exe`, with the three Microsoft Visual C++ runtime DLLs it uses (`msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`), which are in the zip for the same reason.

## 2. Download

1. `whirl-0.2.2-windows-x64.zip` from the [WHIRL releases page](https://github.com/tsaipifong/whirl-llm/releases).
   Compare its SHA-256 with the value published next to it:
   `Get-FileHash .\whirl-0.2.2-windows-x64.zip -Algorithm SHA256`.
2. A supported GGUF model (list in the [README](../README.md#supported-models)). The fastest choice is our own
   MXFP4 quantization
   [`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF);
   the commands below assume it is saved in `C:\models\`.

## 3. Unpack and run

Open PowerShell in your Downloads folder and paste:

```powershell
Unblock-File .\whirl-0.2.2-windows-x64.zip
Expand-Archive .\whirl-0.2.2-windows-x64.zip -DestinationPath C:\whirl
cd C:\whirl\whirl-0.2.2-windows-x64
.\whirl.exe devices
.\whirl.exe chat C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf "Explain TCP slow start in two sentences." --max-tokens 400
```

`whirl devices` should list `AMD Radeon AI PRO R9700 (gfx1201)` and/or the Radeon 8060S (gfx1151); with both present the R9700 is the default, `--device 8060s` selects the 8060S. (`Unblock-File` removes the
"downloaded from the internet" mark so that Windows does not ask about every file; see
[Windows security prompts](windows_security.md).)

**The first run with a model tunes the GPU kernels once**: on the R9700 a few seconds for our MXFP4 models,
longer on the Radeon 8060S (a message says so). The result is cached in
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

Stop it with **Ctrl+C**: running requests finish and sessions that were reused are written to the SSD tier
(the stop waits up to 60 seconds for this), so the next start can restore them.

## 5. If something goes wrong

WHIRL prints what happened and what to do, and exits with a code:

| Code | Meaning | Try |
|---|---|---|
| 2 | bad command line | `.\whirl.exe chat --help` |
| 3 | GPU / driver problem | install Adrenalin 26.8.1 or newer; check `.\whirl.exe devices` |
| 4 | model file problem | the file exists, is a GGUF, and its architecture is `qwen35` / `qwen35moe` |
| 5 | out of GPU memory | a smaller `--ctx` (server: also `--ctx-per-slot`, or fewer slots with `-np 1`); with `--precise` on a dense model, try the default mode (its KV cache is about half the size) |
| 6 | server port in use | `--port 8081` |

If Windows shows "Windows protected your PC", or the program is blocked, see
[Windows security prompts](windows_security.md).

## Next

- Step-by-step recipes (connect an agent, long sessions, long context, images, log lines): [recipes.md](recipes.md)
- Every option and environment variable: [usage reference](usage.md)
- Server details (API, sampling, tool calls, caching): [server.md](guide/en/server.md)
- Which GGUF to use: [quantization.md](guide/en/quantization.md#rules)
- Measured performance: [benchmarks.md](benchmarks.md)
