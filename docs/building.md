**English** | [繁體中文](building_zh-TW.md)

# Building from source

Only needed if you want to change WHIRL; the release zip runs with just the graphics driver
([quick start](quickstart.md)).

## Requirements

| Tool | Version we use |
|---|---|
| Visual Studio 2022 Build Tools (MSVC, C++20 workload) | 17.14 (MSVC 19.44) |
| CMake | ≥ 3.24 (we use 4.4) |
| Ninja | 1.13 |
| AMD HIP SDK for Windows | 7.2 (its installer sets `HIP_PATH`; default `C:\Program Files\AMD\ROCm\7.2`) |

No Python or other language is needed to build, run or test; all tools and tests in the repository
are C++.

## Get the source

```bat
git clone https://github.com/tsaipifong/whirl-llm.git
cd whirl-llm
```

## Build

```bat
build.bat Release
```

`build.bat` calls `vcvars64.bat` if needed, configures CMake with Ninja into `build\Release` and
builds at low CPU priority. It finds `vcvars64.bat` with `vswhere`, so any Visual Studio 2022
edition with the C++ x64 tools works (Community, Professional, Enterprise or Build Tools); running
from a Developer Command Prompt skips the lookup. Outputs in `build\Release\`:

| File | What |
|---|---|
| `whirl.exe`, `whirl-server.exe` | the two programs of the release |
| `whirl-tool.exe` | GGUF inspection, tokenizer, chat template and device tools |
| `whirl-tests.exe`, `whirl-model-tests.exe`, `whirl-server-tests.exe`, `whirl-tier-tests.exe`, `whirl-vision-tests.exe` | host-side tests (no GPU) |
| `whirl-kernel-test.exe` | GPU kernels against C++ reference implementations (needs the R9700 and model files) |
| `whirl-server-gate.exe`, `whirl-parity.exe` | end-to-end server gates and parity checks against external references |

Host code is compiled by MSVC with a static C++ runtime (`/MT`). Device code is compiled by the
HIP SDK's clang into one code object per GPU architecture, turned into a byte array by
`tools/bin2c` and embedded in the executable. `amdhip64_7.dll` is delay-loaded, so the
executables start without the HIP SDK installed. Only the gfx1201 (R9700) kernel set exists at
the moment; `-DWHIRL_GPU_ARCHS=gfx1201` skips the gfx1151 placeholder object:

```bat
build.bat Release -DWHIRL_GPU_ARCHS=gfx1201
```

More on compiling and loading device code on Windows: [windows-hip.md](guide/en/windows-hip.md#build).

## Test

```bat
build\Release\whirl-tests.exe
build\Release\whirl-model-tests.exe
build\Release\whirl-server-tests.exe --gguf MODEL.gguf
build\Release\whirl-tier-tests.exe
build\Release\whirl-kernel-test.exe
build\Release\whirl.exe selftest MODEL.gguf
build\Release\whirl.exe seqtest MODEL.gguf
```

No test has a machine-specific path built in. Paths come from the command line or from these
environment variables:

| Variable | Used by | Default |
|---|---|---|
| `WHIRL_TEST_GGUF` | `whirl-server-tests` (tokenizer only; the model is mocked), `whirl-kernel-test` (fallback for `--q4`) | none: `whirl-server-tests` needs `--gguf` or this variable |
| `WHIRL_TEST_Q4`, `WHIRL_TEST_MX`, `WHIRL_TEST_MOE`, `WHIRL_TEST_MOEMX` | `whirl-kernel-test` (same as `--q4` / `--mx` / `--moe` / `--moemx`: Qwen3.8-27B Q4_K_M, Qwen3.8-27B MXFP4, Ornith-1.5-35B-A3B Q4_K_M, Ornith-1.5-35B-A3B MXFP4) | unset: that family runs on synthetic data and reports the model-tensor checks as skipped |
| `WHIRL_TEST_TMP` | `whirl-server-tests`, `whirl-tier-tests` (scratch directories for the SSD tier) | `%TEMP%\whirl-tests` |
| `WHIRL_GATE_TEXT_ROOT` | `whirl-server-gate` (test texts: the root of a llama.cpp source checkout, read as data) | none (required) |
| `WHIRL_GATE_WORK` | `whirl-server-gate` (same as `--work`: logs, SSD directories) | `%TEMP%\whirl-tests\gate` |
| `WHIRL_GATE_IMAGES` | `whirl-server-gate` suite `vis` (same as `--images`: `shapes.png`, `dialog.png`, `s1080.png`) | none |
| `WHIRL_GPU_LOCK` | `whirl-server-gate` (lock file held while a server runs, so two gate runs never share the GPU; `--no-lock` turns it off) | `%TEMP%\whirl-gpu.lock` |

The correctness gates we run on every change are described in
[benchmarking.md](guide/en/benchmarking.md#gates).

## Package

```powershell
powershell -ExecutionPolicy Bypass -File tools\package_release.ps1
```

Checks that both executables report the project version and import only Windows system DLLs plus
the driver's `amdhip64_7.dll`, then writes `whirl-<version>-windows-x64.zip` and its SHA-256 to the
output directory (`-OutDir`). It uploads nothing.
