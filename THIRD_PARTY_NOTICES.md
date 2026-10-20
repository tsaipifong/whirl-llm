# Third-party notices

WHIRL includes or is derived from the following third-party material.
Every entry here is also listed in PROVENANCE.md.

## llama.cpp / ggml

- Used for:
  - the IQ3_S lookup table `iq3s_grid` (kernels/iq_tables.h, added in Phase 2);
  - the qwen35 pre-tokenizer split `splitQwen35` in src/tokenizer/unicode.cpp,
    adapted from `unicode_regex_split_custom_qwen35` (src/unicode.cpp);
  - the special-token partition and BPE merge loop in
    src/tokenizer/tokenizer.cpp, adapted from `tokenizer_st_partition` and
    `llm_tokenizer_bpe_session` (src/llama-vocab.cpp).
  - image input (Phase 3, via the research prototype's item VIS): the
    preprocessing in src/vision/image.cpp (smart resize
    `calc_size_preserved_ratio`, the Pillow-compatible bicubic resampler
    `resize_pillow`, PAD_CEIL letterboxing; tools/mtmd/mtmd-image.cpp); the
    qwen3vl_merger vision graph in kernels/vision/vision.hip and
    src/vision/vision.cpp (tools/mtmd/clip.cpp, models/qwen3vl.cpp,
    models/qwen2vl.cpp temporal patch merge), the GGML_ROPE_TYPE_VISION /
    GGML_ROPE_TYPE_IMROPE rotations and the bilinear align-corners position
    embedding resize (ggml-cuda rope.cu / upscale.cu formulas, in
    kernels/vision/vision.hip and the attn_prep_m kernels of kernels/attn.hip),
    and the M-RoPE image token positions (tools/mtmd/mtmd.cpp
    MTMD_POS_TYPE_MROPE) in include/whirl/vismap.h.
  - the gfx1151 chunked Gated DeltaNet prefill (GDN-8S; default on for dense and
    MXFP4 MoE models on the Radeon 8060S, `WHIRL_GDN_STRIX=0` turns it off)
    in kernels/gfx1151/gdn_strix.hip, adapted from `gdn_chunk_prep_kernel` /
    `gdn_chunk_scan_kernel` (ggml/src/ggml-cuda/gated_delta_net.cu) of the
    strix-llama.cpp fork (https://github.com/asf0/strix-llama.cpp, rdna35-perf
    branch, same MIT license and copyright).
- Test-only (not part of any WHIRL binary or source tree): the chat-template
  oracle in tests/template_oracle compiles llama.cpp's `common/jinja` and
  `common/json.cpp` (with the vendored nlohmann/json, MIT) from an external
  checkout; `whirl-parity unicode` (tests/parity) reads `src/unicode-data.cpp` from an
  external checkout for comparison.
- Source: https://github.com/ggml-org/llama.cpp
- License: MIT

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ROCm hip-ep rgp_parser and rocprof-trace-decoder (developer tool only)

- Used for: tools/rgp_parser, a Python reader for Radeon GPU Profiler (.rgp)
  captures used by tools/quick/rgp_capture.py. Not part of any WHIRL binary.
  - tools/rgp_parser is vendored from ROCm/hip-ep `tools/rgp_parser` at commit
    0b9c2a7d47ff2e76bca6d39a90ac2bba3d8d7cc0 (https://github.com/ROCm/hip-ep).
  - tools/rgp_parser/rgp_parser/struct/sqtt_decode.py was modified to add a gfx12
    (RDNA4) SQTT token path ported from ROCm/rocm-systems
    `projects/rocprof-trace-decoder` source/gfx12 (gfx12token.cpp, gfx12parser.h,
    gfx12token.h) (https://github.com/ROCm/rocm-systems).
  - Details in tools/rgp_parser/PROVENANCE.txt; license text in tools/rgp_parser/LICENSE.
- License: MIT (both sources), Copyright (c) Advanced Micro Devices, Inc.

```
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Unicode Character Database

- Used for: the code point property tables in src/tokenizer/unicode_tables.inc,
  generated by the C++ tool tools/gen-unicode-tables (whirl-gen-unicode) from the UCD text files
  UnicodeData.txt / PropList.txt (general category, White_Space, simple lowercase
  mapping, Age).
- Source: https://www.unicode.org/ucd/
- License: Unicode License v3 (https://www.unicode.org/license.txt).
  Copyright (c) 1991-2026 Unicode, Inc.

## Model chat templates

- Used for: the fixed prompt strings in src/chat/chat_template.cpp (tool-call
  instructions, reasoning-effort sentences, vision placeholders), taken from the
  `tokenizer.chat_template` metadata of the supported Qwen3.5-family GGUF
  checkpoints (Qwen models are released under Apache-2.0).

## stb_image

- Used for: image decoding (PNG / JPEG / BMP / GIF / TGA / PSD / PNM) of image
  input: src/vision/stb_image.h (v2.30, unmodified) and its implementation unit
  src/vision/stb_image_impl.cpp.
- Source: https://github.com/nothings/stb (Sean Barrett)
- License: dual licensed, public domain (The Unlicense) or MIT; WHIRL uses it
  under the MIT alternative. Full text at the end of src/vision/stb_image.h.

```
MIT License
Copyright (c) 2017 Sean Barrett
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


## rocm_kpack.dll (AMD ROCm runtime component, redistributed in the release zip)

- Used for: nothing in WHIRL's own code. It ships next to whirl.exe and whirl-server.exe
  because the GPU runtime of the "AMD Software: Adrenalin Edition for ROCm 10.1" driver
  (amdhip64_7.dll 10.0.3684) needs it and that driver does not install it (GitHub issue #2).
  Unmodified binary, built by AMD: ROCm Core SDK 10.1.0 (TheRock `therock-10.1`), taken from
  AMD's `rocm-sdk-core` 10.1.0 Windows wheel on stable.repo.amd.com. File, hashes and
  provenance: redist/rocm_kpack/SOURCE.md.
- Source: https://github.com/ROCm/rocm-systems/tree/develop/shared/kpack
- License: MIT, Copyright Advanced Micro Devices, Inc.

```
MIT License

Copyright Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

rocm_kpack.dll statically links the following.

Zstandard (https://github.com/facebook/zstd), used under its BSD license:

```
BSD License

For Zstandard software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Facebook, nor Meta, nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

msgpack-c, C++ (header-only; https://github.com/msgpack/msgpack-c), Boost Software License 1.0:

```
Boost Software License - Version 1.0 - August 17th, 2003

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

## Microsoft Visual C++ 2015-2022 runtime (x64), redistributed in the release zip

- Files: msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll, version 14.44.35211.0, unmodified,
  Authenticode-signed by Microsoft. Taken from the Visual Studio Build Tools 2022 redist folder
  (VC\Redist\MSVC\14.44.35112\x64\Microsoft.VC143.CRT). Hashes and provenance:
  redist/vcruntime/SOURCE.md.
- Used for: nothing in WHIRL's own code (WHIRL links the C runtime statically). They ship next to
  whirl.exe and whirl-server.exe because rocm_kpack.dll (above) and the ROCm 10.1 driver's
  amd_comgr_3.dll import them, so the zip runs without installing the Visual C++ Redistributable.
- License: Microsoft Visual Studio 2022 license terms, "Distributable Code". The files under
  [VisualStudioFolder]\VC\redist may be copied and distributed, unmodified, with a program:
  https://learn.microsoft.com/visualstudio/releases/2022/redistribution
  Copyright (c) Microsoft Corporation. All rights reserved.
