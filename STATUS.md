# WHIRL 狀態

WHIRL = Windows HIP Inference for RDNA LLMs。以 C++ + HIP 撰寫的推論引擎，授權 Apache-2.0。

**狀態（2026-10-03）：版本 0.1.0。** R9700（gfx1201）的 kernel、載入器、forward、推測解碼（MTP + n-gram）、
OpenAI 相容伺服器（continuous batching、prefix cache、VRAM → RAM → SSD 分層快取、Ctrl+C 正常收尾）、圖片輸入、
Ornith MXFP4 專家路徑都已完成。C++ 程式是 WHIRL 自己的 Zig 研究原型的乾淨重寫；行為以黑箱方式對該原型驗證
（輸出逐 token / 逐位元相同）。研究原型已退役，不再開發，也不在本儲存庫中。

gfx1151（Radeon 8060S）：`gfx1151` 分支上為預覽版（結果正確、尚未調校），見第 9 節。使用者文件：README.md、docs/usage.md、docs/quickstart.md、docs/benchmarks.md；
各檔來源見 PROVENANCE.md。

以下各節是各階段的驗收紀錄（數字為當時量測值）。

## 1. 主機端基礎（tokenizer、GGUF、chat template）

| 項目 | 檔案 | 驗證 |
|---|---|---|
| 建置 | `CMakeLists.txt`、`build.bat`、`tools/bin2c/` | CMake + Ninja；MSVC 編 host（C++20，/W4 無警告）；HIP SDK 7.2 clang 為 gfx1201 / gfx1151 各編一個 code object，以 bin2c 內嵌 |
| HIP runtime wrapper | `include/whirl/hip.h`、`src/hip/hip_runtime.cpp` | 裝置列舉 / 選擇、記憶體、stream / event / graph、`hipModuleLoadData`、`launch` |
| GGUF reader | `include/whirl/gguf.h`、`src/gguf/gguf.cpp` | v2/v3、全部 value type、巢狀 array、alignment、mmap；13 個 GGUF 與 gguf-py 完全一致 |
| Unicode 表 | `tools/gen-unicode-tables/` → `src/tokenizer/unicode_tables.inc` | 輸出 Unicode 15.1 等級以對齊 llama.cpp；與 llama.cpp `unicode-data.cpp` 逐碼位 0 差異 |
| Tokenizer | `src/tokenizer/{unicode,tokenizer}.cpp` | qwen35 預切詞 + byte-level BPE；110 檔 × 2 模型 × parse_special 開 / 關 = 440 組與 llama-tokenize 逐 token 一致 |
| JSON + chat template | `src/chat/{json,chat_template}.cpp` | 45 案例 × 2 種樣板與 llama.cpp Jinja 引擎渲染 GGUF 樣板比對：0 個非預期差異 |
| 單元測試 | `whirl-tests.exe` | 164 / 164 |

細節與指令：`docs/phase1_parity.md`。

已知差異（刻意）：

- llama.cpp 的 Unicode 表是 15.1；UCD 16.0 新增的 5,185 個碼位在 llama.cpp 是 UNDEFINED。產生器預設輸出 15.1 等級（`--unicode 16.0` 可切換）。
- F5–F7 前導位元組會解出 > U+10FFFF 的碼位：llama-tokenize 因未處理的例外而終止；WHIRL 回報 `TokenizerError`。
- chat template 與 Jinja 不同之處：中段 system 訊息保留、arguments 為 JSON 字串時解析、未知 role 報錯、缺 function name 報錯、圖片 / 影片預設報錯（`ChatOptions::allow_media` 可輸出 vision placeholder）；trim 只去 ASCII 空白（同 llama.cpp）。
- llama.cpp Jinja 的 `tojson` 浮點數格式與 Python 不同；WHIRL 依 Python `json.dumps`。

## 2. Kernel（gfx1201）

- 24 個 kernel 族檔案（`kernels/*.hip`），由 `whirl_kernels.hip` 依序 include；`iq_tables.h` 保留 ggml MIT 檔頭。
- 與研究原型同旗標編譯：**2689 個 kernel 的機器碼逐指令相同**；kernarg 版面、SGPR / VGPR / spill / LDS / scratch metadata 與 occupancy 也相同。
- `whirl-kernel-test.exe`（全部 C++）：每種量化型別的 block 解碼、GEMV / GEMM、paged attention、Gated DeltaNet、MoE、norm / RoPE / SiLU、e4m3、requant、draft head 的 CPU 參考比對，以及 bitwise invariance（多 token == 1 token、grouped == ungrouped、prefill GEMM 選擇不變、attn 分組）。真權重：Qwen3.8-27B UD-Q4_K_M、Qwen3.8-27B MXFP4、Ornith-1.5-35B Q4_K_M。
- 結果：最新全套 **412 / 412**（exact 83、tolerance 179、invariance 150）。
- `silu_mul_x16h`（Q4_K_M relaxed mode）依設計不是 bitwise（f16 儲存只捨入一次），以 1 f16 ulp 容差檢查。
- gfx1151 的 kernel 集見第 9 節。

## 3. 模型與 CLI

| 項目 | 檔案 | 內容 |
|---|---|---|
| 模型 API | `include/whirl/model.h` | kernel 參數結構 / kernel 表使用 `kernels_abi.h` 的唯一定義 |
| 權重載入 | `src/model/loader.cpp`、`config.cpp` | `general.architecture` 只收 `qwen35` / `qwen35moe`（其他 `UnsupportedArch`）、無 kernel 的 tensor 型別 `UnsupportedTensorType`、MXFP4 重排、Q4 MTP requant、draft head、KV pool（f16 / q8 / q8h / q8v、`WHIRL_KV`）、序列 / 頁表 / 快照 / replay |
| forward | `src/model/forward.cpp` | prefill（分段、chunk、多請求 segmented）、decode（融合 kernel、grouped GEMV、HIP graph）、MTP draft / verify、MoE、vision 掛鉤 |
| autotune | `src/model/tune.cpp` | 11 個 batch bucket，快取 `%LOCALAPPDATA%\whirl\tune4-*.txt`；bitwise 自檢 |
| 推測解碼策略 | `src/model/spec.cpp` | MTP 草稿數成本模型、n-gram co-drafting |
| CLI | `src/cli/main.cpp` | `whirl chat` / `bench` / `selftest` / `seqtest` / `devices`；`WHIRL_*` 環境變數見 `docs/usage.md` |

### 驗收（R9700，對研究原型 exe 黑箱比對）

3 模型（Qwen3.8-27B UD-Q4_K_M、Qwen3.8-27B MXFP4、Ornith-1.5-35B Q4_K_M）× 7 提示（中 / 英 coding、thinking 開 / 關、2k、8k thinking、32k）= 21 組：

| 檢查 | 結果 |
|---|---|
| greedy 輸出（預設 MTP + n-gram）逐 token = 原型 | 21/21 |
| greedy 輸出（`WHIRL_MTP=0` plain）逐 token = 原型 | 21/21 |
| WHIRL 自身 MTP 輸出 == plain 輸出 | 21/21 |
| 最後一個 prompt 位置 logits 逐位元 = 原型（`--out`） | 21/21（含 32k、thinking 分段） |
| `whirl selftest` | 3/3 模型 ok |
| `whirl seqtest`（segmented prefill、2 序列 batched decode、literal drafts verify == 單序列） | 3/3 模型 ok |

速度與原型相同（交錯 3 輪中位數，18 項全部在 ±1% 內）。

已知：MXFP4 32k 提示的 MTP decode 呈雙峰（約 56.7 或 61.7 tok/s），取決於成本模型是否在某一 cycle 採用 n-gram 草稿；原型與 WHIRL 相同，輸出不受影響。

## 4. 伺服器與 KV 分層快取

- `src/server/`：OpenAI 相容 HTTP（/health、/v1/models、/v1/chat/completions、/v1/completions、SSE、CORS）、tool calls、reasoning_content、取樣參數、continuous batching（1..16 slots）、每 slot 的 MTP / n-gram 草稿、prefix cache、共用 system / prefix checkpoint、page refcount / copy-on-write、host tiers（RAM → SSD、QD4 讀取、prefetch、delay hit、restore tail）。
- `src/tier/`：KV 分層快取，裝置操作經 `DeviceOps` 介面（可換成 host 記憶體 mock）。
- 無 GPU 測試：`whirl-tier-tests`、`whirl-server-tests`（mock 模型跑 HTTP 端到端：greedy == 參考、stream == non-stream、多輪 prefix cache == 冷啟動、C=4 / C=16 併發、LRU 驅逐、共用 checkpoint、RAM / SSD 還原與重啟索引、seeded sampling、stop / eos）。
- 真模型關卡 `whirl-server-gate.exe`：Qwen3.8-27B UD-Q4_K_M，每個 suite 先跑原型 exe 留參考，再跑 WHIRL 逐字比對：basic 23/23、mt_cache 5/5、pool 7/7、tier 25/25、sys 11/11、restore_conc 12/12，記錄輸出全部相同。Ornith MXFP4 同樣全過。
- 未做：`WHIRL_PROFILE=2` 的 GPU 分段計時；`WHIRL_GEMV_R/W/WH` 在 server 被忽略。

## 5. 圖片輸入（vision）

- `qwen3vl_merger` mmproj（F16；BF16 / F32 載入時轉 f16），27 層 ViT 編碼器在獨立 code object（第一張圖才載入），權重在 pinned host RAM（887 MiB），VRAM 夠就常駐、否則逐層串流；活化值借 LM prefill scratch；`--vis-idle-s` 秒後釋放。
- 前處理：stb_image 解碼 + smart resize + Pillow 相容 bicubic + PAD_CEIL（改寫自 llama.cpp mtmd，MIT）。
- LM：multi-section RoPE、影像 token id = 內容雜湊（prefix cache / tier 可重用）、圖片結尾共享 checkpoint。
- 驗證（R9700，對原型黑箱）：7 張圖前處理 RGB 與 embedding 逐位元相同 7/7（常駐與串流）；CLI 3 模型 × 4 圖 × {MTP, plain} greedy 輸出相同 24/24；server gate `vis` 19/19，記錄輸出 7/7 相同；編碼速度差 ≤ ±1%。
- 限制：只支援 gfx1201；只支援無 deepstack 的 qwen3vl_merger mmproj；不支援影片；不抓 http 圖片網址。

## 6. Ornith MXFP4 專家

- MXFP4 routed experts：通用 int8 decode / f16 WMMA prefill、decode 整塊 kernel `moe_gu_mxfp4w` / `moe_down_mxfp4w`（decode 與 verify 共用 → MTP == plain）、prefill MXFP4 × fp8 grouped GEMM + `moe_gather_fp8`。預設 fp8（`WHIRL_MOE_FP8=0` → f16）；`WHIRL_MOE_MXW=0` → 通用 decode kernel。
- 既有 kernel 機器碼不變，只新增 11 個 kernel。
- 驗證：kernel-test 新家族 `moemx` 16 項；Ornith MXFP4 7 提示 MTP + n-gram == MTP == plain 7/7、對原型輸出 7/7 相同、最後位置 logits 7/7 逐位元相同；server gate 全部 suite 通過且輸出與原型相同。
- Server 正常收尾（Ctrl+C / 關閉主控台）：排隊請求回 503、執行中的跑完、閒置 slot spill、RAM 層未寫入項目寫入 SSD（≤ 10 s）後結束；重啟後同一請求從 SSD 還原，輸出相同。

### 速度（R9700；WHIRL greedy；llama.cpp b11214 ROCm `-fa on`、`-ub 2048`）

| | WHIRL MXFP4 | WHIRL Q4_K_M | llama.cpp MXFP4 | llama.cpp Q4_K_M |
|---|---|---|---|---|
| Prefill 88 / 2k / 8k / 32k（tok/s） | 2,542 / 11,729 / 10,853 / 7,978 | 1,338 / 5,862 / 5,992 / 4,980 | 1,820 / 4,820 / 4,760 / 3,936 | 1,725 / 4,383 / 4,328 / 3,639 |
| Decode MTP + n-gram（bench 144-token 提示） | 303.6 | 269.9 | — | — |
| Decode MTP only（bench） | 258.1 | 238.1 | — | — |
| Decode no MTP（bench；llama.cpp tg256） | 193.1 | 179.8 | 120.4 | 102.5 |
| Server 單請求 decode 中位數（MTP + n-gram；llama.cpp no MTP） | 246.0 | 203.2 | 113.3 | 96.0 |
| Server C=4 aggregate（MTP + n-gram / no MTP；llama.cpp no MTP / MTP） | 435 / 355 | 347 / 321 | 227 / 138 | 219 / 84 |

（表中 Q4_K_M 欄為 Ornith-1.5-35B-A3B。完整量測見 docs/benchmarks.md。）

已知：部分 256-token 提示上 MTP only 比 MTP + n-gram 快 1–5%；n-gram 成本模型尚未調整。Ornith MXFP4 的視覺路徑未在 C++ 版測試。

## 7. 工具鏈

| 工具 | 版本 |
|---|---|
| CMake | 4.4.2 |
| Ninja | 1.13.2 |
| MSVC | 19.44（VS 2022 Build Tools 17.14）；`build.bat` 會自動呼叫 vcvars64 |
| HIP SDK | 7.2（device 編譯用其 `clang.exe`，host 連結 `amdhip64.lib`） |

## 8. 重現驗證（全部是 C++ 工具；不需要 Python）

```
build.bat Release
rem host 單元測試（不用 GPU）
build\Release\whirl-tests.exe
build\Release\whirl-model-tests.exe
build\Release\whirl-server-tests.exe --gguf <GGUF>
build\Release\whirl-tier-tests.exe
rem 對照外部參考（llama.cpp 原始碼 / llama-tokenize / template oracle / GGUF 參考 dump）
build\Release\whirl-parity.exe unicode   --inc src\tokenizer\unicode_tables.inc --llama <llama.cpp 原始碼目錄>
build\Release\whirl-parity.exe gguf      --whirl-tool build\Release\whirl-tool.exe --ref-dir <DIR> <GGUF...>
build\Release\whirl-parity.exe make-corpus --out tests\corpus\synthetic
build\Release\whirl-parity.exe tokenizer --whirl-tool build\Release\whirl-tool.exe --llama-tokenize <llama-tokenize.exe> --model <GGUF> --work <DIR>
cmake -DWHIRL_LLAMACPP_SRC=<llama.cpp 原始碼目錄> build\Release && cmake --build build\Release --target whirl-template-oracle
build\Release\whirl-parity.exe template  --whirl-tool build\Release\whirl-tool.exe --oracle build\Release\tests\template_oracle\whirl-template-oracle.exe --work <DIR> --model-a <Qwen3.8 GGUF> --model-b <Ornith GGUF>
rem GPU：kernel 對 CPU 參考、模型自檢、多序列路徑、輸出 / logits / 速度
build\Release\whirl-kernel-test.exe                  (Radeon 8060S: --device 8060s)
build\Release\whirl.exe selftest <GGUF>
build\Release\whirl.exe seqtest <GGUF>
build\Release\whirl.exe chat <GGUF> "PROMPT" --max-tokens 256 --no-stream          (WHIRL_MTP=0：plain)
build\Release\whirl.exe chat <GGUF> --tokens 760,6511,314 --max-tokens 1 --out logits.f32
build\Release\whirl.exe bench <GGUF> --prefill 2048,8192,32768 --decode 256
set WHIRL_GATE_TEXT_ROOT=<llama.cpp 原始碼目錄>
build\Release\whirl-server-gate.exe --exe build\Release\whirl-server.exe --kind whirl --model <GGUF> --suite basic --results OUT.json [--compare REF.json]
```

## 9. gfx1151（Radeon 8060S）bring-up（預覽、尚未調校）

| 項目 | 內容 |
|---|---|
| Kernel | `kernels/gfx1151/` 17 個家族檔（研究原型的 gfx1151 kernel 原始碼，含 MXFP4、f16 prefill 融合、grouped 1-token GEMV）；與原始碼同旗標編譯逐 kernel 機器碼相同。新增：`MOE_ENTRIES(mxfp4)`、`attn_prefill_wmma*` 的 `h0`（原本缺，長 prefill 分 head 區段會算錯）、能力標記 `whirl_cap_xd_sum` / `whirl_cap_attn_group1` |
| 建置 | `WHIRL_GPU_ARCHS` 預設 `gfx1201;gfx1151`，可只編一套；每個 code object 只依賴自己的原始檔；執行時依 `gcnArchName` 選 |
| Host | `kernels::Caps`（從 code object 探測：fp8_gemm、kv_q8v、kv_q8h、gemvw、gdn_replay、mrope、xd_sum、attn_group1）取代架構判斷：KV auto（f16 → q8v → q8h → q8，缺的跳過；指定沒有的格式回報 `UnsupportedKvFormat`）、attention query 分組、DeltaNet replay。預設裝置：有 gfx1201 kernel 時第一張 R9700，否則第一張有 kernel 的 GPU。server：內顯（`integrated`）預設 KV pool 262,144 token、RAM / SSD 層關（`--kv-ram-mb` 開啟）。圖片輸入在 gfx1151 上回報不支援 |
| 測試 | `whirl-kernel-test --device 8060s`：exact 74/74、tol 151/151、invariance 89/89（27 項 skip：gfx1151 沒有的 kernel）；CPU 參考依 `Caps::xd_sum` 處理 packed scale word。`whirl-server-gate` 加 `--kv`、`--server-env` |

### 驗收（8060S）

| 檢查 | 結果 |
|---|---|
| MTP + n-gram（預設）輸出 == plain greedy，7 提示（中 / 英、thinking 開 / 關、2k、8k、32k） | 27B Q4_K_M 7/7、Swift MXFP4-A 7/7、Ornith MXFP4 7/7 |
| `whirl selftest` / `seqtest`（27B Q4_K_M） | ok / ok |
| server gate（27B Q4_K_M） | basic 22/22、mt_cache 4/4；pool 6/6、sys 10/10、restore_conc 11/11、tier 24/24（`--server-env WHIRL_KV_RAM_MB=8192 --kv f16`：tier 預設關、沒有 q8v）；Ornith MXFP4 basic + mt_cache 26/26 |
| 對 R9700（同提示 greedy） | 27B Q4_K_M：6 提示中 2 個全同，其餘在第 37–178 token 分岔；最後 prompt 位置 logits top-1 6/6 相同、KL 5e-8–1.2e-3。Ornith MXFP4：第 5–90 token 分岔（R9700 用 fp8 prefill / fp8 專家，8060S 用 f16），top-1 6/6 相同、KL 6.5e-5–5.7e-2 |
| 對 llama.cpp b11214（8060S，en 提示 128 token） | Swift 全同；27B Q4_K_M 前 66 字元相同；Ornith 前 76 字元相同（措辭分岔） |
| 首次 autotune | 27B Q4_K_M 460.8 s、Swift MXFP4-A 85.3 s、Ornith MXFP4 10.1 s |

### 速度快照（bring-up，未調校；llama.cpp b11214 ROCm `-fa on`；2 輪交錯平均）

| 8060S | WHIRL Ornith MXFP4 | llama.cpp | WHIRL Swift MXFP4-A | llama.cpp | WHIRL Q4_K_M | llama.cpp |
|---|---|---|---|---|---|---|
| Prefill 88 / 2k / 8k（tok/s） | 560 / 1,475 / 1,372 | 779 / 1,312 / 1,186 | 252 / 389 / 366 | 264 / 292 / 272 | 234 / 357 / 337 | 260 / 284 / 264 |
| Decode no MTP（bench；llama.cpp tg256） | 85.9 | 66.8 | 14.9 | 13.5 | 13.2 | 12.4 |
| Decode MTP / MTP + n-gram（bench） | 106.3 / 117.9 | — | 29.5 / 29.3 | — | 28.8 / 33.0 | — |

待做（調校）：短提示用的 gfx11 WMMA 小批次 GEMM、多 token grouped GEMV 雙胞胎、prefill GEMM 重新調校、MXFP4 整塊 decode 專家、圖片輸入。
