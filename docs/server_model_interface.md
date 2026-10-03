# Server ↔ model interface

The server (`src/server`) is built against the model API `include/whirl/model.h`
and uses only the members listed below. The header mirrors the model interface
of the WHIRL Zig research prototype, and the server follows the same prototype's
server logic, so the two sides map one to one. This file records what the server
relies on, so that both sides keep it stable.

## How the server reaches the model

The engine never calls `qwen35::Model` directly. It talks to two interfaces,
so that it also runs (in tests, without a GPU) against a host-memory mock:

| Interface | File | Production implementation |
|---|---|---|
| `tier::DeviceOps` (copies, streams, events, pinned / device memory) | `src/tier/device_ops.h` | `src/tier/device_ops_hip.cpp` → `whirl/hip.h` |
| `server::ServerModel` (model operations) | `src/server/backend.h` | `src/server/backend_qwen35.cpp` → `qwen35::Model` |

`backend_qwen35.cpp` is a thin forwarder; every `ServerModel` call has the
semantics of the `Model` member of the same name.

## Members and functions of `qwen35::Model` the server uses

Startup (`src/server/server_main.cpp`):
`Config::fromGguf`, `Model::load(f, 0, stats, LoadOptions{max_batch, embd_on_host})`,
`loadOrTune`, `pickDevice`, `mtpDefaults`, `requantMtpQ4`, `buildDraftHeadEx`,
`setupSeqs(parallel, slot_ctx)` (before `allocKvPool`), `ensureSnapshots`,
`kvAutoDense`, `setKvFormat`, `kvBytesPerTokenFmt`, `kvBytesPerToken`,
`allocKvPool`, `kvName`, `pendLayerBytes`, `gdn_ord`, `convBytes`, `ssmBytes`,
`fusedDecode`, members `no_fuse`, `float_gemv`, `naive_attn`, `gdn_chunked`,
`moe_bn_force`, `max_batch`, `draft_vocab`, `use_graph`, `gdn_replay`,
`snap_sets`, `pool_pages`, `embd_host`, `mtp`, `module`, `k.gdn_step_norm`,
`envGet` (`WHIRL_*` names).

Per request / cycle (through `ServerModel`):

| Call | Relied-on semantics |
|---|---|
| `selectSeq(s)`, `cur_seq`, `conv_state`, `ssm_state`, `mtp_h` | after `selectSeq(s)` the three views are sequence s's (0 entries = attention layers) |
| `seqs[s].hid` | at least `max_small_batch` rows of `n_embd` f32: the MTP hidden rows of the pend tokens |
| `commitSeq(s)`, `setPending(s, rows)`, `dropPending(s)` | replay mode: a verify does not change the persistent state; its kept rows (setPending) reach the state at the next verify or `commitSeq`; `verifyBatchEnqueue` clears `npend` of its sequences |
| `restoreSeqSnapshot(s, set)` | snapshot mode (`gdn_replay` off): state after row `set - snap_base` |
| `mapPages(s, first, phys)` | page-table entries `first ..` of sequence s = pool pages `phys` |
| `reset()` | current sequence back to the empty state |
| `forward(tokens, pos0)` / `prefillMtpChunk(tokens, pos0, off, n, prev_hidden)` | rows `tokens[off .. off+n)` at positions `pos0 + off ..`; `prev_hidden` = normed hidden row of position `pos0 + off - 1` (only used at a chunk start after a checkpoint); after it `logits` row 0 = last row, `hn` rows = every row, `mtp_h` row 0 = last normed hidden row |
| `canSegment()`, `prefillSegs(segs, mtp)`, `segLogitsToFront(k)` | `hn` rows concatenated in segment order; segment k's last-row logits moved to row 0 by `segLogitsToFront(k)`; bit-identical to the solo chunks |
| `argmax()` | argmax of `logits` row 0 |
| `draft_p_min`, `draft_n_min` | device-side p-min cutoff of the MTP drafts during one cycle (reset to 0 after) |
| `mtpBatchStepEx(segs, r, draft)` | r = 0: each seg's pend tokens at `pend_pos ..` with hidden rows from `seqs[seq].hid`; r > 0: one chained row (draft r-1); draft r lands in the seq's control area `ctl_drafts + r`, kept count in `ctl_nd`; `draft = false` only fills the MTP KV rows |
| `verifyBatchEnqueue(vsegs)` | seg k runs `[next, drafts..]` at `pos ..`; drafts from the control area when `drafts` is empty, literal otherwise (n-gram); `logits` / `hn` rows in seg order; row argmaxes in `ctl_rows + r` of each sequence; returns the total rows |
| `readCtl(dst)` | one synchronizing copy of the control areas of sequences `0 .. n-1` (`ctl_words` i32 each) |
| `mtpEnqueue(hidden, tokens, 0, pos0, nullopt)` | MTP KV rows only (end of a request) |
| `logits`, `hn` | row r at `+ r * n_vocab * 4` / `+ r * n_embd * 4` |
| `kcache`, `vcache`, `kscale`, `vscale`, `mtp_kc/vc/ks/vs`, `kv_q8`, `kv_kf16`, `cfg.n_head_kv`, `cfg.head_dim` | the host tiers copy whole pool pages: per attention layer then the MTP layer, K (`elems * (q8 && !kf16 ? 1 : 2)` bytes per row), V (`elems * (q8 ? 1 : 2)`), K scales (q8 / q8h only), V scales (q8*): `elems / 32 * 2`; the sum of these rows must equal `kvBytesPerToken()` (the server checks it and turns the tiers off otherwise) |
| `prof` + `Profile` + `opClassName` | `WHIRL_PROFILE=1` per-op-class times |
| kernel `topk_rows` (`module.getFunction`) | sampling candidates: `(x, n, K, inv_t, ids, vals, stats)`, stats = {max, Σ exp((x - max) * inv_t)} |

Host-only policy code from the model library used as is: `DraftAccept`,
`DraftTiming`, `pickDrafts`, `Ngram`, `NgramPolicy` (`src/model/spec.cpp`; the
mock-model test executable compiles that file directly).

## Stability rules and notes

1. Keep the names and semantics above stable (the header note says
   "additions are allowed, renames are not").
2. The server entry point is `whirl::server::serveMain(argc, argv)`
   (`src/server/server_main.h`, library target `whirl_server_core` plus the
   files of the `whirl-server` target); `whirl-server.exe` is the server
   executable.
3. `WHIRL_GEMV_R / WHIRL_GEMV_W / WHIRL_GEMV_WH` are CLI-only: their parsers are
   not exposed by `model.h`, and the server ignores them with a warning.
4. The kernel `topk_rows` must stay in the code object (the server's sampler
   looks it up by name).
