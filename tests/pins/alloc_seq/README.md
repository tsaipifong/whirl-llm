# alloc_seq pins

Expected allocation sequences of the release gate step `alloc_seq` (tools/gates/alloc_seq_check.py, arch025-allocseq).
One file per card arch (`gfx1201.txt` R9700, `gfx1151.txt` Radeon 8060S), one line per model:

    <key> precise ctx4096 <model file> <bytes> events <N> seqhash <H>

plus the event list of each line (`<arch>_<key>.events.txt`; only sizes, kinds and ordinals, no addresses), which the
step uses to print the first differing event on a mismatch.

The value is the WHIRL_ALLOC_LOG seqhash of `whirl chat <gguf> x --precise --ctx 4096 --max-tokens 1 --device <card>
--dump-only` (the chat process's log). It pins invariant I2 (the Model's device / pinned allocations keep their sizes
and order), which no output check sees (mutant m1-state-order).

Record on the GPU, from a known-good build of the same tree, one card at a time:

    python -I tools/gates/alloc_seq_check.py run --exe <build>\whirl.exe --device r9700 --arch gfx1201 ^
        --model swift=<Swift-1.5 27B MXFP4-A gguf> --model moe=<Ornith-1.5-35B-A3B MXFP4 gguf> ^
        --pins tests/pins/alloc_seq --work <scratch dir> --record

`--record` runs every model twice and writes nothing when the two sequences differ. Re-record only when an allocation
change is intended (or a model file changed) and say why in the commit; a missing pin makes the step fail.
