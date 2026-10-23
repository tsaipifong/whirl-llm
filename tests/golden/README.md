# Golden hashes

Reference output of `whirl golden` for the golden-hash regression gate. One folder per GPU
architecture: `<arch>/<model>/<mode>_<suite>.txt`. Record and check them with
`tools/golden/golden.ps1`. See [docs/dev/testing.md](../../docs/dev/testing.md).

| Arch | State |
|---|---|
| gfx1201 (R9700) | recorded: swift and moe suites single, pair, ngram, server; dense (Q4_K_M, compatibility only) single, pair, ngram |
| gfx1151 (Radeon 8060S) | recorded: swift and moe suites single, pair, ngram, server; dense (Q4_K_M, compatibility only) single, pair, ngram |

The server suite (`<mode>_server.txt`) was recorded in 40dc199 (gfx1201) and e89e919 (gfx1151) and is part of
the default `golden.ps1` run.

Do not edit these files by hand. A commit that changes output on purpose re-records them and
says `golden update: <reason>` in its message.

`models.txt` holds the name, size and hash (size + first and last 4 MiB, SHA-256) of the two models the references were recorded with; `golden.ps1 check` refuses another file.
