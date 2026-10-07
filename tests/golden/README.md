# Golden hashes

Reference output of `whirl golden` for the golden-hash regression gate. One folder per GPU
architecture: `<arch>/<model>/<mode>_<suite>.txt`. Record and check them with
`tools/golden/golden.ps1`. See [docs/dev/testing.md](../../docs/dev/testing.md).

| Arch | State |
|---|---|
| gfx1201 (R9700) | recorded |
| gfx1151 (Radeon 8060S) | recorded |

Do not edit these files by hand. A commit that changes output on purpose re-records them and
says `golden update: <reason>` in its message.
