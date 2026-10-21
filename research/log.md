# Decision / dead-end log  (provenance: user | ai)

- 2026-09-30 user: git repair via `git worktree repair`; harness full protocol fix; v3 base = v2 lifecycle degradation (CLAUDE.md decision 2 to be updated by user); severity bins = docx 4 bins.
- 2026-09-30 user: "Thực hiện code theo kế hoạch plan.md" → plan approved for Phase 2.
- 2026-09-30 ai: `git worktree repair` alone silently did nothing (`.git` file held a Windows path). Rewrote `B-Free-loss-backbone/.git` gitdir line to the Linux path, then repair worked. 38 files show CRLF-only diffs vs LF blobs; only intentionally-edited files were converted to LF and committed.
- 2026-09-30 ai: `tests/` is in the repo `.gitignore` → committed with `git add -f`.
- 2026-09-30 ai: GSR window=3 fully masks a 4×4 patch grid (softmax NaN); CPU tests use img 112 (8×8). Real 504 → 36×36 unaffected.
- 2026-09-30 ai: v2 notebook hardcoded `--batch_size 64` (violates CLAUDE.md auto-batch rule) → v3 uses 0.
- 2026-09-30 ai: assumed noise σ unit 0–255 (docx σ_n=2..30); `single_op` chosen as default compose (docx ambiguous). Both flagged in direction README.
- 2026-09-30 ai: attempted a synthetic end-to-end `train_lora.py` CPU smoke run; user declined the command. Not run: main()'s loop/CSV edits are only py_compile'd and covered by glue tests, NOT executed end-to-end. First Kaggle A1 run is the real check.
- 2026-09-30 user: code only on `integration/loss-backbone`, commit+push as P-Bao <BaoPV.B23KH006@stu.ptit.edu.vn>; harness scope = minimal (only new version3 ours eval copy; b-free/gf untouched); training notebook = A1, one seed (42) only; keep research/ but A1-only.
- 2026-09-30 ai: `v3-dev` removed; the 4 commits were re-created with P-Bao author (git commit-tree, trees identical), fast-forwarded into `integration/loss-backbone`, pushed. Co-Authored-By trailer dropped.
- 2026-09-30 ai: setup notebooks checked, no library additions; observed (not changed) `py7zr` missing wheel and transformers 4.56.2 vs 4.55.4 mismatch in setup-eval.
- 2026-10-02 user: training exceeds the 12h Kaggle session → split epochs over 2 notebooks via checkpoint.
- 2026-10-02 ai: 4+4 epoch split, cosine T_max kept at 8 epochs; stage 2 reuses stage-1 batch size (steps/epoch fixes the schedule). Details in iter-2.md.
