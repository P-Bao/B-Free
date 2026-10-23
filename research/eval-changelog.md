# Evaluation-harness changelog

| date | file | what | why | verified how |
|---|---|---|---|---|
| 2026-09-30 | notebooks/version3/xla-eval-harness-ours.ipynb | new copy of version2 ours harness; only OURS_CKPT → v3 last-epoch ckpt, outputs cleared | evaluate v3 with an unchanged yardstick | cell-by-cell diff vs v2: only cell OURS_CKPT differs |
| — | b-free / globalforge-dinov3 harnesses | NOT edited | protocol unchanged; baselines stay comparable | mtime unchanged |
| 2026-10-22 | `code/eval_protocol.py` (new) | 5-crop patch-grid, mean logit, dmetrics-based metrics, bootstrap CI, paired bootstrap, parquet writer, Arrow-dir resolver | protocol v2 (`research/eval-v2-proposal.md`); not imported by any published notebook | `tests/test_eval_protocol.py` pass; 5-crop diff vs `Wrapper5crops` < 1e-5 |
| 2026-10-22 | `notebooks/version4/xla-eval-harness-ours.ipynb` (new copy of v3) | protocol v2 cells, 500 fake/gen, no test-pool calibration, per-image parquet + provenance, `seen` flag | Phase 0 H1–H8, H11 | cell diff vs v3 + AST/json validation; **not run on Kaggle yet** |
| 2026-10-22 | `notebooks/version4/xla-eval-harness-b-free.ipynb` (new copy) | same protocol; bypass `Wrapper5crops.forward`, raw logit (fixes H11) | B-Free must be measured on the same 5 crops | AST/json validation; step-0 equivalence still gate G0 |
| 2026-10-22 | `notebooks/version4/xla-eval-harness-globalforge-dinov3.ipynb` (new copy) | same protocol; `l1-l0` logit instead of softmax | H3 | AST/json validation |
| 2026-10-22 | `notebooks/version4/xla-report-v4.ipynb` (new) | reads `scores_*.parquet`, paired bootstrap ours−baseline | Phase 3 table | AST/json validation |
