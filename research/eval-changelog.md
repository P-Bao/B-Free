# Evaluation-harness changelog

| date | file | what | why | verified how |
|---|---|---|---|---|
| 2026-09-30 | notebooks/version3/xla-eval-harness-ours.ipynb | new copy of version2 ours harness; only OURS_CKPT → v3 last-epoch ckpt, outputs cleared | evaluate v3 with an unchanged yardstick | cell-by-cell diff vs v2: only cell OURS_CKPT differs |
| — | b-free / globalforge-dinov3 harnesses | NOT edited | protocol unchanged; baselines stay comparable | mtime unchanged |
