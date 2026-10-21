# PATHS — verified 2026-09-30 (ls / find)

| CLAUDE.md expects | Reality |
|---|---|
| `agent.md` | MISSING anywhere. Rules taken from CLAUDE.md + plan.md only |
| `.claude/hooks/guard_edits.py` | MISSING. `settings.json` at root (not active). No hook → discipline manual |
| `scripts/lock_readonly.sh` | root `lock_readonly.sh` (never run) |
| `B-Free/` | present; HEAD = `integration/train-eval` (not main) |
| `B-Free-integration/` | MISSING |
| `B-Free-loss-backbone/` | present. `.git` file fixed 2026-09-30 (was Windows path) → `git -C B-Free worktree repair`. Branch `integration/loss-backbone` (all code work here; no side branch) |
| tests | none existed; new in `B-Free-loss-backbone/tests/` |
| config | `B-Free-loss-backbone/code/configs/bfree_dcs.yaml`, `loader.py` |
| train entry | `B-Free-loss-backbone/code/train_lora.py` (Kaggle, accelerate) |
| `notebooks/version1/` | `xla-training.ipynb`, `xla-training-phase-2.ipynb`, `xla-eval-harness-ours.ipynb` |
| `notebooks/version2/` | `xla-training.ipynb`, `xla-eval-harness-ours.ipynb` |
| harnesses outside version dirs | `notebooks/xla-eval-harness-b-free.ipynb`, `notebooks/xla-eval-harness-globalforge-dinov3.ipynb` |
| `notebooks/version3/` | to create |
| `papers/` | actually `paper/`: `paper/latex_paper_bfree_globalforge/latex_paper_bfree_globalforge/main.tex` (+`main_vi.tex`, `references.bib`). `paper/B-free/`, `paper/globalForge/` = original sources, read-only |
| docx | root `Multi-Severity DCS AI Detection.docx`; copy in `research/directions/D-multiseverity-dcs/source/` |
| extra | `test/00_offline_assets_setup.ipynb`, `01_eval_harness_template.ipynb`, `02_aggregate_compare.ipynb` |

Local CPU env: conda env `xla-cpu` (py3.11, torch CPU 2.8.0, timm 1.0.22, peft 0.15.2, nbformat). `face_recognition` env untouched.
Line endings: worktree files are CRLF, committed blobs LF → 38 files show "modified" (whitespace only). Commit only files intentionally touched.
