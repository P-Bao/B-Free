# iter-1 — Multi-Severity DCS (v3), run A1 seed 42

## Changed (branch `integration/loss-backbone`, author P-Bao, pushed)
- `configs/bfree_dcs.yaml`: `multi_severity` section (default off, levels from docx).
- `modules/dcs_loss.py`: `multi_severity_dcs` (weighted mean of unchanged `info_nce_loss`).
- `networks/bfree_globalforge_vit.py`: `compute_loss` takes Tensor (v2 path, unchanged) or list of K views; same 3 scalars; per-view terms in `last_dcs_terms`.
- `bfree_datasets/multi_severity.py`: `MultiSeverityDegrader`.
- `train_lora.py`: reads `--config` `multi_severity`; K-view batches; batch finder probes K views; `--select last|best` (+ `selected_ckpt.txt`); `dcs_k*` CSV columns in MS mode.
- `notebooks/version3/xla-training.ipynb` (A1, seed 42 only) and `notebooks/version3/xla-eval-harness-ours.ipynb` (copy of v2; only `OURS_CKPT` changed).
- Setup notebooks: no new libraries needed (code uses numpy/PIL/pyyaml/torch/timm, all bundled). b-free / globalforge harnesses untouched (protocol unchanged).

## Verified on CPU (env `xla-cpu`, torch 2.8.0+cpu)
- `tests/test_dcs_loss.py`, `test_backbone_shape.py`, `test_multi_severity.py` ALL PASS; default path bit-equal to pre-change reference (`tests/ref_single_view.pt`); `tests/overfit_test.py` **GATE PASS**.
- Notebooks: `json.tool` + AST OK; eval copy differs from v2 only in `OURS_CKPT` (+ changelog cell).
- **Not verified**: end-to-end `train_lora.py` main() on CPU — the first Kaggle run is the real check.

## You must do
1. Re-run `xla-setup` (online) so `bfree_src` contains the pushed code; attach its output to training.
2. Run `notebooks/version3/xla-training.ipynb`. Confirm the eval notebook's `OURS_CKPT` slug (`xla-training-v3` assumed).
3. Edit CLAUDE.md yourself (decision 2 → v2 lifecycle pipeline; `papers/`→`paper/`; remove `agent.md` / `.claude/hooks` / `B-Free-integration` refs).

## Paste back
- training notebook last cell + `/kaggle/working/train_log.csv`; first ~30 lines of `train_log.txt` (`[multi_severity] ON`, auto batch size)
- last cell of `notebooks/version3/xla-eval-harness-ours.ipynb`
