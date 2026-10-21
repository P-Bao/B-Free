# iter-2 — split A1 training over 2 Kaggle sessions (12h limit)

## Changed
- `train_lora.py` (branch `integration/loss-backbone`):
  - `--end_epoch N` (0 = all): stop after epoch N. `--epochs` stays 8 and fixes the cosine T_max, so the LR curve equals an uninterrupted run.
  - `--resume_from resume_state.pth`: restores model, optimizer, scheduler, loader generator, python/numpy/torch/cuda RNG, train_log, best bAcc.
  - `resume_state.pth` is written (atomically) after every epoch. Stage 2 takes batch size from it (finder skipped; T_max depends on steps/epoch) and asserts steps/epoch is unchanged.
  - Resume refuses to start if any hyperparameter in `RESUME_CHECK_KEYS` differs from the stage-1 config.
  - `train_log.csv` of stage 2 carries stage-1 rows. `selected_ckpt.txt` is only written when `end_epoch == epochs`.
  - Also (commit `dfc6f83`): model moved to device before the batch-size finder (was the cause of the BFloat16/bias crash).
- `tests/resume_test.py`: CPU gate, 2 epochs straight == 1 epoch + resume. Prints RESUME PASS.
- `notebooks/version3/xla-training.ipynb` = stage 1 (`END_EPOCH=4`); new `xla-training-stage2.ipynb` = stage 2 (`END_EPOCH=8`, `RESUME_FROM=...`). Everything else identical to v3 A1.

## You must do
1. Push `integration/loss-backbone`; re-run `xla-setup` so `bfree_src` has the new `train_lora.py`.
2. Run `xla-training` (stage 1, epochs 1–4) → Save & Run All. Note the `auto batch size = N` line.
3. Create stage-2 notebook from `xla-training-stage2.ipynb`, Add Input → Your Work → the stage-1 notebook output. Check `RESUME_FROM` slug (default `/kaggle/input/notebooks/baopv051/xla-training/resume_state.pth`). Run.
4. Eval notebook: `OURS_CKPT` must point at the **stage-2** output.

## Caveats
- Use the same GPU type for both stages (batch size comes from stage 1; a smaller GPU would OOM).
- With `num_workers>0`, DataLoader worker augmentation RNG restarts at stage 2 (the loader/shuffle order and main-process RNG are restored). Per-epoch metrics are therefore statistically, not bit-wise, equal to an uninterrupted run. `tests/resume_test.py` uses `num_workers=0` where equality is exact.
- `_best.pth` from stage 2 only exists if it beats the stage-1 best; `--select last` makes this moot.
- Each stage must itself fit in 12h; if stage 1 (4 epochs) does not, set `END_EPOCH` lower (1–3) and chain more stages — `resume_state.pth` is saved every epoch. Unverified: whether Kaggle keeps `/kaggle/working` when a commit run is killed at the time limit — do not rely on it.

## Paste back
- stage 1: `auto batch size` line, last cell, wall time. Stage 2: first lines of `train_log.txt` (`[resume] ...` with lr ≈ 5.0e-5), last cell, `train_log.csv` (should hold epochs 1–8).
