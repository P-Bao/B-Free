# D-recipe-v5 — B-Free recipe ladder (preregistered)

Direction: rebuild the training recipe (aug / pairing / CLS head / CE ls=0 / val aug), then
test whether the architectural modules (LIB/GSR, DCS) add anything on top of it.
Reference: `AGENT_PLAN_V5.md` §4–§7. Harness: `research/eval-v2-proposal.md` (PENDING).
Nothing here is a result yet; all numbers below are **predictions and gates**.

## 0. Rungs (one change per rung)

| Run | recipe | status in code |
|---|---|---|
| S0 | `bfree_official`, no training | harness only (`notebooks/version4/xla-eval-harness-b-free.ipynb`) |
| S1 | DINOv2 + LoRA r16 + **CLS head** + CE on clean images, `label_smoothing=0`, no LIB/GSR/DCS, **no AIGB**, paired real/fake, `val_aug` | ready (`xla-training-v4.ipynb`, `RUN_ID=S1`) |
| S2 | S1 + online aug `bfree` (Appendix A parameters are PLACEHOLDERS) | ready (`RUN_ID=S2`) |
| S2b | S2 with aug `light` / `strong` | ready (`S2b-light`, `S2b-strong`) |
| S3 | S2 + AI-GenBench 25% (`other_frac=0.25`) | ready (`RUN_ID=S3`) |
| S4 | S3 but full fine-tune | **BLOCKED**: `train_lora.py` is LoRA-only; `train_full_ft.py` has not been audited against the v4 dataset/loss/val pipeline. Do not start S4 from these notebooks. |
| S5 | S3 + LIB/GSR zero-init + `window=7` + `ClsPlusPatchHead(gamma=0)` | ready (`RUN_ID=S5`) |
| S6 | S3 + class-aware DCS (hard negatives = opposite-class pairs) | ready (`RUN_ID=S6`) |
| S7 | best rung, 6 epochs, seeds 42/43/44 | ready (`RUN_ID=S7`, set `S7_FROM`) |
| W1 | warm-start from `bfree_official` | code path ready (`--init_from bfree --bfree_ckpt ...`), needs the real checkpoint and the step-0 test |

Cost per rung at 504 px, batch 64 (auto), paired views, ~50k stems: ≈0.7–0.8 h/epoch,
2 epochs/rung; each notebook run is capped at `--time_budget_hours 10` (stops after a
completed epoch, saves `resume_state.pth`), so the 12 h Kaggle limit is never reached.

## 1. Implementation notes and deviations (read before running)

1. **Head**: `head_mode=cls` uses timm `forward_head` (CLS token → `fc_norm` → `head_drop`
   → `head`). `cls_plus_patch` adds `gamma * head(proj(patch tokens))` with `gamma=0`.
   `gamma` is logged every epoch in `train_log.csv`.
2. **S5 zero-init is implemented as a gate**, not as zero weights: the GSR adapter output is
   `tokens + gate * proj(tokens)` with `gate=0`; LIB uses `beta=-10` (alpha≈4.5e-5).
   At step 0 this equals "modules off" (test below). Difference from the plan text: with a
   zero gate, `proj` receives zero gradient until `gate` leaves 0, whereas zero-init weights
   would train `proj` immediately. This is recorded as a deviation, not hidden.
3. **S6 second view** is the existing lifecycle-degraded view (`degraded_view`), not a second
   online-augmented view. Online aug applies to the clean view; the DCS pair is therefore
   (aug'ed clean, lifecycle-degraded). `dcs_mode=class_aware` excludes same-class positives
   and returns 0 when no opposite-class negative exists.
4. **Pairing** is per stem with the same crop/flip/epoch seed (`PairedStemDataset`,
   `PairedBatchSampler`), batches stay uniform 4-D so AIGB can be mixed at `other_frac`.
   AIGB items are never paired.
5. **Checkpoint selection**: `--select last` (default). Selecting by WildRF `val` AUC is
   *not* implemented in the training notebook because it requires attaching WildRF during
   training; the plan's §1.5 selection rule stays an open item.
6. **Validation** is the repository's md5 split with seeded `val_aug`, not an OOD set.
7. **Online-aug parameters are placeholders** pending Appendix A of arXiv 2412.17671; S2b
   exists to measure that sensitivity.
8. **Label smoothing = 0** is written into the per-run `run_config.yaml` by the recipe cell
   (not passed on the CLI), like every other recipe knob (`head_mode`, `use_*`, `gsr_*`,
   `lib_*`, `dcs_*`, `paired`, `other_frac`, `val_aug`, `online_aug`). The CLI carries only
   `lr`, LoRA/val-split/degradation knobs, workers, seed and the stage/budget flags.
   Regression test: `tests/test_yaml_defaults.py` (YAML must beat the argparse default).
9. **DCS feature in `cls` head mode** is the CLS feature (`fc_norm(head_drop(cls))`), i.e. the
   same representation the classifier uses; `meanpool` keeps the v2 patch-mean feature.

## 2. Local test evidence (CPU)

| check | command | result |
|---|---|---|
| overfit gate, both DCS modes | `tests/overfit_test.py` | `GATE PASS` (single + multi_severity) |
| stage resume, csv equality | `tests/resume_test.py` | `RESUME PASS` |
| DCS legacy bit-identity vs pre-change reference | `tests/test_dcs_loss.py` | ALL PASS |
| multi-severity contract | `tests/test_multi_severity.py` | ALL PASS |
| backbone shapes | `tests/test_backbone_shape.py` | ALL PASS |
| `cls_plus_patch` gamma=0 ≡ CLS head (bitwise logits) | `tests/test_heads_init.py` | pass |
| GSR gate=0 ≡ no GSR (bitwise logits) | `tests/test_heads_init.py` | pass |
| LIB beta=-10 ≈ identity; beta=-0.85 is not | `tests/test_heads_init.py` | pass |
| class-aware InfoNCE finite / permutation-invariant / single-class=0 | `tests/test_class_aware_dcs.py` | pass |
| online aug size/mode/determinism | `tests/test_online_aug.py` | pass |
| paired dataset + sampler mixing | `tests/test_paired_dataset.py` | pass |
| B-Free checkpoint remap + 1→2 logit head (synthetic) | `tests/test_bfree_init.py` | pass |
| 5-crop ≡ `Wrapper5crops` patch-for-patch | `tests/test_eval_protocol.py` | pass |
| YAML beats argparse defaults; CLI still wins | `tests/test_yaml_defaults.py` | pass |
| end-to-end paired+aug+class-aware CPU smoke | `tests/smoke_paired.py` | `SMOKE PASS` |

Not runnable locally: the real-B-Free step-0 equivalence (no checkpoint/config in the repo),
W1 with real weights, parquet writing (no `pyarrow`), Arrow loaders (no `datasets`).

## 3. Preregistered gates (thresholds from plan §6)

| Gate | condition | if it fails |
|---|---|---|
| **G0** | 5-crop equivalence on the real B-Free model (`max|Δlogit| < 1e-3`) + two identical harness runs | stop; fix the harness |
| **G1** | S2 − S1: mean ΔAUC over the 4 OOD sets ≥ **+0.02**, paired-bootstrap 95% lower bound > 0 | decision tree below |
| **G2** | S3 vs S2: AIGB **unseen** AUC up, OOD mean down by at most 0.01 | drop AIGB to 10% |
| **G3** | S5 or S6 vs best of {S2,S3,S4}: mean ΔAUC OOD ≥ **+0.005**, CI lower > 0, no benchmark down > 0.01 | present modules as negative/neutral |
| **G4** | S7 vs S0: honest report with CI + 3 seeds | remaining-error analysis |
| **H1** | RealDeg with **held-out** operators: bAcc at levels 3–4 and compound 3–5 improves ≥ **3 points** vs both S0 and S2 | no robustness claim |

Paired bootstrap = 2000 resamples, same images, protocol v2.

## 4. Hypotheses (falsifiable, written before the runs)

- **H-A (RC1, main).** Compression/resize augmentation in the CE path is the missing piece:
  S2−S1 ≥ +0.02 AUC on the 4 OOD sets, with the largest gains on Facebook/Twitter/viral
  (the compressed sets) and the smallest on Reddit.
- **H-B (data).** AIGB raises AIGB-unseen more than it costs OOD; if OOD drops > 0.01,
  H-B is false at 25%.
- **H-C (architecture).** LIB/GSR + `cls_plus_patch` (S5) add nothing once the recipe is
  fixed; predicted ΔAUC within ±0.005. If G3 passes, the modules are worth a separate claim.
- **H-D (loss).** Class-aware DCS (S6) reduces `train_dcs` without moving AUC; any gain must
  show in the CI, not only in the training loss.
- **H-E (capacity).** LoRA is not the bottleneck only if S2 is already close to S0
  (≤0.02 mean AUC); otherwise S4/W1 must be tested before blaming the recipe.

## 5. Run order and artifacts

1. Phase 0: three v4 harnesses × 4 models (ours v1, ours v2, B-Free, GlobalForge) → G0.
   Model identity for "ours v1/v2" still needs the user's answer (plan §0.1).
2. S1 → S2 (G1). If G1 fails, S2b (aug strength) before any other rung.
3. S3 (G2) → S5, S6 (G3).
4. S4 only after auditing `train_full_ft.py`; W1 only if H-E requires it.
5. S7 × seeds 42/43/44 (G4, H1), then `xla-report-v4.ipynb` for paired deltas.

Every run writes: `train_log.csv` (with `gamma`), `run_meta.json` (git sha + ckpt sha256),
`run_config.yaml`. The harness writes `eval_results_{model}.csv` and
`scores_{model}.parquet` with a matching provenance hash. A sha mismatch invalidates the
comparison.

## 6. Open items

- Appendix A exact augmentation parameters (blocker for calling S2 "B-Free recipe").
- `train_full_ft.py` audit (S4).
- GlobalForge official preprocessing (H9) and held-out RealDeg operators (H10).
- Checkpoint selection by WildRF `val` (plan §1.5).
- Identity of the checkpoint behind each published "v1" number (plan §0.1).
