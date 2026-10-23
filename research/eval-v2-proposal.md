# Eval protocol v2 — proposal (Phase 0, 🔒 PENDING USER APPROVAL)

Status: **not adopted for any published number**. The protocol is implemented only in new
files (`notebooks/version4/*`, `code/eval_protocol.py`). `notebooks/version3/*`, the root
call `notebooks/xla-eval-harness-*.ipynb`, and every previously reported number are untouched.

Source of the requirements: `AGENT_PLAN_V5.md` §3 (Phase 0) and §6 (gates), plus the
unfairness list U1–U7 in `research/eval-diff.md`.

## 1. What changes, and why

| ID | Old behaviour (v1/v2/v3 harness) | v2 behaviour | Where |
|---|---|---|---|
| H1 | B-Free: 1 PIL center crop + bicubic upscale, then the official `Wrapper5crops` sees 504 px ⇒ 5 identical crops | same 5 patch-grid crops for every model, no resize, `replicate_wrap` for small images | `code/eval_protocol.py::bfree_five_crops` |
| H2 | `ours` up to 9 crops covering the image; baselines 1 | all models get exactly 5 crops; optional 1-crop variant can be derived from the parquet | `ModelWrapper.score` |
| H3 | ours averages probabilities, B-Free averages logits | per-image score = mean **logit** over the 5 crops; `p` only inside metrics | `eval_protocol.mean_logit` |
| H4 | hand-written `binary_ece`, not class-balanced | `balanced_ece_binary`, `balanced_nll_binary`, `pd_at_far`, `calculate_eer2` | inlined copies of `utils/dmetrics.py` in `eval_protocol.py` |
| H5 | Youden threshold fitted on the AIGB test pool, reused on WildRF/viral | primary = **bAcc@0.5** (`logit > 0`); no threshold fitted on test | metrics cell |
| H6 | no per-image scores → no CI | `scores_{model}.parquet` (id, label, logit, score, benchmark, generator, condition) + bootstrap AUC CI + paired bootstrap | `write_scores_parquet`, `bootstrap_auc_ci`, `paired_bootstrap_diff` |
| H7 | SynthWildX "neither a Dataset nor a DatasetDict directory" | `resolve_arrow_dir` finds the dir holding `dataset_dict.json` / `dataset_info.json` / `*.arrow` | `eval_protocol.resolve_arrow_dir` |
| H8 | 1 real pool shared by all generators, no grouping rule | pooled + per-generator rows both written; pooling rule documented in the notebook | `run_ai_genbench` |
| H9 | GlobalForge off-protocol (224 crop, fp32, JPEG re-encode) | not changed yet; column `protocol_note` absent, so GlobalForge rows remain **AUC-only** | open item |
| H10 | RealDeg operators identical to training operators | not changed yet; v4 only logs the RealDeg-style rows in a separate condition column | open item |
| H11 | `forward_fn` returned `cat([-l, l])`, then `softmax(...)[:,1]` ⇒ `p = sigmoid(2l)` | `forward_fn` returns the raw logit `l` (B-Free bypasses `Wrapper5crops.forward` and calls `patch_embed → model`); ours/GF return `l1-l0` | `_single_logit`, `_two_logit_diff` |

H1–H3 are believed to **under**-rate B-Free; if true, the current gap is a lower bound.

## 2. Sampling and splits (v2)

| item | v1/v2/v3 | v4 |
|---|---|---|
| AI-GenBench split | `validation` | `validation` (unchanged) |
| fake per generator | 150 | **500** |
| real pool | 1500 | 1500 |
| generators | 36 | 36 (+ `seen` column for the 13 trained by `ours`) |
| WildRF | `test` only | `test` scored; `val` named as dev, **not** used to fit any threshold |
| viral | 785 | 785 |
| seed | 42 | 42 |
| calibration | Youden on AIGB test | none |
| run budget | 10.5 h | 10.0 h |

500 fakes/generator × 36 generators ≈ 18k fake + 1.5k real images per model per run; 5 crops
⇒ ~100k forward passes at 504 px, ≈3.3× the v3 fake-image count. This is the main runtime
risk (see §5).

## 3. Metrics (v2, primary and secondary)

Primary: **AUC** (per generator and macro over generators) with 95% image-bootstrap CI.
Primary thresholded number: **balanced accuracy @ 0.5** on the logit.

Secondary: balanced ECE (15 bins), balanced NLL, Pd@10%FAR, EER, FPR@95%TPR, TPR/FPR@0.5.
`bAcc@calib` and `thr_calib` are **removed** from the v4 outputs.

Paired model comparison: `paired_bootstrap_diff(y, z_a, z_b, n_boot=2000)` on images present
in both parquet files (same ids only).

## 4. Verification done locally (CPU)

- `tests/test_eval_protocol.py::test_five_crops_match_wrapper5crops` — the 5 crops reproduce
  a conv-stride-14 reference of `Wrapper5crops.forward` patch-for-patch on 700×520, 512²,
  300×260, 504² (max |Δ| < 1e-5 in fp32). The reference was corrected once: `replicate_wrap`
  is all-or-nothing and truncates, it does not center the tiled canvas.
- Metrics/bootstrap degenerate-label paths return NaN AUC without raising.
- `tests/test_bfree_init.py` covers the W1 checkpoint remap + 1→2 logit head in a synthetic
  wrapper-format checkpoint (no real B-Free weights in the repo).
- **Not yet run:** step 0.3 with the real B-Free model (`max|Δlogit| < 1e-3`), and the
  2× determinism run. Both are USER-KAGGLE steps and are **gate G0**; without them the v2
  table must not be published.

## 5. Risks / costs

| risk | mitigation |
|---|---|
| 500 fakes/gen + 5 crops exceeds 10 h | run one model per notebook session; AIGB is checkpointed per generator (`save_partial`), so a rerun resumes at the next generator |
| new bootstrap CI numbers differ from v3 even for the same model | expected; v2 defines a new frozen table, old numbers stay labelled v3 |
| 5-crop reference mismatch on real weights | G0 step 0.3 on Kaggle before any comparison |
| `pyarrow`/`datasets` absent in the local CPU env | parquet and Arrow loaders are only exercised on Kaggle; `resolve_arrow_dir` has a local unit test with dummy files |
| GlobalForge/RealDeg remain off-protocol (H9/H10) | their rows are reported as AUC-only / separate conditions; no cross-model bAcc claim |

## 6. Rollback

Delete `notebooks/version4/` and `code/eval_protocol.py`; nothing else imports the module
(`train_lora.py` does not). No change was made to any published harness copy or CSV.

## 7. Approval checklist (user)

- [ ] accept 5-crop + mean-logit as the single protocol for all models
- [ ] accept bAcc@0.5 as the only thresholded headline number (drop `bAcc@calib`)
- [ ] accept 500 fakes/generator and the runtime cost
- [ ] accept "freeze the v2 table only after G0 passes twice"
- [ ] decide H9 (rebuild GlobalForge preprocessing) and H10 (held-out degradation operators)
      before or after the S-ladder
