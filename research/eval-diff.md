# Harness diff (read-only analysis, 2026-09-30)

All harnesses share one codebase. Differences: `OURS_CKPT` (v1 = phase-2 best, v2 = v2 best); `MODEL_ID`; GF adds `pip uninstall torchao`. Per-model behaviour lives in `ModelWrapper`:

| | ours | bfree_official | globalforge_dinov3 |
|---|---|---|---|
| resize | bicubic upscale short side to 504 if <504 | same | same |
| crop | up to 9 × 504 crops covering the full image | 1 center 504 (5-crop wrapper → 5 identical crops) | 1 center 504 |
| norm | ImageNet | ImageNet | inside model |
| score | softmax[:,1], mean over crops | sigmoid(l) | softmax[:,1] |
| dtype | bf16 | bf16 | bf16 (official: fp32) |

Shared: AI-GenBench `validation` (1,500 real pool + 150 fake/generator, 36 generators, rng 42); WildRF test (fb 320 / reddit 1500 / twitter 683); bfree_viral 785. SynthWildX fails in every harness. RealDeg OFF. Metrics: AUC, bAcc@0.5, bAcc@calib, ECE (15 bins), balanced NLL. Calibrated threshold = Youden J fitted on the AI-GenBench test scores, reused on WildRF/viral.

## Unfairness flags
- U1 ours trained on AI-GenBench `train` (13 generators + 46.8k real), evaluated on its `validation`; baselines zero-shot.
- U2 9-crop full coverage (ours) vs 1 center crop (baselines); also breaks B-Free's official full-image protocol.
- U3 GF off-protocol (official: 224 crop, resize-to-1296, JPEG q100 re-encode, fp32). Real mean p=0.4965, bAcc@0.5 stuck 76.10 → only AUC roughly comparable.
- U4 upscaling matches ours' training, not baselines'.
- U5 calibrated threshold tuned on test set: optimistic for every model.
- U6 label smoothing floors ours' ECE.
- U7 v2 degradations = RealDeg eval operator set → future RealDeg comparison biased.

`test/01_eval_harness_template` is a different protocol (Synthbuster/GenImage, RealDeg on, no calibration), never run.

Planned fix (P4, NOT yet applied; needs user OK per CLAUDE.md before touching metric/split/preprocessing): H-a seen/unseen aggregates; H-b unified crop policy; H-c GF official preprocessing fp32; H-d RealDeg on with per-image seeds; H-e SynthWildX loader. All baselines then "must re-run".
