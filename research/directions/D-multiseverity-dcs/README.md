# D-multiseverity-dcs — Multi-Severity DCS

Source: `source/Multi-Severity DCS AI Detection.docx` (AI-generated report, 106 equation images, raw `[cite]` artefacts, off-topic URLs). **All docx numbers are hypothetical; do not cite.** (Its 91.4 vs 62.9 degraded-avg table has no experiment behind it; intro and table disagree on the DINOv2 baseline, 86.2 vs 98.4.)

## 1. Motivation
Low-pass degradations (JPEG / blur / resize / noise) erase HF artefacts. One random degraded view per step (DCPT, DDA, GF-DCS) may let features drift across severity and fit an "average" severity. GlobalForge itself trains with a single compound view; multi-severity appears only at eval (RealDeg-Bench).

## 2. Method actually implemented (subset of the docx)
- K=4 severity bins, ranges in `B-Free-loss-backbone/code/configs/bfree_dcs.yaml :: multi_severity` (user-approved 2026-09-30): JPEG Q [85-95]/[65-80]/[40-60]/[20-35]; blur σ [0.5-1.0]/[1.2-2.0]/[2.2-3.0]/[3.2-4.5]; resize [.75-.90]/[.50-.70]/[.30-.45]/[.15-.25]; noise σ (0-255 scale, **assumed unit**) [2-5]/[6-12]/[13-20]/[21-30].
- Loss: `L = Σ_k w_k · InfoNCE(z, z_k) / Σ_k w_k`, each term the unchanged faithful symmetric InfoNCE (in-batch negatives, **not** class-aware, not SupCon). `multi_severity.enabled=false` (default) = v2 path, bit-identical (test vs pre-change reference).
- `compose`: `single_op` (default; one random op per view) or `compound` (jpeg→blur→resize→noise). The docx is ambiguous; single_op chosen, compound available.
- Not tested (undefined values or conflict with locked decisions): docx γ, δ(s), ω(s), Severity(), α, L_orth, L_deg head, MoCo queue 65,536, ViT-L (decision 5), two-stage training, class-aware negatives.

## 3. Known caveats
- Bins overlap RealDeg-Bench eval levels → RealDeg is **in-distribution robustness**, not held-out (same for v2 lifecycle pipeline).
- K views ⇒ (1+K)× forward memory/time per step; batch size found automatically.

## 4. Hypotheses (success criteria fixed BEFORE any run)
Single run: **A1** (K=4 views, uniform weights, seed 42). Baseline = **v2 numbers already saved** in `notebooks/version2` eval outputs (no re-run). Limitation: one seed and no baseline re-run ⇒ seed-to-seed spread is unknown; conclude only by the fixed thresholds below and report the result as a single-seed observation, never as a proven gain.

- **H1** MS-DCS improves robustness at high severity. Metric: mean bAcc@0.5 over RealDeg jpeg/blur/resize/noise + compound 3–5 steps (needs RealDeg enabled in the harness — currently OFF; otherwise `TODO: pending run`). Predicts ≥ +3.0 pts vs v2. Guard: clean AI-GenBench ALL AUC drop ≤ 0.005 (v2: 0.992977). **Refuted if** gain < +1.0, or clean AUC drops > 0.01.
- **H2** MS-DCS improves social-media transfer. Metric: mean AUC over WildRF fb/reddit/twitter + bfree_viral (v2: .905/.951/.920/.850). Predicts ≥ +0.02. **Refuted if** gain < +0.005.
- (Dropped: multi-view-vs-distribution test — needs a 1-view ablation that is not run.)

## 5. Run
| id | config | seed |
|---|---|---|
| A1 | `enabled:true`, K=4, uniform weights, all bins/step | 42 |

Model selection: last-epoch checkpoint (val split saturates ~0.999). No test-set selection.

## 6. Status
Code on `integration/loss-backbone` (P-Bao), CPU-tested. No training/eval results yet: `TODO: pending run`.
