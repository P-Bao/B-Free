"""Sinh các hình "đối thủ kém hơn Ours bao nhiêu" cho slide.

Nguồn số liệu: outputs/5_eval_final_merged (notebook xla-eval-final-merged).
Chạy:  python3 docs/slides/make_gap_figs.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs" / "5_eval_final_merged"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

LABEL = {
    "bfree_official": "B-Free (official)", "dda": "DDA",
    "commfor384": "CommFor (ViT-S/16-384)", "globalforge_dinov3": "GlobalForge (DINOv3)",
    "gapl": "GAPL", "cospy": "Co-Spy", "effort": "Effort",
    "c2p_clip": "C2P-CLIP", "safe": "SAFE",
}
ORDER = list(LABEL)  # theo hạng AUC trung bình
BENCH = {"itw_sm": "ITW-SM", "wildrf": "WildRF", "bfree_viral_images": "B-Free viral"}

RED, GREEN, GREY, NAVY = "#C0392B", "#1E8449", "#95A5A6", "#12326B"

# ---------------------------------------------------------------- data
boot = pd.read_csv(SRC / "paired_bootstrap_auc.csv")
boot = boot[boot.level == "pooled"]
# gap = ref - ours  (âm = đối thủ kém hơn)
gap = np.zeros((len(ORDER), len(BENCH)))
lo = np.zeros_like(gap); hi = np.zeros_like(gap); verdict = np.empty(gap.shape, dtype=object)
for i, m in enumerate(ORDER):
    for j, b in enumerate(BENCH):
        r = boot[(boot.ref == m) & (boot.benchmark == b)].iloc[0]
        gap[i, j] = -r.d_auc
        lo[i, j], hi[i, j] = -r.d_hi, -r.d_lo
        verdict[i, j] = r.verdict
avg_gap = gap.mean(axis=1)

def ours_val(s):
    return float(str(s).split("±")[0])

eer = pd.read_csv(SRC / "slide_eer_pooled.csv", index_col=0)
eer_ours = ours_val(eer.iloc[0]["Avg"])
eer_ref = {LABEL[m]: float(eer.loc[LABEL[m], "Avg"]) for m in ORDER}

auc = pd.read_csv(SRC / "slide_auc_pooled.csv", index_col=0)

names = [LABEL[m] for m in ORDER]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- 1. heatmap gap
fig, ax = plt.subplots(figsize=(9.5, 5.6))
data = np.column_stack([gap, avg_gap])
cols = list(BENCH.values()) + ["Trung bình"]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("rg", ["#922B21", "#F5B7B1", "#FFFFFF", "#58D68D"])
norm = matplotlib.colors.TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.015)
ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        if j < 3 and verdict[i, j] == "n.s.":
            txt = f"{v:+.3f}\n≈ (n.s.)"
        else:
            txt = f"{v:+.3f} {'▲' if v > 0 else '▼'}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                color="white" if v < -0.25 else "black", fontweight="bold" if j == 3 else None)
ax.set_xticks(range(len(cols)), cols, fontweight="bold")
ax.set_yticks(range(len(names)), names)
ax.axvline(2.5, color="black", lw=1.5)
ax.set_title("Đối thủ kém Ours S7 bao nhiêu AUC?  (ΔAUC = Đối thủ − Ours, pooled)\n"
             "đỏ = kém hơn Ours · xanh = hơn Ours · ≈ = không khác biệt có ý nghĩa (95% CI)",
             fontsize=12, fontweight="bold")
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "gap_heatmap_auc.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- 2. bar avg gap
idx = np.argsort(avg_gap)[::-1]  # gần 0 nhất ở trên
fig, ax = plt.subplots(figsize=(9, 5.2))
y = np.arange(len(idx))
ax.barh(y, avg_gap[idx], color=RED, height=0.6)
for k, i in enumerate(idx):
    ax.text(avg_gap[i] - 0.006, k, f"{avg_gap[i]:+.3f}", ha="right", va="center",
            fontsize=10, fontweight="bold", color=RED)
ax.set_yticks(y, [names[i] for i in idx])
ax.invert_yaxis()
ax.axvline(0, color=NAVY, lw=3)
ax.text(0.004, 8.3, "Ours S7\n= 0,976\n(mốc 0)", color=NAVY, fontweight="bold", va="center", fontsize=10)
ax.set_xlim(-0.5, 0.075)
ax.set_xlabel("AUC trung bình 3 benchmark của đối thủ − AUC của Ours S7")
ax.set_title("Tất cả 9 đối thủ đều nằm bên trái mốc Ours (kém hơn)", fontweight="bold")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "gap_bars_avg_auc.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- 3. EER multiplier
mult = {k: v / eer_ours for k, v in eer_ref.items()}
ks = sorted(mult, key=mult.get)
fig, ax = plt.subplots(figsize=(9, 5.2))
y = np.arange(len(ks))
ax.barh(y, [mult[k] for k in ks], color=RED, height=0.6)
for k_i, k in enumerate(ks):
    ax.text(mult[k] + 0.08, k_i, f"×{mult[k]:.1f}  (EER {eer_ref[k]:.3f})", va="center", fontsize=10)
ax.axvline(1, color=NAVY, lw=3)
ax.text(1.05, len(ks) - 0.4, f"Ours S7: EER = {eer_ours:.3f} (×1,0)", color=NAVY, fontweight="bold", fontsize=10)
ax.set_yticks(y, ks)
ax.set_xlim(0, 8.8)
ax.set_xlabel("Tỷ lệ lỗi EER của đối thủ / EER của Ours (trung bình 3 benchmark)")
ax.set_title("Đối thủ mắc lỗi nhiều gấp bao nhiêu lần Ours?", fontweight="bold")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "eer_multiplier.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- 4. bảng slide đã sửa
def fmt_gap(v, ver):
    if ver == "n.s.":
        return f"{v:+.3f} ≈"
    return f"{v:+.3f} {'▲' if v > 0 else '▼'}"

header = ["Hạng", "Phương pháp", "ITW-SM", "WildRF", "B-Free viral", "AUC TB", "EER TB (↓)",
          "Kém Ours (AUC TB)"]
rows = [["1", "Ours S7 · 3 seeds", "0,964", "0,991", "0,974", "0,976", "0,067", "Mốc tham chiếu"]]
for i, m in enumerate(ORDER):
    rows.append([str(i + 2), names[i]] + [fmt_gap(gap[i, j], verdict[i, j]) for j in range(3)]
                + [auc.loc[names[i], "Avg"].replace(".", ","), f"{eer_ref[names[i]]:.3f}".replace(".", ","),
                   f"{avg_gap[i]:+.3f}"])
fig, ax = plt.subplots(figsize=(13, 5.8))
ax.axis("off")
tb = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center",
              colWidths=[0.06, 0.22, 0.12, 0.12, 0.12, 0.09, 0.1, 0.15])
tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1, 1.75)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#D5D8DC")
    if r == 0:
        cell.set_facecolor("#0B1F3A"); cell.get_text().set_color("white"); cell.get_text().set_fontweight("bold")
        continue
    if r == 1:
        cell.set_facecolor("#E8F8F0"); cell.get_text().set_color(GREEN); cell.get_text().set_fontweight("bold")
        continue
    txt = cell.get_text().get_text()
    if "▼" in txt or (c == 7 and txt.startswith("-")):
        cell.get_text().set_color(RED)
    elif "▲" in txt:
        cell.get_text().set_color(GREEN); cell.get_text().set_fontweight("bold")
    elif "≈" in txt:
        cell.get_text().set_color(GREY)
    if c == 7:
        cell.get_text().set_fontweight("bold")
ax.set_title("So sánh với 9 đối thủ — cột benchmark ghi ΔAUC = Đối thủ − Ours (đỏ ▼ = kém hơn Ours)",
             fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "slide_table_gap.png", dpi=300)
plt.close(fig)

print("Saved to", OUT)
for i, n in enumerate(names):
    print(f"{n:25s}", " ".join(f"{gap[i, j]:+.3f}" for j in range(3)), f"avg {avg_gap[i]:+.3f}",
          f"EERx {mult[n]:.2f}")
