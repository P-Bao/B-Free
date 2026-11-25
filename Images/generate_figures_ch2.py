"""Generate high-resolution (300 DPI) Chapter II figures for the report."""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["mathtext.fontset"] = "dejavusans"

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "Images"
DEMO_IMG = ROOT / "code" / "demo_images" / "img0000.png"


def _load_square_sample(size: int = 256) -> np.ndarray:
    """Load and center-crop demo image into a clean square RGB array."""
    im = Image.open(DEMO_IMG).convert("RGB")
    w, h = im.size
    # Focus slightly toward the upper-center statue region
    side = min(w, h)
    top = max(0, int((h - side) * 0.25))
    left = (w - side) // 2
    crop = im.crop((left, top, left + side, top + side))
    crop = crop.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(crop, dtype=np.float32) / 255.0


def _draw_arrow(ax, p1, p2, color="#4A5568", lw=1.6, ls="-", mutation_scale=13):
    ax.annotate(
        "",
        xy=p2,
        xytext=p1,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=ls,
            mutation_scale=mutation_scale,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=4,
    )


def _draw_trapezoid(ax, x_left, x_right, y_center, h_left, h_right, fc, ec, lw=1.6):
    verts = [
        (x_left, y_center - h_left / 2.0),
        (x_right, y_center - h_right / 2.0),
        (x_right, y_center + h_right / 2.0),
        (x_left, y_center + h_left / 2.0),
    ]
    poly = patches.Polygon(verts, closed=True, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(poly)


def _draw_rect(ax, x, y, w, h, fc, ec, lw=1.6, radius=0.0, ls="-", zorder=3):
    if radius > 0:
        box = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle=ls,
            zorder=zorder,
        )
    else:
        box = patches.Rectangle(
            (x, y),
            w,
            h,
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle=ls,
            zorder=zorder,
        )
    ax.add_patch(box)
    return box


def draw_hinh2_1(save_path: Path):
    """Hình 2.1: So sánh cơ chế hoạt động của bốn họ mô hình sinh ảnh."""
    fig, ax = plt.subplots(figsize=(10, 6.6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    # Palette
    c_green_fc, c_green_ec = "#C5E0B4", "#548235"
    c_blue_fc, c_blue_ec = "#BDD7EE", "#2F5597"
    c_red_fc, c_red_ec = "#F4B183", "#C55A11"
    c_purple_fc, c_purple_ec = "#D9C2E9", "#7030A0"
    c_yellow_fc, c_yellow_ec = "#FFE699", "#BF9000"
    c_orange_fc, c_orange_ec = "#F8CBAD", "#C55A11"

    row_ys = [5.55, 3.95, 2.35, 0.78]

    # Left-side descriptions
    ax.text(
        1.55, row_ys[0],
        r"$\mathbf{GAN:}$ Adversarial" + "\ntraining",
        ha="center", va="center", fontsize=11, color="#1A202C", linespacing=1.35,
    )
    ax.text(
        1.55, row_ys[1],
        r"$\mathbf{VAE:}$ maximize" + "\nvariational lower bound",
        ha="center", va="center", fontsize=11, color="#1A202C", linespacing=1.35,
    )
    ax.text(
        1.55, row_ys[2],
        r"$\mathbf{Flow\text{-}based\ models:}$" + "\nInvertible transform of\ndistributions",
        ha="center", va="center", fontsize=10.5, color="#1A202C", linespacing=1.35,
    )
    ax.text(
        1.55, row_ys[3],
        r"$\mathbf{Diffusion\ models:}$" + "\nGradually add Gaussian\nnoise and then reverse",
        ha="center", va="center", fontsize=10.5, color="#1A202C", linespacing=1.35,
    )

    # Subtle row separators
    for sep_y in [4.75, 3.15, 1.56]:
        ax.plot([0.3, 9.7], [sep_y, sep_y], color="#E2E8F0", lw=0.9, ls="--", zorder=1)

    # --- ROW 1: GAN ---
    y = row_ys[0]
    _draw_rect(ax, 3.05, y - 0.58, 0.36, 1.16, c_blue_fc, c_blue_ec)
    ax.text(3.23, y, r"$\mathbf{x}'$", ha="center", va="center", fontsize=11.5, fontweight="bold")
    _draw_rect(ax, 3.58, y - 0.58, 0.36, 1.16, c_green_fc, c_green_ec)
    ax.text(3.76, y, r"$\mathbf{x}$", ha="center", va="center", fontsize=11.5, fontweight="bold")
    _draw_arrow(ax, (3.94, y), (4.35, y))
    _draw_trapezoid(ax, 4.35, 5.55, y, 1.18, 0.44, c_green_fc, c_green_ec)
    ax.text(4.88, y, "Discriminator\n" + r"$D(\mathbf{x})$", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#1A202C")
    _draw_arrow(ax, (5.55, y), (5.88, y))
    circ = patches.Circle((6.16, y), 0.28, facecolor=c_purple_fc, edgecolor=c_purple_ec, linewidth=1.6, zorder=3)
    ax.add_patch(circ)
    ax.text(6.16, y, "0/1", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1A202C")

    _draw_rect(ax, 6.68, y - 0.23, 0.42, 0.46, c_red_fc, c_red_ec)
    ax.text(6.89, y, r"$\mathbf{z}$", ha="center", va="center", fontsize=11.5, fontweight="bold")
    _draw_arrow(ax, (7.10, y), (7.48, y))
    _draw_trapezoid(ax, 7.48, 8.68, y, 0.44, 1.18, c_blue_fc, c_blue_ec)
    ax.text(8.16, y, "Generator\n" + r"$G(\mathbf{z})$", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#1A202C")
    _draw_arrow(ax, (8.68, y), (9.12, y))
    _draw_rect(ax, 9.12, y - 0.58, 0.36, 1.16, c_blue_fc, c_blue_ec)
    ax.text(9.30, y, r"$\mathbf{x}'$", ha="center", va="center", fontsize=11.5, fontweight="bold")

    # --- ROW 2: VAE ---
    y = row_ys[1]
    _draw_rect(ax, 3.58, y - 0.58, 0.36, 1.16, c_green_fc, c_green_ec)
    ax.text(3.76, y, r"$\mathbf{x}$", ha="center", va="center", fontsize=11.5, fontweight="bold")
    _draw_arrow(ax, (3.94, y), (4.35, y))
    _draw_trapezoid(ax, 4.35, 5.55, y, 1.18, 0.44, c_green_fc, c_green_ec)
    ax.text(4.88, y, "Encoder\n" + r"$q_\phi(\mathbf{z}|\mathbf{x})$", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#1A202C")
    _draw_arrow(ax, (5.55, y), (6.30, y))
    _draw_rect(ax, 6.30, y - 0.23, 0.42, 0.46, c_red_fc, c_red_ec)
    ax.text(6.51, y, r"$\mathbf{z}$", ha="center", va="center", fontsize=11.5, fontweight="bold")
    _draw_arrow(ax, (6.72, y), (7.48, y))
    _draw_trapezoid(ax, 7.48, 8.68, y, 0.44, 1.18, c_blue_fc, c_blue_ec)
    ax.text(8.16, y, "Decoder\n" + r"$p_\theta(\mathbf{x}|\mathbf{z})$", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#1A202C")
    _draw_arrow(ax, (8.68, y), (9.12, y))
    _draw_rect(ax, 9.12, y - 0.58, 0.36, 1.16, c_blue_fc, c_blue_ec)
    ax.text(9.30, y, r"$\mathbf{x}'$", ha="center", va="center", fontsize=11.5, fontweight="bold")

    # --- ROW 3: Flow-based models ---
    y = row_ys[2]
    _draw_rect(ax, 3.58, y - 0.58, 0.36, 1.16, c_green_fc, c_green_ec)
    ax.text(3.76, y, r"$\mathbf{x}$", ha="center", va="center", fontsize=11.5, fontweight="bold")
    _draw_arrow(ax, (3.94, y), (4.42, y))
    _draw_rect(ax, 4.42, y - 0.43, 1.10, 0.86, c_green_fc, c_green_ec)
    ax.text(4.97, y, "Flow\n" + r"$f(\mathbf{x})$", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1A202C")
    _draw_arrow(ax, (5.52, y), (6.30, y))
    _draw_rect(ax, 6.30, y - 0.23, 0.42, 0.46, c_red_fc, c_red_ec)
    ax.text(6.51, y, r"$\mathbf{z}$", ha="center", va="center", fontsize=11.5, fontweight="bold")
    _draw_arrow(ax, (6.72, y), (7.52, y))
    _draw_rect(ax, 7.52, y - 0.43, 1.10, 0.86, c_blue_fc, c_blue_ec)
    ax.text(8.07, y, "Inverse\n" + r"$f^{-1}(\mathbf{z})$", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1A202C")
    _draw_arrow(ax, (8.62, y), (9.12, y))
    _draw_rect(ax, 9.12, y - 0.58, 0.36, 1.16, c_blue_fc, c_blue_ec)
    ax.text(9.30, y, r"$\mathbf{x}'$", ha="center", va="center", fontsize=11.5, fontweight="bold")

    # --- ROW 4: Diffusion models ---
    y = row_ys[3]
    _draw_rect(ax, 3.58, y - 0.58, 0.36, 1.16, c_green_fc, c_green_ec)
    ax.text(3.76, y, r"$\mathbf{x}_0$", ha="center", va="center", fontsize=11.5, fontweight="bold")

    _draw_rect(ax, 4.48, y - 0.58, 0.36, 1.16, c_yellow_fc, c_yellow_ec)
    ax.text(4.66, y, r"$\mathbf{x}_1$", ha="center", va="center", fontsize=11.5, fontweight="bold")

    _draw_rect(ax, 5.38, y - 0.58, 0.36, 1.16, c_orange_fc, c_orange_ec)
    ax.text(5.56, y, r"$\mathbf{x}_2$", ha="center", va="center", fontsize=11.5, fontweight="bold")

    ax.text(7.05, y + 0.02, r"$\cdots\ \cdots$", ha="center", va="center", fontsize=16, fontweight="bold", color="#4A5568")

    _draw_rect(ax, 9.04, y - 0.58, 0.44, 1.16, c_red_fc, c_red_ec)
    ax.text(9.26, y, r"$\mathbf{z}$", ha="center", va="center", fontsize=11.5, fontweight="bold")

    # Forward (solid) & Reverse (dashed) arrows
    pairs = [(3.94, 4.48), (4.84, 5.38), (5.74, 6.62), (7.48, 9.04)]
    for x1, x2 in pairs:
        _draw_arrow(ax, (x1, y + 0.13), (x2, y + 0.13), color="#4A5568", lw=1.4, ls="-", mutation_scale=11)
        _draw_arrow(ax, (x2, y - 0.13), (x1, y - 0.13), color="#4A5568", lw=1.4, ls="--", mutation_scale=11)

    fig.tight_layout(pad=0.35)
    fig.savefig(save_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {save_path}")


def draw_hinh2_3_diffusion(save_path: Path):
    """Hình 2.2: Quá trình khuếch tán thuận và khử nhiễu ngược."""
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    base_img = _load_square_sample(240)
    rng = np.random.default_rng(42)

    def add_noise(img: np.ndarray, sigma: float) -> np.ndarray:
        noise = rng.normal(0.0, sigma, size=img.shape)
        return np.clip((1.0 - sigma * 0.65) * img + noise, 0.0, 1.0)

    def place_image(ax_obj, img_arr, cx, cy, zoom=0.41, frame_ec="#2D3748"):
        oi = OffsetImage(img_arr, zoom=zoom)
        ab = AnnotationBbox(
            oi, (cx, cy),
            frameon=True,
            pad=0.04,
            bboxprops=dict(edgecolor=frame_ec, facecolor="white", linewidth=1.3),
            zorder=4,
        )
        ax_obj.add_artist(ab)

    # Top row: Training Data/Image -> Diffusion Model -> Generated Data/New Image
    ax.text(2.05, 4.86, "Training Data / Image", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#1A202C")
    place_image(ax, base_img, 2.05, 4.02, zoom=0.36)

    ax.text(7.95, 4.86, "Generated Data / New Image", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#1A202C")
    place_image(ax, base_img, 7.95, 4.02, zoom=0.36)

    # Diffusion Model central box
    _draw_rect(ax, 4.22, 3.60, 1.56, 1.04, "#FDE8E8", "#B91C1C", lw=1.8, radius=0.06, zorder=3)
    # Dashed bottom extension indicator matching classic diagram
    _draw_rect(ax, 4.14, 3.30, 1.72, 1.44, "none", "#991B1B", lw=1.4, ls="--", radius=0.04, zorder=2)
    ax.text(5.00, 4.12, "Diffusion\nModel", ha="center", va="center", fontsize=12, fontweight="bold", color="#7F1D1D", zorder=4)

    _draw_arrow(ax, (2.72, 4.02), (4.06, 4.02), color="#B91C1C", lw=2.2, mutation_scale=15)
    _draw_arrow(ax, (5.92, 4.02), (7.28, 4.02), color="#B91C1C", lw=2.2, mutation_scale=15)

    # Expansion bracket lines from central box down to detailed diffusion strip
    ax.plot([4.22, 1.25], [3.30, 2.90], color="#991B1B", lw=1.5, zorder=2)
    ax.plot([5.78, 8.75], [3.30, 2.90], color="#991B1B", lw=1.5, zorder=2)
    ax.plot([1.25, 1.25], [2.90, 0.35], color="#991B1B", lw=1.5, zorder=2)
    ax.plot([8.75, 8.75], [2.90, 0.35], color="#991B1B", lw=1.5, zorder=2)

    # Forward Diffusion arrow + label
    ax.text(5.00, 2.75, "Forward Diffusion", ha="center", va="center", fontsize=12, fontweight="bold", color="#1A202C")
    _draw_arrow(ax, (1.65, 2.50), (8.35, 2.50), color="#B91C1C", lw=2.0, mutation_scale=15)

    # Four progressively noisy frames
    sigmas = [0.0, 0.14, 0.32, 0.58]
    xs = [2.25, 4.08, 5.92, 7.75]
    sublabels = [r"$\mathbf{x}_0$ (Clean)", r"$\mathbf{x}_1$ (Light noise)", r"$\mathbf{x}_t$ (Noisy)", r"$\mathbf{x}_T$ (Noisy Image)"]
    for cx, sig, lbl in zip(xs, sigmas, sublabels):
        noisy = base_img if sig == 0.0 else add_noise(base_img, sig)
        place_image(ax, noisy, cx, 1.72, zoom=0.35)
        ax.text(cx, 0.92, lbl, ha="center", va="center", fontsize=10.0, fontweight="bold", color="#2D3748")

    # Reverse Diffusion arrow + label
    _draw_arrow(ax, (8.35, 0.54), (1.65, 0.54), color="#B91C1C", lw=2.0, mutation_scale=15)
    ax.text(5.00, 0.25, "Reverse Diffusion", ha="center", va="center", fontsize=12, fontweight="bold", color="#1A202C")

    fig.tight_layout(pad=0.35)
    fig.savefig(save_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {save_path}")


def draw_hinh2_2_vit(save_path: Path):
    """Hình 2.3: Kiến trúc tổng quan của mạng Vision Transformer (ViT)."""
    fig, ax = plt.subplots(figsize=(10, 6.4), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    base_img = _load_square_sample(220)

    # Title box on upper-left
    _draw_rect(ax, 0.55, 3.70, 3.85, 1.20, "#FFFFFF", "#2D3748", lw=1.4, radius=0.06)
    ax.text(
        2.475, 4.30,
        "Vision Transformer (ViT)\nArchitecture",
        ha="center", va="center", fontsize=15.5, fontweight="bold", color="#2D3748", linespacing=1.35,
    )

    # Stacked input images on lower-left
    for dx, dy in [(0.18, 0.18), (0.12, 0.12), (0.06, 0.06), (0.0, 0.0)]:
        _draw_rect(ax, 1.45 + dx, 1.72 + dy, 1.25, 1.25, "#E2E8F0", "#2D3748", lw=1.1, zorder=3)
    oi = OffsetImage(base_img, zoom=0.34)
    ab = AnnotationBbox(
        oi, (2.075, 2.345),
        frameon=True,
        pad=0.02,
        bboxprops=dict(edgecolor="#2D3748", facecolor="white", linewidth=1.2),
        zorder=4,
    )
    ax.add_artist(ab)

    # Generate Image Patches box
    _draw_rect(ax, 1.25, 0.42, 1.85, 0.72, "#9CC3F2", "#2B6CB0", lw=1.5, radius=0.14)
    ax.text(2.175, 0.78, "Generate Image\nPatches", ha="center", va="center", fontsize=11, fontweight="bold", color="#1A202C")

    _draw_arrow(ax, (2.08, 1.66), (2.08, 1.18), color="#4A5568", lw=1.6)

    # Bottom-right: Patch row inside gray tray
    _draw_rect(ax, 5.05, 0.38, 4.55, 0.80, "#F1F5F9", "#475569", lw=1.3, radius=0.14)
    patch_coords = [(0, 0), (0, 110), (110, 0), (110, 110)]
    patch_xs = [5.68, 6.76, 7.84, 8.92]
    for px, (r0, c0) in zip(patch_xs, patch_coords):
        patch = base_img[r0:r0 + 110, c0:c0 + 110]
        poi = OffsetImage(patch, zoom=0.32)
        pab = AnnotationBbox(
            poi, (px, 0.78),
            frameon=True,
            pad=0.02,
            bboxprops=dict(edgecolor="#334155", facecolor="white", linewidth=1.0),
            zorder=4,
        )
        ax.add_artist(pab)

    _draw_arrow(ax, (3.15, 0.78), (5.00, 0.78), color="#4A5568", lw=1.6)

    # Linear Projection + Position Embedding bar
    _draw_rect(ax, 5.05, 1.50, 4.55, 0.56, "#9CC3F2", "#2B6CB0", lw=1.5, radius=0.12)
    ax.text(7.325, 1.78, "Linear Projection + Position Embedding", ha="center", va="center", fontsize=10.8, fontweight="bold", color="#1A202C")

    for px in patch_xs:
        _draw_arrow(ax, (px, 1.18), (px, 1.48), color="#4A5568", lw=1.4, mutation_scale=11)
        _draw_arrow(ax, (px, 2.08), (px, 2.38), color="#4A5568", lw=1.4, mutation_scale=11)

    # Transformer Encoder container
    _draw_rect(ax, 5.05, 2.38, 4.55, 2.40, "#F1F5F9", "#475569", lw=1.4, radius=0.20, zorder=2)

    # Detailed Transformer Block (dashed orange outline)
    _draw_rect(ax, 5.25, 2.52, 4.15, 1.14, "#F8FAFC", "#D97706", lw=1.5, ls=":", radius=0.14, zorder=3)
    ax.text(5.42, 3.49, "Transformer Block", ha="left", va="center", fontsize=9.8, fontweight="bold", color="#D97706", zorder=4)

    _draw_rect(ax, 5.55, 2.60, 3.65, 0.36, "#B7E1CD", "#1E8449", lw=1.3, radius=0.09, zorder=4)
    ax.text(7.375, 2.78, "Multi-Head Self-Attention", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1A202C", zorder=5)

    _draw_rect(ax, 5.55, 3.03, 3.65, 0.36, "#9CC3F2", "#2B6CB0", lw=1.3, radius=0.09, zorder=4)
    ax.text(7.375, 3.21, "Feed Forward Neural Network", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1A202C", zorder=5)

    # Ellipsis + 2 stacked Transformer Block bars
    ax.text(7.325, 3.84, "⋮", ha="center", va="center", fontsize=16, fontweight="bold", color="#334155", zorder=4)

    _draw_rect(ax, 5.40, 3.98, 3.85, 0.32, "#F9CB9C", "#B45309", lw=1.3, radius=0.08, zorder=4)
    ax.text(7.325, 4.14, "Transformer Block", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1A202C", zorder=5)

    _draw_rect(ax, 5.40, 4.37, 3.85, 0.32, "#F9CB9C", "#B45309", lw=1.3, radius=0.08, zorder=4)
    ax.text(7.325, 4.53, "Transformer Block", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1A202C", zorder=5)

    # Top-right: Patch Embeddings
    ax.text(7.325, 6.08, "Patch Embeddings", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#1A202C")
    _draw_rect(ax, 5.05, 5.26, 4.55, 0.58, "#F1F5F9", "#475569", lw=1.4, radius=0.14)
    token_xs = [5.32, 6.40, 7.48, 8.56]
    for tx in token_xs:
        _draw_rect(ax, tx, 5.38, 0.78, 0.34, "#E6B8CE", "#8E44AD", lw=1.2, radius=0.07, zorder=4)

    _draw_arrow(ax, (7.325, 4.78), (7.325, 5.24), color="#4A5568", lw=1.6)

    # Feed-Forward Network (MLP head) + Classification on top-left
    ax.text(3.38, 6.26, "Feed-Forward\nNetwork", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#1A202C", linespacing=1.18)

    in_ys = [5.31, 5.55, 5.79]
    hid_ys = [5.19, 5.43, 5.67, 5.91]
    out_ys = [5.31, 5.55, 5.79]
    x_in, x_hid, x_out = 3.88, 3.38, 2.88

    # Arrow from first token to MLP
    _draw_arrow(ax, (5.32, 5.55), (4.05, 5.55), color="#4A5568", lw=1.6)

    for yi in in_ys:
        for yh in hid_ys:
            ax.plot([x_in, x_hid], [yi, yh], color="#64748B", lw=0.85, zorder=3)
    for yh in hid_ys:
        for yo in out_ys:
            ax.plot([x_hid, x_out], [yh, yo], color="#64748B", lw=0.85, zorder=3)

    for yi in in_ys:
        ax.add_patch(patches.Circle((x_in, yi), 0.082, facecolor="#9CC3F2", edgecolor="#1E40AF", lw=1.1, zorder=5))
    for yh in hid_ys:
        ax.add_patch(patches.Circle((x_hid, yh), 0.082, facecolor="#FFE699", edgecolor="#B45309", lw=1.1, zorder=5))
    for yo in out_ys:
        ax.add_patch(patches.Circle((x_out, yo), 0.082, facecolor="#B7E1CD", edgecolor="#166534", lw=1.1, zorder=5))

    _draw_arrow(ax, (2.68, 5.55), (1.75, 5.55), color="#4A5568", lw=1.6)
    ax.text(1.02, 5.55, "Classification", ha="center", va="center", fontsize=12, fontweight="bold", color="#1A202C")

    fig.tight_layout(pad=0.35)
    fig.savefig(save_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {save_path}")


if __name__ == "__main__":
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    draw_hinh2_1(IMG_DIR / "hinh2_1_generative_models.png")
    draw_hinh2_3_diffusion(IMG_DIR / "hinh2_3_diffusion_process.png")
    draw_hinh2_2_vit(IMG_DIR / "hinh2_2_vit_architecture.png")
