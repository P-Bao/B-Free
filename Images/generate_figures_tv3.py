import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Cấu hình phong cách đồ họa chuẩn bài báo quốc tế (Nature/IEEE style)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# ==============================================================================
# HÌNH 3.1: Sơ đồ thuật toán Auto Batch Size Finder
# ==============================================================================
def draw_figure_3_1(save_path="hinh3_1_auto_batch_finder.png"):
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=300)
    ax.set_xlim(0, 140)
    ax.set_ylim(-1.2, 8.5)
    
    # 1. Các vùng nền phân định trạng thái VRAM
    # Vùng VRAM an toàn (<= 32, chiếm ~82% VRAM < 90% threshold)
    rect_safe = patches.Rectangle((0, -0.6), 36, 8.8, linewidth=0, facecolor='#d4edda', alpha=0.6, label='VRAM An toàn (≤ 90% giới hạn)')
    # Vùng buffer/tiệm cận (36 -> 48)
    rect_warn = patches.Rectangle((36, -0.6), 12, 8.8, linewidth=0, facecolor='#fff3cd', alpha=0.6, label='Vùng cận giới hạn (Buffer dự phòng)')
    # Vùng OOM (> 48)
    rect_oom = patches.Rectangle((48, -0.6), 92, 8.8, linewidth=0, facecolor='#f8d7da', alpha=0.6, label='Vùng OOM (Out Of Memory)')
    
    ax.add_patch(rect_safe)
    ax.add_patch(rect_warn)
    ax.add_patch(rect_oom)
    
    # Đường phân cách mốc
    ax.axvline(x=48, color='#dc3545', linestyle='--', linewidth=2, label='Ngưỡng OOM (~48)')
    ax.axvline(x=32, color='#155724', linestyle='-', linewidth=2.5, label='Batch Size Tối ưu (bs = 32)')

    # --- PHA 1: DOUBLING PHASE ---
    doubling_steps = [2, 4, 8, 16, 32, 64]
    doubling_y = 6.0
    ax.text(2, doubling_y + 1.2, "PHA 1: TĂNG GẤP ĐÔI (Doubling Phase: 2 → 4 → 8 → 16 → 32 → 64)", 
            fontsize=12, fontweight='bold', color='#155724')
    
    for i in range(len(doubling_steps)-1):
        x_start = doubling_steps[i]
        x_end = doubling_steps[i+1]
        color = '#28a745' if x_end <= 32 else '#dc3545'
        ax.annotate('', xy=(x_end, doubling_y), xytext=(x_start, doubling_y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.4, mutation_scale=16))
        ax.plot(x_start, doubling_y, 'o', color='#28a745', markersize=8)
    
    # Điểm 64 bị OOM
    ax.plot(64, doubling_y, 'X', color='#dc3545', markersize=12)
    ax.text(64, doubling_y + 0.45, "OOM!\n(bs=64)", color='#721c24', fontsize=9.5, fontweight='bold', ha='center')
    
    # --- PHA 2: BINARY SEARCH PHASE ---
    bs_y = 3.6
    ax.text(2, bs_y + 1.2, "PHA 2: TÌM KIẾM NHỊ PHÂN (Binary Search Phase trong khoảng [32, 64])", 
            fontsize=12, fontweight='bold', color='#004085')
    
    # Step 1: Thử mid = 48 -> OOM
    ax.plot(32, bs_y, 'o', color='#28a745', markersize=8)
    ax.annotate('', xy=(48, bs_y), xytext=(32, bs_y),
                arrowprops=dict(arrowstyle="->", color='#dc3545', lw=2.2, linestyle=':', mutation_scale=14))
    ax.plot(48, bs_y, 'X', color='#dc3545', markersize=11)
    ax.text(40, bs_y + 0.35, "Bước 1: Thử mid = 48 ➔ Thất bại (OOM)", 
            fontsize=9.5, fontweight='bold', color='#dc3545', ha='center',
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#dc3545", alpha=0.9))
    
    # Step 2: Thu hẹp [32, 48], thử mid = 40 -> Vượt quá 90% VRAM buffer
    bs_y2 = 1.9
    ax.plot(32, bs_y2, 'o', color='#28a745', markersize=8)
    ax.annotate('', xy=(40, bs_y2), xytext=(32, bs_y2),
                arrowprops=dict(arrowstyle="->", color='#d39e00', lw=2.2, linestyle=':', mutation_scale=14))
    ax.plot(40, bs_y2, 's', color='#d39e00', markersize=9)
    ax.text(36, bs_y2 + 0.35, "Bước 2: Thử mid = 40 ➔ Vượt ngưỡng an toàn 90% VRAM", 
            fontsize=9.5, fontweight='bold', color='#856404', ha='center',
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#d39e00", alpha=0.9))
    
    # --- KẾT QUẢ TỐI ƯU ---
    ax.scatter([32], [0.1], s=260, color='#28a745', edgecolors='#155724', linewidth=2.5, zorder=5)
    ax.text(32, -0.75, "★ LỰA CHỌN TỐI ƯU: BATCH_SIZE = 32\n(Sử dụng ~82% VRAM, an toàn tuyệt đối cho 2-pass DCS Loss)", 
            fontsize=10.5, fontweight='bold', color='#155724', ha='center',
            bbox=dict(boxstyle="round,pad=0.4", fc="#d4edda", ec="#28a745", lw=1.5))

    # Tinh chỉnh hiển thị trục X
    ax.set_xticks([2, 8, 16, 32, 40, 48, 64, 96, 128])
    ax.set_xlabel("Kích thước Batch Size (Số lượng ảnh / batch)", fontsize=11.5, fontweight='bold', labelpad=8)
    ax.set_yticks([])
    ax.set_title("HÌNH 3.1: SƠ ĐỒ THUẬT TOÁN AUTO BATCH SIZE FINDER (DOUBLING + BINARY SEARCH)", 
                 fontsize=13, fontweight='bold', pad=15, color='#111111')
    
    ax.legend(loc='upper right', framealpha=0.95, fontsize=9.5)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Đã vẽ và lưu: {save_path}")


# ==============================================================================
# HÌNH 3.2: Training Curves (Train Loss, Val AUC, Val bAcc qua 8 Epoch)
# ==============================================================================
def draw_figure_3_2(csv_path="train_log.csv", save_path="training_curves.png"):
    if not os.path.exists(csv_path):
        print(f"[WARN] Không tìm thấy {csv_path}, bỏ qua Hình 3.2")
        return
        
    df = pd.read_csv(csv_path)
    
    fig, ax1 = plt.subplots(figsize=(10, 5.8), dpi=300)
    ax2 = ax1.twinx()
    
    # Trục 1: Train Loss & Val Loss
    line1 = ax1.plot(df['epoch'], df['train_loss'], color='#1f77b4', marker='o', linewidth=2.4, label='Train Loss')
    line2 = ax1.plot(df['epoch'], df['val_loss'], color='#4a90e2', linestyle='--', marker='s', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch Huấn luyện', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Loss (Cross-Entropy + DCS)', color='#1f77b4', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_ylim(0.18, 0.32)
    
    # Trục 2: Val AUC & Val bAcc
    line3 = ax2.plot(df['epoch'], df['val_auc'], color='#2ca02c', marker='^', linewidth=2.2, label='Val AUC')
    line4 = ax2.plot(df['epoch'], df['val_bacc'], color='#ff7f0e', marker='D', linewidth=2.2, label='Val bAcc')
    ax2.set_ylabel('Độ chính xác (AUC / bAcc)', color='#d95f02', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#d95f02')
    ax2.set_ylim(0.98, 1.002)
    
    # Đường đánh dấu best checkpoint tại Epoch 5
    best_epoch = 5
    ax1.axvline(x=best_epoch, color='#d62728', linestyle=':', linewidth=2)
    ax2.annotate('Best Checkpoint (Epoch 5)\nbAcc = 99.97%', 
                 xy=(best_epoch, df.loc[df['epoch'] == best_epoch, 'val_bacc'].values[0]),
                 xytext=(best_epoch - 1.8, 0.993),
                 arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9.5, fontweight='bold', color='#d62728',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fff5f5", ec="#d62728", lw=1))
    
    # Gom legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', framealpha=0.95, fontsize=9.5)
    
    ax1.set_xticks(range(1, 9))
    ax1.grid(True, linestyle=':', alpha=0.5)
    plt.title("HÌNH 3.2: ĐƯỜNG CONG HUẤN LUYỆN (TRAINING & VALIDATION CURVES QUA 8 EPOCH)", 
              fontsize=12, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Đã vẽ và lưu: {save_path}")


# ==============================================================================
# HÌNH 3.3: Val AUC theo Epoch cho từng biến thể SD2.1
# ==============================================================================
def draw_figure_3_3(csv_path="train_log_variants.csv", save_path="hinh3_3_variant_auc_curves.png"):
    if not os.path.exists(csv_path):
        print(f"[ERROR] Không tìm thấy file {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    palette = {
        'SD2.1_inpainted_diffcat': '#2b5c8f',
        'SD2.1_inpainted_diffcat_origBG': '#4682b4',
        'SD2.1_inpainted_samecat': '#2e8b57',
        'SD2.1_inpainted_samecat_origBG': '#20b2aa',
        'SD2.1_selfconditioned': '#ff8c00',
        'SD2.1_selfconditioned_origBG': '#d90429'  # Đỏ đậm nổi bật
    }
    
    markers = {
        'SD2.1_inpainted_diffcat': 'o',
        'SD2.1_inpainted_diffcat_origBG': 's',
        'SD2.1_inpainted_samecat': '^',
        'SD2.1_inpainted_samecat_origBG': 'v',
        'SD2.1_selfconditioned': 'D',
        'SD2.1_selfconditioned_origBG': 'P'
    }
    
    for variant in df['variant'].unique():
        sub = df[df['variant'] == variant].sort_values('epoch')
        is_hardest = (variant == 'SD2.1_selfconditioned_origBG')
        
        ax.plot(
            sub['epoch'], sub['val_auc'],
            label=f"{variant}" + (" (Khó nhất)" if is_hardest else ""),
            color=palette.get(variant, '#333333'),
            marker=markers.get(variant, 'o'),
            linewidth=2.8 if is_hardest else 1.8,
            markersize=8 if is_hardest else 6,
            alpha=1.0 if is_hardest else 0.85,
            zorder=5 if is_hardest else 3
        )
        
    ax.annotate('Khởi đầu thấp nhất:\nAUC = 0.9873 (bAcc 94.18%)', 
                xy=(1, 0.9873), xytext=(1.4, 0.9885),
                arrowprops=dict(facecolor='#d90429', shrink=0.08, width=1.5, headwidth=7),
                fontsize=9, fontweight='bold', color='#d90429',
                bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2f2", ec="#d90429", lw=1))
    
    ax.annotate('Hội tụ sau 8 epoch:\nAUC = 0.9999 (bAcc 99.65%)', 
                xy=(8, 0.9999), xytext=(5.6, 0.9960),
                arrowprops=dict(facecolor='#d90429', shrink=0.08, width=1.5, headwidth=7),
                fontsize=9, fontweight='bold', color='#d90429',
                bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2f2", ec="#d90429", lw=1))
    
    ax.set_xlim(0.8, 8.2)
    ax.set_ylim(0.985, 1.0005)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("Epoch Huấn luyện", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel("Validation AUC", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title("HÌNH 3.3: DIỄN BIẾN VAL AUC QUA 8 EPOCH THEO TỪNG BIẾN THỂ SD2.1", 
                 fontsize=12.5, fontweight='bold', pad=15)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', framealpha=0.95, fontsize=9.5)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Đã vẽ và lưu: {save_path}")


if __name__ == "__main__":
    os.makedirs("Images", exist_ok=True)
    draw_figure_3_1("Images/hinh3_1_auto_batch_finder.png")
    draw_figure_3_2("train_log.csv", "Images/training_curves.png")
    draw_figure_3_3("train_log_variants.csv", "Images/hinh3_3_variant_auc_curves.png")
