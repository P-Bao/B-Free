import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_loss_dynamics(log_csv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(log_csv)

    required_cols = ["epoch", "train_loss", "train_ce", "train_dcs"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Log CSV must contain column: {col}")

    # Tính toán tương quan giữa CE Loss và DCS Loss
    corr_pearson = df["train_ce"].corr(df["train_dcs"], method="pearson")
    corr_spearman = df["train_ce"].corr(df["train_dcs"], method="spearman")

    print("\n--- Phân tích tương tác hàm mất mát (CE vs DCS) ---")
    print(f"Pearson Correlation : {corr_pearson:.4f}")
    print(f"Spearman Correlation: {corr_spearman:.4f}")

    if corr_pearson < 0:
        print("[!] CẢNH BÁO: Xuất hiện tương quan nghịch (DCS tăng khi CE giảm). Có dấu hiệu xung đột gradient!")
    else:
        print("[*] TỐT: CE loss và DCS loss cùng hội tụ đồng thuận.")

    sns.set_theme(style="darkgrid")
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)

    color_ce = "tab:blue"
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Cross-Entropy Loss", color=color_ce, fontsize=12)
    l1 = ax1.plot(df["epoch"], df["train_ce"], color=color_ce, marker="o", label="CE Loss", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color_ce)

    ax2 = ax1.twinx()
    color_dcs = "tab:orange"
    ax2.set_ylabel("Degradation Contrastive Loss (DCS)", color=color_dcs, fontsize=12)
    l2 = ax2.plot(df["epoch"], df["train_dcs"], color=color_dcs, marker="s", linestyle="--", label="DCS Loss", linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color_dcs)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title(f"CE Loss vs DCS Loss Dynamics (Corr: {corr_pearson:.2f})", fontsize=14)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "loss_interaction.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[*] Biểu đồ phân tích loss đã được lưu tại: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/loss_analysis")
    args = parser.parse_args()

    analyze_loss_dynamics(args.log_csv, args.output_dir)