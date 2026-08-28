import os
import argparse
import pandas as pd
import numpy as np
import torch
import tqdm
from PIL import Image
from torchvision.transforms import Compose

from utils.normalization import get_list_norm
from utils import dmetrics
from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
from configs.loader import load_config, build_model_kwargs


def load_target_model(config_path: str, checkpoint_path: str, device: str):
    cfg = load_config(config_path)
    kwargs = build_model_kwargs(cfg)
    model = BFreeGlobalForgeViT(**kwargs)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_checkpoint(checkpoint_path, strict=False)
    model.to(device).eval()
    return model, cfg["backbone"]["img_size"]


def evaluate_dataset(model, dataset_csv: str, root_dir: str, img_size: int, device: str):
    df = pd.read_csv(dataset_csv)
    transform = Compose(get_list_norm("resnet"))
    scores = []

    with torch.no_grad():
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {os.path.basename(dataset_csv)}"):
            img_path = os.path.join(root_dir, row["filename"])
            img = Image.open(img_path).convert("RGB")
            
            # Cắt ảnh chuẩn 504x504 hoặc giữ nguyên kích thước qua normalize
            tensor = transform(img).unsqueeze(0).to(device)
            out = model(tensor)
            logits = out["logits"]
            score = (logits[:, 1] - logits[:, 0]).item()
            scores.append(score)

    scores = np.array(scores)
    labels = (df["label"] != "REAL") & (df["label"] != 0) & (df["label"] != "0")

    return {
        "AUC": dmetrics.roc_auc_score(labels, scores),
        "bAcc": dmetrics.balanced_accuracy_score(labels, scores > 0) * 100,
        "NLL": dmetrics.balanced_nll_binary(labels, scores),
        "ECE": dmetrics.balanced_ece_binary(labels, scores),
        "Pd10": dmetrics.pd_at_far(labels, scores, 0.10) * 100,
        "EER": dmetrics.calculate_eer2(labels, scores) * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bfree_dcs.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint_name=path pairs")
    parser.add_argument("--benchmark_csvs", nargs="+", required=True, help="List of benchmark_name=csv_path pairs")
    parser.add_argument("--data_root", type=str, default="./")
    parser.add_argument("--output_csv", type=str, default="./results/benchmark_comparison.csv")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    
    ckpt_dict = dict(item.split("=") for item in args.checkpoints)
    bench_dict = dict(item.split("=") for item in args.benchmark_csvs)

    all_results = []
    for ckpt_name, ckpt_path in ckpt_dict.items():
        print(f"\n================ Loading Model: {ckpt_name} ================")
        model, img_size = load_target_model(args.config, ckpt_path, args.device)

        for bench_name, csv_path in bench_dict.items():
            metrics = evaluate_dataset(model, csv_path, args.data_root, img_size, args.device)
            row = {"Model": ckpt_name, "Benchmark": bench_name, **metrics}
            all_results.append(row)

    res_df = pd.DataFrame(all_results)
    res_df.to_csv(args.output_csv, index=False)
    print("\n" + "=" * 30 + " BENCHMARK SUMMARY " + "=" * 30)
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))


if __name__ == "__main__":
    main()