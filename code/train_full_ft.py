import os
import time
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.loader import load_config, build_model_kwargs
from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
from datasets.bfree_dataset import BFreeDataset
from utils.dmetrics import balanced_accuracy_score, roc_auc_score


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss_sum, ce_loss_sum, dcs_loss_sum = 0.0, 0.0, 0.0
    correct, total_samples = 0, 0

    start_time = time.time()
    for step, (img_clean, img_deg, labels) in enumerate(loader):
        img_clean = img_clean.to(device, non_blocking=True)
        img_deg = img_deg.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss_total, loss_ce, loss_dcs = model.compute_loss(img_clean, img_deg, labels)

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)
        total_loss_sum += loss_total.item() * batch_size
        ce_loss_sum += loss_ce.item() * batch_size
        dcs_loss_sum += loss_dcs.item() * batch_size
        total_samples += batch_size

        if step % 20 == 0:
            print(
                f"Epoch [{epoch}] Step [{step}/{len(loader)}] | "
                f"Total: {loss_total.item():.4f} (CE: {loss_ce.item():.4f}, DCS: {loss_dcs.item():.4f}) | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    elapsed = time.time() - start_time
    return {
        "train_loss": total_loss_sum / total_samples,
        "train_ce": ce_loss_sum / total_samples,
        "train_dcs": dcs_loss_sum / total_samples,
        "time": elapsed,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_scores = []
    all_labels = []
    val_loss_sum = 0.0
    total_samples = 0

    for img_clean, img_deg, labels in loader:
        img_clean = img_clean.to(device)
        img_deg = img_deg.to(device)
        labels = labels.to(device)

        loss_total, _, _ = model.compute_loss(img_clean, img_deg, labels)
        out = model(img_clean)
        logits = out["logits"]

        # score = logit_fake - logit_real
        score = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        
        val_loss_sum += loss_total.item() * labels.size(0)
        total_samples += labels.size(0)
        all_scores.extend(score)
        all_labels.extend(labels.cpu().numpy())

    all_scores = torch.tensor(all_scores).numpy()
    all_labels = torch.tensor(all_labels).numpy()

    auc = roc_auc_score(all_labels, all_scores)
    bacc = balanced_accuracy_score(all_labels, all_scores > 0)
    return {
        "val_loss": val_loss_sum / total_samples,
        "val_auc": auc,
        "val_bacc": bacc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bfree_dcs.yaml")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/full_ft")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_config(args.config)
    model_kwargs = build_model_kwargs(cfg)

    model = BFreeGlobalForgeViT(**model_kwargs)
    model.to(args.device)

    train_dataset = BFreeDataset(
        csv_file=args.train_csv,
        data_root=args.data_root,
        img_size=cfg["backbone"]["img_size"],
        is_train=True,
        degradation_cfg=cfg.get("degradation"),
    )
    val_dataset = BFreeDataset(
        csv_file=args.val_csv,
        data_root=args.data_root,
        img_size=cfg["backbone"]["img_size"],
        is_train=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-7)

    log_file = os.path.join(args.output_dir, "train_log.csv")
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,train_ce,train_dcs,val_loss,val_auc,val_bacc\n")

    best_bacc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_res = train_one_epoch(model, train_loader, optimizer, scheduler, args.device, epoch)
        val_res = evaluate(model, val_loader, args.device)

        print(
            f"--> Epoch {epoch} Final: Train Loss={tr_res['train_loss']:.4f} "
            f"(CE={tr_res['train_ce']:.4f}, DCS={tr_res['train_dcs']:.4f}) | "
            f"Val Loss={val_res['val_loss']:.4f}, AUC={val_res['val_auc']:.4f}, bAcc={val_res['val_bacc']:.4f}"
        )

        with open(log_file, "a") as f:
            f.write(
                f"{epoch},{tr_res['train_loss']:.4f},{tr_res['train_ce']:.4f},{tr_res['train_dcs']:.4f},"
                f"{val_res['val_loss']:.4f},{val_res['val_auc']:.4f},{val_res['val_bacc']:.4f}\n"
            )

        if val_res["val_bacc"] > best_bacc:
            best_bacc = val_res["val_bacc"]
            ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_bacc": best_bacc}, ckpt_path)
            print(f"[*] Checkpoint saved: {ckpt_path} (bAcc: {best_bacc:.4f})")


if __name__ == "__main__":
    main()