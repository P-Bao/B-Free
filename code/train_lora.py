import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from peft import LoraConfig, get_peft_model

from configs.loader import load_config, build_model_kwargs
from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
from datasets.bfree_dataset import BFreeDataset
from train_full_ft import train_one_epoch, evaluate


def apply_lora_to_backbone(model: BFreeGlobalForgeViT, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """Áp dụng LoRA lên timm VisionTransformer backbone, giữ full train cho LIB, GSR và Head."""
    
    # Target modules trong blocks của timm ViT
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["qkv", "proj", "fc1", "fc2"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    
    # Bọc LoRA cho self.model (VisionTransformer)
    model.model = get_peft_model(model.model, lora_config)
    
    # Đảm bảo LIB, GSR, fc_norm và head luôn được huấn luyện đầy đủ
    trainable_submodules = [model.lib, model.gsr, model.model.base_model.model.head, model.model.base_model.model.fc_norm]
    for sub in trainable_submodules:
        if sub is not None:
            for p in sub.parameters():
                p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] LoRA Configured (rank={r}): {trainable_params:,} / {total_params:,} parameters trainable ({100 * trainable_params / total_params:.2f}%)")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bfree_dcs.yaml")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/lora_r16")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_config(args.config)
    model_kwargs = build_model_kwargs(cfg)

    model = BFreeGlobalForgeViT(**model_kwargs)
    model = apply_lora_to_backbone(model, r=args.lora_rank)
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

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
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
            ckpt_path = os.path.join(args.output_dir, "best_model_lora.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_bacc": best_bacc}, ckpt_path)
            print(f"[*] Best LoRA checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()