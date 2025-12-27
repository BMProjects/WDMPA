"""Training script for WDMPA-Net.

Usage:
    python tools/train.py --config configs/train_mpiigaze.yaml
    python tools/train.py --data-root datasets/MPIIFaceGaze --dataset mpiigaze --fold 0
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm


# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wdmpa import WDMPANet
from wdmpa.data import MPIIGazeDataset, Gaze360Dataset, create_dataloader
from wdmpa.utils import GazeLoss, angular_error
from wdmpa.models.baselines import MobileNetV3Gaze, ShuffleNetV2Gaze
from wdmpa.models.ablation import WDMPANetAblation


# Supported models
MODEL_REGISTRY = {
    "wdmpa": lambda: WDMPANet(),
    "mobilenetv3": lambda: MobileNetV3Gaze(),
    "shufflenetv2": lambda: ShuffleNetV2Gaze(),
    # Ablation variants
    "wdmpa_awwd_fixed": lambda: WDMPANetAblation(downsample_type="awwd_fixed", attention_type="mpa"),
    "wdmpa_stride_conv": lambda: WDMPANetAblation(downsample_type="stride_conv", attention_type="mpa"),
    "wdmpa_channel_only": lambda: WDMPANetAblation(downsample_type="awwd", attention_type="channel"),
    "wdmpa_spatial_only": lambda: WDMPANetAblation(downsample_type="awwd", attention_type="spatial"),
    "wdmpa_single_scale": lambda: WDMPANetAblation(downsample_type="awwd", attention_type="single"),
    "wdmpa_no_attention": lambda: WDMPANetAblation(downsample_type="awwd", attention_type="none"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train WDMPA-Net")

    # Data
    parser.add_argument("--data-root", type=str, default="datasets/MPIIFaceGaze",
                        help="Path to dataset root")
    parser.add_argument("--dataset", type=str, choices=["mpiigaze", "gaze360"],
                        default="mpiigaze", help="Dataset name")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold for leave-one-out (MPIIGaze only)")

    # Training
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (optimized for RTX 4090)")
    parser.add_argument("--lr", type=float, default=1.6e-3, help="Learning rate (scaled for batch 512)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (AdamW default: 0.01)")
    parser.add_argument("--warmup-epochs", type=int, default=8, help="Warmup epochs for large batch")
    parser.add_argument("--amp", action="store_true", default=True, help="Use mixed precision training")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"], help="Optimizer type")

    # Model
    parser.add_argument("--model", type=str, default="wdmpa",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model to train")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained weights")

    # Output
    parser.add_argument("--output-dir", type=str, default="runs/train",
                        help="Output directory")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")

    # Device
    parser.add_argument("--device", type=str, default="0", help="CUDA device(s)")
    parser.add_argument("--workers", type=int, default=12, help="DataLoader workers (optimized for RTX 4090)")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Batches to prefetch per worker")

    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: GradScaler = None,
    use_amp: bool = False,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward with AMP
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast('cuda'):
                pred = model(images)
                loss, loss_dict = criterion(pred, labels)
            
            # Backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            pred = model(images)
            loss, loss_dict = criterion(pred, labels)
            loss.backward()
            optimizer.step()

        # Stats
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {"train_loss": total_loss / total_samples}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Validate model."""
    model.eval()
    total_loss = 0
    all_errors = []
    total_samples = 0

    for images, labels in tqdm(dataloader, desc="Validating"):
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        loss, _ = criterion(pred, labels)

        errors = angular_error(pred, labels)
        all_errors.extend(errors.cpu().tolist())

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    mean_error = sum(all_errors) / len(all_errors)
    return {
        "val_loss": total_loss / total_samples,
        "angular_error": mean_error,
    }


def main():
    args = parse_args()

    # Device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "mpiigaze":
        train_dataset = MPIIGazeDataset(args.data_root, fold=args.fold, train=True)
        val_dataset = MPIIGazeDataset(args.data_root, fold=args.fold, train=False)
    else:
        train_dataset = Gaze360Dataset(args.data_root, label_file="Label/train.label")
        val_dataset = Gaze360Dataset(args.data_root, label_file="Label/val.label")

    train_loader = create_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.workers, prefetch_factor=args.prefetch_factor)
    val_loader = create_dataloader(val_dataset, args.batch_size, shuffle=False, num_workers=args.workers, prefetch_factor=args.prefetch_factor)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    print(f"Creating model: {args.model}")
    model = MODEL_REGISTRY[args.model]()
    if args.pretrained:
        print(f"Loading pretrained: {args.pretrained}")
        state = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    # Loss, optimizer, scheduler
    criterion = GazeLoss(l1_weight=1.0, angular_weight=0.0)
    
    # Optimizer selection
    if args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Using AdamW optimizer (lr={args.lr}, wd={args.weight_decay})")
    else:
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Using Adam optimizer (lr={args.lr}, wd={args.weight_decay})")
    
    # Scheduler with warmup
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr * 0.01)
    
    # AMP scaler
    scaler = GradScaler('cuda') if args.amp else None
    if args.amp:
        print("Mixed precision training enabled (AMP)")

    # Training loop
    best_error = float("inf")
    for epoch in range(1, args.epochs + 1):
        # Warmup learning rate
        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
        
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, args.amp)
        val_stats = validate(model, val_loader, criterion, device)
        
        # Step scheduler after warmup
        if epoch > args.warmup_epochs:
            scheduler.step()

        # Log
        lr = optimizer.param_groups[0]["lr"]
        print(f"[{epoch}/{args.epochs}] "
              f"Train Loss: {train_stats['train_loss']:.4f} | "
              f"Val Loss: {val_stats['val_loss']:.4f} | "
              f"Error: {val_stats['angular_error']:.2f}° | "
              f"LR: {lr:.6f}")

        # Save
        if val_stats["angular_error"] < best_error:
            best_error = val_stats["angular_error"]
            torch.save(model.state_dict(), output_dir / "best.pkl")
            print(f"  -> New best: {best_error:.2f}°")

        torch.save(model.state_dict(), output_dir / "last.pkl")

    print(f"\nTraining complete. Best error: {best_error:.2f}°")
    print(f"Weights saved to: {output_dir}")


if __name__ == "__main__":
    main()
