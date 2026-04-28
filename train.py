"""
Train CIFAR ResNet-18 for a single dropout mode; save best val checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_cifar10_loaders
from models import MODE_CONFIG, build_model
from runtime_utils import resolve_device, to_device_batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    desc: str,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = to_device_batch(x, y, device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = to_device_batch(x, y, device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


def main() -> None:
    p = argparse.ArgumentParser(description="Train CIFAR ResNet-18 (MC Dropout modes)")
    p.add_argument("--mode", type=str, required=True, choices=list(MODE_CONFIG.keys()))
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda (default if available) or cpu",
    )
    args = p.parse_args()

    set_seed(args.seed)

    device = resolve_device(
        args.device,
        cuda_warning=(
            "WARNING: --device cuda requested but torch.cuda.is_available() is False. "
            "Install a CUDA build of PyTorch and GPU drivers, or run with --device cpu. "
            "Falling back to CPU."
        ),
    )

    pin_mem = device.type == "cuda"
    if not pin_mem:
        print("Using CPU: pin_memory disabled (no warning).")

    train_loader, val_loader = get_cifar10_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.workers,
        val_fraction=0.1,
        seed=args.seed,
        pin_memory=pin_mem,
    )

    model = build_model(mode=args.mode).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_path = out_dir / f"resnet18_{args.mode}_best.pt"

    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            desc=f"train {epoch}/{args.epochs}",
        )
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            desc=f"val   {epoch}/{args.epochs}",
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "mode": args.mode,
                },
                best_path,
            )
            print(f"  saved best -> {best_path}")

    with open(out_dir / f"resnet18_{args.mode}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Done. Best val_loss={best_val_loss:.4f} checkpoint: {best_path}")


if __name__ == "__main__":
    main()
