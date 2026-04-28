"""
Shared runtime helpers for train/eval scripts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import torch


def resolve_device(device_arg: str, *, cuda_warning: str | None = None) -> torch.device:
    """Resolve CLI device arg and handle CUDA fallback consistently."""
    wants_cuda = device_arg.lower().startswith("cuda")
    if wants_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    if wants_cuda:
        if cuda_warning:
            print(cuda_warning)
        else:
            print("WARNING: CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def to_device_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)


def write_json(path: str | Path, rows: list[dict], *, indent: int = 2) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=indent)


def write_csv(path: str | Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
