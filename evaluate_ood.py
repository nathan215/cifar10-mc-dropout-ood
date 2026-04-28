"""
OOD evaluation: reads CIFAR-10-C from data/CIFAR-10-C.tar (fixed path).
Model is 10-class CIFAR-10; archive inner folder may be CIFAR-10-C/ (see dataset.resolve_cifar10c_inner_prefix).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    CIFAR10CArrayDataset,
    labels_for_cifar10c_severity,
    list_cifar10c_corruption_names_in_tar,
    load_cifar10c_corruption_full_from_tar,
    load_cifar10c_labels_from_tar,
    resolve_cifar10c_inner_prefix,
)
from metrics import diagnostics_bundle_from_mc_probs, metrics_from_mc_probs
from models import MODE_CONFIG, build_model, enable_mc_dropout
from runtime_utils import resolve_device, to_device_batch, write_csv, write_json

HERE = Path(__file__).resolve().parent
# Fixed location (no search / no extra flags)
CIFAR10C_TAR = HERE / "data" / "CIFAR-10-C.tar"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOD eval on CIFAR-10-C (data/CIFAR-10-C.tar)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--mode", type=str, default=None, choices=list(MODE_CONFIG.keys()))
    p.add_argument("--mc-samples", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--out-csv", type=str, default=None)
    p.add_argument("--corruptions", type=str, default=None)
    p.add_argument(
        "--save-diagnostics",
        type=str,
        default=None,
        help="Comma-separated corruption stems (e.g. fog,gaussian_noise). Saves .pt per (corruption,severity) "
        "with ACE 15-bin table, mean_probs, targets, pred_entropy for reliability / entropy plots.",
    )
    p.add_argument(
        "--diagnostics-dir",
        type=str,
        default=None,
        help="Folder for diagnostic .pt files (default: same dir as checkpoint)",
    )
    return p.parse_args()


@torch.no_grad()
def collect_mc_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_samples: int,
    use_mc_dropout: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for x, y in tqdm(loader, desc="batches"):
        x, y = to_device_batch(x, y, device)
        if use_mc_dropout:
            enable_mc_dropout(model)
        else:
            model.eval()
        stacks = [F.softmax(model(x), dim=-1) for _ in range(num_samples)]
        all_probs.append(torch.stack(stacks, dim=1).cpu())
        all_targets.append(y.cpu())

    return torch.cat(all_probs, dim=0), torch.cat(all_targets, dim=0)


def _strip_npy(name: str) -> str:
    name = name.strip()
    return name[:-4] if name.lower().endswith(".npy") else name


def load_checkpoint(path: str, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_mode(args: argparse.Namespace, ckpt: dict) -> str:
    mode = args.mode or ckpt.get("mode", "none")
    if mode not in MODE_CONFIG:
        raise ValueError(f"Unknown mode {mode}")
    return mode


def resolve_corruptions(arg_value: str | None, inner: str) -> list[str]:
    if arg_value:
        return [_strip_npy(c) for c in arg_value.split(",") if c.strip()]
    names = list_cifar10c_corruption_names_in_tar(CIFAR10C_TAR, inner)
    return [Path(n).stem for n in names]


def maybe_save_diagnostics(
    *,
    stem: str,
    severity: int,
    mode: str,
    num_samples: int,
    probs_d: torch.Tensor,
    targets_d: torch.Tensor,
    diag_stems: set[str],
    diag_dir: Path,
) -> None:
    if stem not in diag_stems:
        return
    bundle = diagnostics_bundle_from_mc_probs(probs_d, targets_d)
    bundle["corruption"] = stem
    bundle["severity"] = severity
    bundle["mode"] = mode
    bundle["T"] = num_samples
    fname = diag_dir / f"diag_{mode}_{stem}_sev{severity}.pt"
    torch.save(bundle, fname)
    print(f"  saved diagnostics -> {fname}")


def evaluate_corruption(
    *,
    stem: str,
    full_imgs: np.ndarray,
    labels_np: np.ndarray,
    args: argparse.Namespace,
    pin_mem: bool,
    model: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    use_mc: bool,
    mode: str,
    diag_stems: set[str],
    diag_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    for severity in range(1, 6):
        sl = full_imgs[(severity - 1) * 10000 : severity * 10000]
        lab = labels_for_cifar10c_severity(labels_np, severity)
        ds = CIFAR10CArrayDataset(sl, lab)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=pin_mem,
        )
        probs, targets = collect_mc_probs(
            model, loader, device, num_samples=num_samples, use_mc_dropout=use_mc
        )
        probs_d = probs.to(device)
        targets_d = targets.to(device)
        m = metrics_from_mc_probs(probs_d, targets_d)
        row = {
            "corruption": f"{stem}.npy",
            "severity": severity,
            "mode": mode,
            "T": num_samples,
            "mi": float(m["mi"].item()),
            "brier": float(m["brier"].item()),
            "ace": float(m["ace"].item()),
            "accuracy": float(m["accuracy"].item()),
        }
        rows.append(row)
        print(
            f"{stem} sev={severity} acc={row['accuracy']:.4f} "
            f"MI={row['mi']:.6f} Brier={row['brier']:.6f} ACE={row['ace']:.6f}"
        )
        maybe_save_diagnostics(
            stem=stem,
            severity=severity,
            mode=mode,
            num_samples=num_samples,
            probs_d=probs_d,
            targets_d=targets_d,
            diag_stems=diag_stems,
            diag_dir=diag_dir,
        )
    return rows


def main() -> None:
    args = parse_args()
    if not CIFAR10C_TAR.is_file():
        print(f"ERROR: missing {CIFAR10C_TAR}", file=sys.stderr)
        sys.exit(1)

    device = resolve_device(args.device, cuda_warning="WARNING: CUDA not available; using CPU.")
    pin_mem = device.type == "cuda"
    ckpt = load_checkpoint(args.checkpoint, device)
    mode = resolve_mode(args, ckpt)

    num_samples = 1 if mode == "none" else args.mc_samples
    use_mc = mode != "none"

    model = build_model(mode=mode).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    inner = resolve_cifar10c_inner_prefix(CIFAR10C_TAR)
    print(f"Data: {CIFAR10C_TAR} (inner folder: {inner}/)")

    labels_np = load_cifar10c_labels_from_tar(CIFAR10C_TAR, inner)
    corruptions = resolve_corruptions(args.corruptions, inner)

    if not corruptions:
        print("ERROR: no corruptions to run", file=sys.stderr)
        sys.exit(1)

    diag_stems: set[str] = set()
    if args.save_diagnostics:
        diag_stems = {_strip_npy(x) for x in args.save_diagnostics.split(",") if x.strip()}

    ckpt_dir = Path(args.checkpoint).resolve().parent
    diag_dir = Path(args.diagnostics_dir) if args.diagnostics_dir else ckpt_dir
    if diag_stems:
        diag_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for corrupt in corruptions:
        stem = _strip_npy(corrupt)
        try:
            full_imgs = load_cifar10c_corruption_full_from_tar(CIFAR10C_TAR, stem, inner)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skip {stem}: {e}")
            continue
        results.extend(
            evaluate_corruption(
                stem=stem,
                full_imgs=full_imgs,
                labels_np=labels_np,
                args=args,
                pin_mem=pin_mem,
                model=model,
                device=device,
                num_samples=num_samples,
                use_mc=use_mc,
                mode=mode,
                diag_stems=diag_stems,
                diag_dir=diag_dir,
            )
        )

    out_json = args.out_json or str(Path(args.checkpoint).parent / f"ood_{mode}.json")
    write_json(out_json, results)
    print(f"Wrote {out_json}")

    out_csv = args.out_csv or str(Path(args.checkpoint).parent / f"ood_{mode}.csv")
    if results:
        write_csv(out_csv, results)
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
