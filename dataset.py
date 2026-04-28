"""
CIFAR-10 (train/val) and CIFAR-10-C (.npy per corruption, severity slice).
"""

from __future__ import annotations

import tarfile
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Common CIFAR-10 normalization (used for train, val, and CIFAR-10-C).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def cifar_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def cifar_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def get_cifar10_loaders(
    root: str,
    batch_size: int,
    num_workers: int = 2,
    val_fraction: float = 0.1,
    seed: int = 42,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Train/val split from CIFAR-10 training set (50k). Val fraction default 10% (5k / 45k).
    """
    train_tf = cifar_train_transform()
    eval_tf = cifar_eval_transform()

    full_train = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
    n = len(full_train)
    n_val = int(round(n * val_fraction))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=gen).tolist()
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_set = Subset(full_train, train_idx)

    # Val uses eval transform — build base without transform and apply eval transform via wrapper
    base_eval = datasets.CIFAR10(root=root, train=True, download=True, transform=eval_tf)
    val_set = Subset(base_eval, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def get_cifar10_test_loader(
    root: str,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> DataLoader:
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=cifar_eval_transform())
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _pil_from_hwc_uint8(img: np.ndarray) -> Image.Image:
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


class CIFAR10CNPYDataset(torch.utils.data.Dataset):
    """
    One corruption file: (50000, 32, 32, 3) uint8 typically, HWC.
    Severity s in {1,...,5}: rows [(s-1)*10000 : s*10000].
    labels: labels.npy length 10000 (same order as CIFAR-10 test).
    """

    def __init__(
        self,
        images_npy_path: str | Path,
        labels_npy_path: str | Path,
        severity: int,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        if severity < 1 or severity > 5:
            raise ValueError("severity must be in 1..5")
        self.severity = severity
        self.transform = transform or cifar_eval_transform()

        images = np.load(images_npy_path, mmap_mode="r")
        start = (severity - 1) * 10000
        end = severity * 10000
        self._images = images[start:end]

        labels = np.load(labels_npy_path)
        if labels.ndim > 1:
            labels = labels.reshape(-1)
        self._labels = labels.astype(np.int64)
        if len(self._labels) != 10000:
            raise ValueError(f"labels.npy must have length 10000, got {len(self._labels)}")
        if len(self._images) != 10000:
            raise ValueError(f"Expected 10000 images for severity slice, got {len(self._images)}")

    def __len__(self) -> int:
        return 10000

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = np.asarray(self._images[idx])
        pil = _pil_from_hwc_uint8(img)
        return self.transform(pil), int(self._labels[idx])


class CIFAR10CArrayDataset(torch.utils.data.Dataset):
    """In-memory (N, H, W, C) uint8 slice + labels (10000,) — shared __getitem__ for dir / tar."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self._images = images
        self._labels = labels.astype(np.int64)
        self.transform = transform or cifar_eval_transform()
        if len(self._labels) != len(self._images):
            raise ValueError("images and labels length mismatch")

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = np.asarray(self._images[idx])
        pil = _pil_from_hwc_uint8(img)
        return self.transform(pil), int(self._labels[idx])


def _read_npy_member(tar: tarfile.TarFile, member_path: str) -> np.ndarray:
    member = tar.getmember(member_path)
    bio = tar.extractfile(member)
    if bio is None:
        raise FileNotFoundError(f"Cannot read tar member: {member_path}")
    # Tar file objects lack fileno(); np.load needs a real buffer — use BytesIO.
    return np.load(BytesIO(bio.read()))


def resolve_cifar10c_inner_prefix(tar_path: str | Path) -> str:
    """Folder inside the Zenodo-style archive: CIFAR-100-C or CIFAR-10-C."""
    with tarfile.open(tar_path, "r:*") as tar:
        names = set(tar.getnames())
    for pref in ("CIFAR-100-C", "CIFAR-10-C"):
        if f"{pref}/labels.npy" in names:
            return pref
    raise FileNotFoundError(
        f"Expected CIFAR-100-C/labels.npy or CIFAR-10-C/labels.npy inside {tar_path}"
    )


def list_cifar10c_corruption_names_in_tar(
    tar_path: str | Path,
    inner_prefix: str = "CIFAR-10-C",
) -> list[str]:
    """
    List corruption .npy basenames inside official Zenodo CIFAR-10-C.tar
    (members like CIFAR-10-C/fog.npy). Excludes labels.npy.
    """
    pref = inner_prefix.strip("/").rstrip("/") + "/"
    found: list[str] = []
    with tarfile.open(tar_path, "r:*") as tar:
        for name in tar.getnames():
            if not name.endswith(".npy"):
                continue
            base = Path(name).name
            if base == "labels.npy":
                continue
            if name.startswith(pref) or name == base:
                found.append(base)
    return sorted(set(found))


def load_cifar10c_labels_from_tar(tar_path: str | Path, inner_prefix: str = "CIFAR-10-C") -> np.ndarray:
    pref = inner_prefix.strip("/").rstrip("/")
    with tarfile.open(tar_path, "r:*") as tar:
        try:
            labels = _read_npy_member(tar, f"{pref}/labels.npy")
        except KeyError:
            labels = _read_npy_member(tar, "labels.npy")
    if labels.ndim > 1:
        labels = labels.reshape(-1)
    labels = labels.astype(np.int64)
    n = len(labels)
    if n not in (10000, 50000):
        raise ValueError(f"labels.npy must have length 10000 or 50000, got {n}")
    return labels


def labels_for_cifar10c_severity(labels: np.ndarray, severity: int) -> np.ndarray:
    """
    Some releases ship labels.npy with 10000 rows (CIFAR-10 test order, shared across severities).
    Others use 50000 rows (one block of 10000 labels per severity 1..5). severity in 1..5.
    """
    if severity < 1 or severity > 5:
        raise ValueError("severity must be in 1..5")
    n = len(labels)
    if n == 10000:
        return labels
    if n == 50000:
        start = (severity - 1) * 10000
        return labels[start : start + 10000]
    raise ValueError(f"labels must have length 10000 or 50000, got {n}")


def load_cifar10c_corruption_full_from_tar(
    tar_path: str | Path,
    corruption: str,
    inner_prefix: str = "CIFAR-10-C",
) -> np.ndarray:
    """Load full (50000, 32, 32, 3) array for one corruption from the tar."""
    stem = corruption[:-4] if corruption.lower().endswith(".npy") else corruption
    fname = f"{stem}.npy"
    pref = inner_prefix.strip("/").rstrip("/")
    with tarfile.open(tar_path, "r:*") as tar:
        for path in (f"{pref}/{fname}", fname):
            try:
                arr = _read_npy_member(tar, path)
                break
            except KeyError:
                continue
        else:
            raise FileNotFoundError(f"No {fname} under prefix {pref!r} in {tar_path}")
    if arr.shape[0] != 50000:
        raise ValueError(f"Expected 50000 images in {fname}, got shape {arr.shape}")
    return arr


def list_cifar10c_corruption_files(cifar10c_root: str | Path) -> list[str]:
    """Return basenames of .npy files excluding labels.npy."""
    root = Path(cifar10c_root)
    out = []
    for p in sorted(root.glob("*.npy")):
        if p.name == "labels.npy":
            continue
        out.append(p.name)
    return out
