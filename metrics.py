"""
Uncertainty and calibration metrics for MC softmax samples.

- MI: epistemic term H(p_bar) - E_t[H(p_t)].
- Brier: mean squared error between predictive probabilities and one-hot labels
  (sum over classes, mean over samples).
- ACE: sort confidences ascending, split into M equal-count bins (tensor_split),
  per-bin |mean_acc - mean_conf|, then unweighted mean over bins (not weighted ECE).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Number of equal-count (approx.) bins for ACE
ACE_NUM_BINS = 15


def _predictive_stats(probs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared predictive stats from MC probabilities."""
    p_mean = probs.mean(dim=1)
    preds = p_mean.argmax(dim=-1)
    conf = p_mean.max(dim=-1).values
    correct = (preds == targets).float()
    return p_mean, preds, conf, correct


def _sorted_equal_count_chunks(
    confidences: torch.Tensor,
    correct: torch.Tensor,
    num_bins: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    order = torch.argsort(confidences)
    conf_sorted = confidences[order]
    corr_sorted = correct.float()[order]
    return torch.tensor_split(conf_sorted, num_bins), torch.tensor_split(corr_sorted, num_bins)


def entropy_from_probs(probs: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy; probs assumed normalized."""
    p = probs.clamp(min=eps)
    return -(p * p.log()).sum(dim=dim)


def mutual_information_mc(probs: torch.Tensor) -> torch.Tensor:
    """
    probs: (N, T, C) softmax probabilities over T MC samples.
    Returns per-sample MI, shape (N,).
    """
    p_mean = probs.mean(dim=1)
    h_mean = entropy_from_probs(p_mean, dim=-1)
    h_each = entropy_from_probs(probs, dim=-1)
    expected_h = h_each.mean(dim=1)
    return h_mean - expected_h


def brier_score_multiclass(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    probs: (N, C) mean predictive probabilities (e.g. mean over MC).
    targets: (N,) class indices.
    Scalar: mean over N of sum_c (p_c - one_hot_c)^2.
    """
    n_classes = probs.size(-1)
    one_hot = F.one_hot(targets, num_classes=n_classes).to(dtype=probs.dtype, device=probs.device)
    err = (probs - one_hot).pow(2).sum(dim=-1)
    return err.mean()


def adaptive_calibration_error(
    confidences: torch.Tensor,
    correct: torch.Tensor,
    num_bins: int = ACE_NUM_BINS,
) -> torch.Tensor:
    """
    ACE with equal-count bins (via tensor_split on sorted indices).

    confidences: (N,) max prob of mean predictive (or deterministic conf).
    correct: (N,) bool or 0/1 tensor, 1 if prediction matches label.

    Sort by confidence ascending, split into num_bins chunks, compute
    mean(|acc_bin - conf_bin|) with unweighted mean across bins.
    """
    if confidences.dim() != 1 or correct.dim() != 1:
        raise ValueError("confidences and correct must be 1D")
    if confidences.numel() != correct.numel():
        raise ValueError("confidences and correct length mismatch")

    conf_chunks, corr_chunks = _sorted_equal_count_chunks(confidences, correct, num_bins)
    errs = []
    for cb, ab in zip(conf_chunks, corr_chunks):
        if cb.numel() == 0:
            continue
        errs.append((ab.mean() - cb.mean()).abs())

    if not errs:
        return torch.tensor(0.0, device=confidences.device, dtype=confidences.dtype)

    return torch.stack(errs).mean()


def ace_reliability_bins(
    confidences: torch.Tensor,
    correct: torch.Tensor,
    num_bins: int = ACE_NUM_BINS,
) -> list[dict[str, float | int]]:
    """
    Equal-count bins (same rule as ACE): sort conf ascending, tensor_split into num_bins.
    Each bin: mean confidence, mean accuracy (fraction correct), count, and |acc - conf|.
    Use for reliability diagrams (plot mean_confidence vs mean_accuracy per bin).
    """
    if confidences.dim() != 1 or correct.dim() != 1:
        raise ValueError("confidences and correct must be 1D")
    if confidences.numel() != correct.numel():
        raise ValueError("length mismatch")

    conf_chunks, corr_chunks = _sorted_equal_count_chunks(confidences, correct, num_bins)
    out: list[dict[str, float | int]] = []
    for bi, (cb, ab) in enumerate(zip(conf_chunks, corr_chunks)):
        if cb.numel() == 0:
            continue
        mc = float(cb.mean().item())
        ma = float(ab.mean().item())
        out.append(
            {
                "bin": len(out),
                "mean_confidence": mc,
                "mean_accuracy": ma,
                "n": int(cb.numel()),
                "abs_gap": float(abs(ma - mc)),
            }
        )
    return out


def diagnostics_bundle_from_mc_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = ACE_NUM_BINS,
) -> dict:
    """
    probs: (N, T, C). Returns serializable-ish dict + tensors for torch.save:
    ace_bins, scalars, mean_probs (float16), targets (int64), pred_entropy (float32) from mean predictive.
    """
    p_mean, preds, conf, correct = _predictive_stats(probs, targets)

    bins = ace_reliability_bins(conf, correct, num_bins=num_bins)
    ace = adaptive_calibration_error(conf, correct, num_bins=num_bins)
    ent = entropy_from_probs(p_mean, dim=-1)

    return {
        "ace_bins": bins,
        "ace": float(ace.item()),
        "mi": float(mutual_information_mc(probs).mean().item()),
        "brier": float(brier_score_multiclass(p_mean, targets).item()),
        "accuracy": float((preds == targets).float().mean().item()),
        "mean_probs": p_mean.detach().cpu().half(),
        "targets": targets.detach().cpu().long(),
        "pred_entropy": ent.detach().cpu().float(),
    }


def metrics_from_mc_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = ACE_NUM_BINS,
) -> dict[str, torch.Tensor]:
    """
    probs: (N, T, C) MC softmax samples.
    targets: (N,) integer labels.

    Uses mean predictive p_bar for confidence (max) and Brier; predictions argmax(p_bar).
    """
    p_mean, preds, conf, correct = _predictive_stats(probs, targets)

    mi = mutual_information_mc(probs).mean()
    brier = brier_score_multiclass(p_mean, targets)
    ace = adaptive_calibration_error(conf, correct, num_bins=num_bins)

    acc = (preds == targets).float().mean()

    return {
        "mi": mi,
        "brier": brier,
        "ace": ace,
        "accuracy": acc,
    }
