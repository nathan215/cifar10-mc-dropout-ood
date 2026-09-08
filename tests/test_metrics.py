"""
Small, fast, CPU-only unit tests for metrics.py (MC-dropout uncertainty /
calibration metrics). No dataset, model, or GPU is needed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from metrics import (
    adaptive_calibration_error,
    brier_score_multiclass,
    metrics_from_mc_probs,
    mutual_information_mc,
)


def test_brier_score_one_hot_perfect_is_zero():
    # probs exactly match the one-hot label -> squared error is 0 everywhere.
    targets = torch.tensor([0, 1, 2, 3])
    probs = F.one_hot(targets, num_classes=4).float()

    brier = brier_score_multiclass(probs, targets)

    assert torch.isclose(brier, torch.tensor(0.0), atol=1e-6)


def test_brier_score_worst_case_is_two():
    # All predictive mass on the wrong class: (1-0)^2 + (0-1)^2 = 2 for every sample.
    targets = torch.tensor([0, 1, 2])
    wrong_class = torch.tensor([1, 2, 0])
    probs = F.one_hot(wrong_class, num_classes=3).float()

    brier = brier_score_multiclass(probs, targets)

    assert torch.isclose(brier, torch.tensor(2.0), atol=1e-6)


def test_brier_score_random_within_bounds():
    # Brier score (sum-over-classes form) is bounded in [0, 2] for any point on
    # the probability simplex compared against a one-hot label.
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(25, 6), dim=-1)
    targets = torch.randint(0, 6, (25,))

    brier = brier_score_multiclass(probs, targets)

    assert 0.0 <= brier.item() <= 2.0


def test_ace_perfectly_calibrated_bins_is_zero():
    # Two equal-size groups where mean accuracy exactly equals mean confidence
    # in each group; with num_bins=2 the groups land in separate bins exactly
    # (confidences are tied within a group, distinct across groups).
    confidences = torch.tensor([0.6] * 10 + [0.9] * 10)
    correct = torch.tensor(
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        dtype=torch.float32,
    )

    ace = adaptive_calibration_error(confidences, correct, num_bins=2)

    assert torch.isclose(ace, torch.tensor(0.0), atol=1e-6)


def test_adaptive_calibration_error_invalid_shape_raises():
    confidences = torch.rand(5, 2)  # not 1D
    correct = torch.randint(0, 2, (5,))

    with pytest.raises(ValueError):
        adaptive_calibration_error(confidences, correct)


def test_mutual_information_zero_when_mc_samples_identical():
    # If every one of the T forward passes produces the same softmax vector,
    # there is no epistemic disagreement, so MI must be (numerically) zero.
    torch.manual_seed(1)
    n, t, c = 4, 6, 5
    v = torch.softmax(torch.randn(n, c), dim=-1)
    probs = v.unsqueeze(1).expand(n, t, c)

    mi = mutual_information_mc(probs)

    assert mi.shape == (n,)
    assert torch.allclose(mi, torch.zeros(n), atol=1e-5)


def test_mutual_information_positive_when_mc_samples_disagree():
    # Alternate between two very different softmax vectors across T passes:
    # the mean prediction is high-entropy while each individual pass is
    # low-entropy, so MI (= H(mean) - mean(H)) must be strictly positive.
    n, t, c = 3, 4, 2
    v1 = torch.tensor([0.95, 0.05])
    v2 = torch.tensor([0.05, 0.95])
    probs = torch.empty(n, t, c)
    for step in range(t):
        probs[:, step, :] = v1 if step % 2 == 0 else v2

    mi = mutual_information_mc(probs)

    assert mi.shape == (n,)
    assert (mi > 0).all()


def test_metrics_from_mc_probs_shapes_and_ranges():
    torch.manual_seed(2)
    n, t, c = 30, 5, 4
    probs = torch.softmax(torch.randn(n, t, c), dim=-1)
    targets = torch.randint(0, c, (n,))

    out = metrics_from_mc_probs(probs, targets)

    assert set(out.keys()) == {"mi", "brier", "ace", "accuracy"}
    for key, value in out.items():
        assert isinstance(value, torch.Tensor)
        assert value.dim() == 0, f"{key} should be a scalar tensor, got shape {value.shape}"

    assert 0.0 <= out["accuracy"].item() <= 1.0
    assert 0.0 <= out["brier"].item() <= 2.0
    assert 0.0 <= out["ace"].item() <= 1.0
    # MI is H(mean) - mean(H); non-negative by concavity of entropy, up to fp noise.
    assert out["mi"].item() >= -1e-4
