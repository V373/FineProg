"""Analytical checks for semantic set mass, ignored labels and gradients."""

import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fineprog.algos.loss.loss_semantic_set import semantic_set_loss


def test_multiple_positives_use_set_mass_and_keep_wrong_classes_in_denominator():
    # Equal distances: two of four expert references belong to R, so P(R)=1/2.
    query = torch.zeros(1, 2, requires_grad=True)
    refs = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], requires_grad=True)
    out = semantic_set_loss(query, refs, torch.tensor([0]), torch.tensor([0, 0, 1, 2]))
    torch.testing.assert_close(out["loss"], torch.tensor(math.log(2)))
    assert out["metrics"]["positive_mass"].item() == pytest.approx(0.5)
    for key, value in (("mass_R", .5), ("mass_M", .25), ("mass_G", .25)):
        assert out["metrics"][key].item() == pytest.approx(value)


def test_squared_distance_temperature_and_both_side_gradients():
    query = torch.tensor([[0.]], requires_grad=True)
    refs = torch.tensor([[1.], [2.]], requires_grad=True)
    # exp(-1/2) / (exp(-1/2)+exp(-4/2)) = sigmoid(1.5).
    out = semantic_set_loss(query, refs, torch.tensor([0]), torch.tensor([0, 2]), tau=2.)
    assert out["loss"].item() == pytest.approx(math.log1p(math.exp(-1.5)))
    out["loss"].backward()
    for tensor in (query, refs):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def test_m_and_unknown_queries_and_references_do_not_contribute():
    query = torch.tensor([[0.], [100.], [-100.], [2.]], requires_grad=True)
    refs = torch.tensor([[0.], [1.], [2.], [0.]], requires_grad=True)
    out = semantic_set_loss(query, refs, torch.tensor([0, 1, -1, 2]), torch.tensor([0, 1, 2, -1]))
    baseline = semantic_set_loss(query[[0, 3]], refs[:3], torch.tensor([0, 2]), torch.tensor([0, 1, 2]))
    torch.testing.assert_close(out["loss"], baseline["loss"])
    assert out["metrics"]["valid_queries"] == 2
    assert out["metrics"]["r_queries"] == out["metrics"]["g_queries"] == 1
    out["loss"].backward()
    assert query.grad[1:3].count_nonzero() == 0
    assert refs.grad[3].count_nonzero() == 0


@pytest.mark.parametrize("qlabels,rlabels", [
    ([0], [1, 2]),  # No positive set.
    ([0], [0, 0]),  # No negative set.
    ([1, -1, 3], [0, 1, 2]),
    ([0, 2], [-1, 3]),
    ([0], []),
    ([], [0, 1, 2]),
])
def test_no_effective_query_returns_connected_zero(qlabels, rlabels):
    query = torch.randn(len(qlabels), 2, requires_grad=True)
    refs = torch.randn(len(rlabels), 2, requires_grad=True)
    out = semantic_set_loss(query, refs, torch.tensor(qlabels), torch.tensor(rlabels))
    assert out["loss"].item() == 0
    assert all(torch.isfinite(value) and value == 0 for value in out["metrics"].values())
    out["loss"].backward()
    assert query.grad is not None and query.grad.count_nonzero() == 0
    assert refs.grad is not None and refs.grad.count_nonzero() == 0


def test_missing_positive_skips_only_affected_class():
    out = semantic_set_loss(torch.zeros(2, 1), torch.ones(2, 1),
                            torch.tensor([0, 2]), torch.tensor([0, 1]))
    assert out["metrics"]["valid_queries"] == out["metrics"]["r_queries"] == 1
    assert out["metrics"]["g_queries"] == 0
    assert out["loss"].item() == pytest.approx(math.log(2))


def test_fp32_under_autocast_is_stable_for_extreme_logits():
    query = torch.tensor([[100.]], dtype=torch.bfloat16, requires_grad=True)
    refs = torch.tensor([[0.], [100.]], dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = semantic_set_loss(query, refs, torch.tensor([0]), torch.tensor([0, 2]), tau=.001)
    assert out["loss"].dtype == torch.float32
    assert out["loss"].item() == pytest.approx(10_000_000., rel=1e-6)
    out["loss"].backward()
    assert torch.isfinite(query.grad).all() and torch.isfinite(refs.grad).all()


@pytest.mark.parametrize("tau", [0., -1., float("nan"), float("inf")])
def test_invalid_temperature(tau):
    with pytest.raises(ValueError, match="tau"):
        semantic_set_loss(torch.zeros(1, 1), torch.zeros(1, 1),
                          torch.tensor([0]), torch.tensor([0]), tau=tau)
