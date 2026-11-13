import itertools

import torch
from torch.nn import functional as F

from fineprog.losses import replay_loss, semantic_set_loss, tcc_loss, temporal_triplet_loss


def test_tcc_matches_direct_ordered_pair_cycles_and_gradients():
    torch.manual_seed(1)
    z = torch.randn(3, 4, 5, requires_grad=True)
    steps = torch.tensor([[0, 1, 4, 7], [1, 2, 4, 6], [0, 3, 5, 8]])
    lengths = torch.tensor([8, 9, 10])
    terms = []
    for i, j in itertools.permutations(range(3), 2):
        logits = -(z[i, :, None] - z[j, None]).square().sum(-1) / (5 * .4)
        neighbors = logits.softmax(-1) @ z[j]
        reverse = -(neighbors[:, None] - z[i, None]).square().sum(-1) / (5 * .4)
        beta = reverse.softmax(-1)
        time = steps[i].float() / lengths[i]
        mean = (beta * time).sum(-1)
        variance = (beta * (time - mean[:, None]).square()).sum(-1).clamp_min(1e-8)
        terms.append(((time - mean).square() / variance + .001 * variance.log()).mean())
    expected = torch.stack(terms).mean()
    actual = tcc_loss(z, steps, lengths, temperature=.4)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, z, retain_graph=True)[0],
                               torch.autograd.grad(expected, z)[0], atol=2e-6, rtol=2e-5)


def test_triplet_matches_explicit_temporal_order_and_duplicate_times():
    torch.manual_seed(2)
    z = torch.randn(2, 4, 3, requires_grad=True)
    steps = torch.tensor([[1, 1, 4, 8], [0, 2, 3, 7]])
    terms = []
    for row in range(2):
        for anchor in range(4):
            for first, second in itertools.combinations([x for x in range(4) if x != anchor], 2):
                ga, gb = abs(int(steps[row, anchor] - steps[row, first])), abs(int(steps[row, anchor] - steps[row, second]))
                if ga == gb:
                    continue
                positive, negative = (first, second) if ga < gb else (second, first)
                terms.append(((z[row, anchor] - z[row, positive]).square().sum()
                              - (z[row, anchor] - z[row, negative]).square().sum() + .2).clamp_min(0))
    expected = torch.stack(terms).mean()
    actual = temporal_triplet_loss(z, steps)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, z, retain_graph=True)[0],
                               torch.autograd.grad(expected, z)[0])
    loss = replay_loss(z, steps, torch.tensor([9, 8]), temperature=.4)
    assert torch.isfinite(loss)


def test_set_loss_uses_all_reference_mass_and_only_rg_query_gradients():
    query = torch.tensor([[0.2, .4], [.5, .8], [.9, .2]], requires_grad=True)
    refs = torch.tensor([[0., 0.], [.1, .1], [.5, 1.], [1., 0.]], requires_grad=True)
    labels, ref_labels = torch.tensor([0, 1, 2]), torch.tensor([0, 0, 1, 2])
    logits = -(query[:, None] - refs[None]).square().sum(-1) / .128
    expected = -(logits.softmax(-1)[0, :2].sum().log() + logits.softmax(-1)[2, 3].log()) / 2
    actual = semantic_set_loss(query, refs, labels, ref_labels)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert query.grad[1].count_nonzero() == 0
    assert query.grad[[0, 2]].abs().sum() > 0
    assert refs.grad[2].abs().sum() > 0


def test_empty_set_and_no_temporal_triplets_keep_connected_zero():
    query = torch.randn(2, 3, requires_grad=True)
    refs = torch.randn(2, 3, requires_grad=True)
    loss = semantic_set_loss(query, refs, torch.tensor([1, -1]), torch.tensor([0, 2]))
    loss.backward()
    assert loss == 0 and query.grad.count_nonzero() == refs.grad.count_nonzero() == 0
    assert temporal_triplet_loss(query[None], torch.zeros(1, 2, dtype=torch.long)) == 0
