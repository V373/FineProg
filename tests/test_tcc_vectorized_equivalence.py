import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.loss.tcc.deterministic_alignment import (
    compute_deterministic_alignment_loss,
    compute_deterministic_alignment_loss_batched,
)


@pytest.mark.parametrize("shape", [(5, 7, 16), (12, 20, 128)])
@pytest.mark.parametrize("loss_type", ["classification", "regression_mse_var"])
def test_batched_alignment_matches_pair_loop(shape, loss_type):
    batch_size, num_steps, embedding_dim = shape
    torch.manual_seed(42)
    base = torch.randn(batch_size, num_steps, embedding_dim, dtype=torch.float32)
    steps = torch.arange(num_steps, dtype=torch.long).expand(batch_size, -1)
    seq_lens = torch.full((batch_size,), num_steps, dtype=torch.long)

    loop_embs = base.clone().requires_grad_(True)
    batched_embs = base.clone().requires_grad_(True)
    kwargs = {
        "steps": steps,
        "seq_lens": seq_lens,
        "similarity_type": "l2",
        "loss_type": loss_type,
        "temperature": 0.1,
    }

    loop_loss = compute_deterministic_alignment_loss(embs=loop_embs, **kwargs)
    batched_loss = compute_deterministic_alignment_loss_batched(
        embs=batched_embs, **kwargs
    )
    loop_grad, = torch.autograd.grad(loop_loss, loop_embs)
    batched_grad, = torch.autograd.grad(batched_loss, batched_embs)

    torch.testing.assert_close(loop_loss, batched_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(loop_grad, batched_grad, rtol=1e-4, atol=1e-6)
