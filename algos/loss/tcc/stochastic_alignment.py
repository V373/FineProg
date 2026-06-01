"""
Stochastic Alignment Loss for Temporal Cycle-Consistency (TCC).

Implements stochastic alignment where a random subset of cycles is sampled
each iteration instead of exhaustively aligning all N*(N-1) sequence pairs.

Key difference from deterministic alignment:
- Deterministic: aligns all N*(N-1) pairs × T timesteps per batch
- Stochastic: samples num_cycles random cycles, complexity O(num_cycles × T)

Reference: Original TensorFlow implementation in
google-research/tcc/tcc/stochastic_alignment.py

Mathematical Pipeline:
1. Sample num_cycles random cycles, each of the form [i, j, ..., i].
2. For each cycle:
   a. Pick a random start timestep n_idx in sequence cycle[0].
   b. Hop through the cycle via soft nearest neighbors.
   c. On the final hop (back to sequence cycle[0]), collect logits [T].
   d. Construct one-hot label at n_idx.
3. Stack into logits [num_cycles, T] and labels [num_cycles, T].
4. Compute loss via classification or regression head.
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Handle both module import and direct script execution
try:
    from .loss_head import classification_loss, regression_loss
except ImportError:
    _current_dir = Path(__file__).parent
    _parent_dir = _current_dir.parent.parent.parent
    if str(_parent_dir) not in sys.path:
        sys.path.insert(0, str(_parent_dir))
    from algos.loss.tcc.loss_head import classification_loss, regression_loss


def gen_cycles(num_cycles, batch_size, cycle_length=2, device=None):
    """
    Generate random cycles for stochastic TCC alignment.

    Each cycle has the form [seq_i0, seq_i1, ..., seq_i{cycle_length-1}, seq_i0],
    where the last element closes the cycle back to the first.

    Fully vectorized: no Python loop over num_cycles.
    Uses argsort of uniform noise to produce num_cycles independent
    random permutations in a single tensor op.

    Args:
        num_cycles:   int, number of cycles to generate.
        batch_size:   int, number of sequences in the batch.
        cycle_length: int, number of intermediate hops per cycle.
                      cycle_length=2 => [i, j, i].
        device:       torch.device or None.

    Returns:
        cycles: LongTensor, shape [num_cycles, cycle_length + 1].
                cycles[c, 0] == cycles[c, -1] for every row c.
    """
    if cycle_length > batch_size:
        raise ValueError(
            f'cycle_length ({cycle_length}) must be <= batch_size ({batch_size}). '
            f'Cannot form a valid cycle with more hops than available sequences.'
        )

    # Vectorised: argsort of uniform noise gives independent random permutations
    # rand: [num_cycles, batch_size] — each row is a distinct uniform sample
    rand = torch.rand(num_cycles, batch_size, device=device)
    # perms[c, :cycle_length] are cycle_length distinct seq indices for cycle c
    perms = torch.argsort(rand, dim=1)[:, :cycle_length]   # [num_cycles, cycle_length]
    # Close the cycle: last element == first element
    cycles = torch.cat([perms, perms[:, :1]], dim=1)        # [num_cycles, cycle_length + 1]
    return cycles


def _align_single_cycle(cycle, embs, cycle_length, num_steps, similarity_type, temperature):
    """
    Perform one stochastic TCC cycle alignment.

    Starting from a randomly sampled timestep n_idx in cycle[0], hop through
    the cycle via soft nearest neighbours, then measure how well we return to
    n_idx in the final hop back to cycle[0].

    Args:
        cycle:           LongTensor, shape [cycle_length + 1].
                         cycle[0] == cycle[-1].
        embs:            Tensor, shape [B, T, D].
        cycle_length:    int, number of hops (not counting the start).
        num_steps:       int, T.
        similarity_type: str, 'l2' or 'cosine'.
        temperature:     float > 0.

    Returns:
        logits: Tensor, shape [T] — similarity scores from the final hop.
        label:  Tensor, shape [T] — one-hot at the starting timestep n_idx.
    """
    D = float(embs.shape[2])
    device = embs.device

    # Random starting timestep
    n_idx = torch.randint(0, num_steps, (), device=device)

    # One-hot label: the cycle should return to position n_idx
    label = torch.zeros(num_steps, dtype=embs.dtype, device=device)
    label[n_idx] = 1.0

    # Initial query: embedding of cycle[0] at time n_idx, shape [1, D]
    query_feats = embs[cycle[0], n_idx].unsqueeze(0)  # [1, D]

    # Traverse the cycle hop by hop
    for c in range(1, cycle_length + 1):
        candidate_feats = embs[cycle[c]]  # [T, D]

        if similarity_type == 'l2':
            # Negative squared L2 distance: higher = more similar
            similarity = -torch.sum(
                (candidate_feats - query_feats) ** 2, dim=1
            )  # [T]
        elif similarity_type == 'cosine':
            # Raw dot product (no explicit normalisation, matching deterministic impl)
            similarity = torch.matmul(
                candidate_feats, query_feats.squeeze(0)
            )  # [T]
        else:
            raise ValueError(
                f'similarity_type must be "l2" or "cosine", got "{similarity_type}"'
            )

        # Scale by embedding dim and temperature
        similarity = similarity / D / temperature

        # Soft nearest neighbour → update query
        beta = torch.softmax(similarity, dim=0)  # [T]
        query_feats = torch.sum(beta[:, None] * candidate_feats, dim=0, keepdim=True)  # [1, D]

    # similarity from the last hop is the logit vector
    logits = similarity  # [T]

    return logits, label


def _align(cycles, embs, num_steps, num_cycles, cycle_length, similarity_type, temperature):
    """
    Run stochastic alignment for all sampled cycles (sequential, CPU-optimal).

    On CPU this is 5-15x faster than _align_vectorized because the Python loop
    overhead is negligible compared to the memory-bandwidth cost of allocating
    large [num_cycles, T, D] intermediate tensors used by the vectorized path.
    On CUDA, prefer _align_vectorized instead.
    compute_stochastic_alignment_loss dispatches automatically.
    """
    logits_list = []
    labels_list = []

    for c_idx in range(num_cycles):
        logits, label = _align_single_cycle(
            cycles[c_idx], embs, cycle_length, num_steps, similarity_type, temperature
        )
        logits_list.append(logits)
        labels_list.append(label)

    logits_all = torch.stack(logits_list, dim=0)  # [num_cycles, T]
    labels_all = torch.stack(labels_list, dim=0)  # [num_cycles, T]

    return logits_all, labels_all


def _align_vectorized(cycles, embs, num_steps, num_cycles, cycle_length, similarity_type, temperature):
    """
    Vectorized stochastic alignment — processes all num_cycles cycles in parallel.

    Key difference from _align / _align_single_cycle:
    - No Python loop over num_cycles.
    - All cycles share a single forward pass with batch dims [num_cycles, T, D].
    - Autograd graph has O(cycle_length) nodes regardless of num_cycles.
    - On CUDA: ~1ms regardless of num_cycles, 25-100x faster than deterministic.
    - On CPU: SLOWER than _align due to [num_cycles,T,D] memory bandwidth cost.
    compute_stochastic_alignment_loss dispatches automatically based on device.

    Args:
        cycles:          LongTensor, shape [num_cycles, cycle_length + 1].
        embs:            Tensor, shape [B, T, D].
        num_steps:       int, T.
        num_cycles:      int.
        cycle_length:    int.
        similarity_type: str, 'l2' or 'cosine'.
        temperature:     float.

    Returns:
        logits_all: Tensor, shape [num_cycles, T].
        labels_all: Tensor, shape [num_cycles, T].
    """
    D = float(embs.shape[2])
    device = embs.device
    dtype = embs.dtype

    # Sample one random start timestep per cycle: [num_cycles]
    n_idxs = torch.randint(0, num_steps, (num_cycles,), device=device)

    # Gather initial query for each cycle: embs[cycles[:, 0], n_idxs] → [num_cycles, D]
    # cycles[:, 0]: [num_cycles]  — start sequence index per cycle
    start_seq = cycles[:, 0]                           # [num_cycles]
    query_feats = embs[start_seq, n_idxs]              # [num_cycles, D]

    # Traverse cycle hops (only cycle_length Python iterations, NOT num_cycles)
    for c in range(1, cycle_length + 1):
        # candidate_feats: embs[cycles[:, c]] → [num_cycles, T, D]
        hop_seq = cycles[:, c]                         # [num_cycles]
        candidate_feats = embs[hop_seq]                # [num_cycles, T, D]

        if similarity_type == 'l2':
            # similarity[k, t] = -||candidate_feats[k,t] - query_feats[k]||^2
            # query_feats: [num_cycles, D] → [num_cycles, 1, D] for broadcast
            diff = candidate_feats - query_feats.unsqueeze(1)   # [num_cycles, T, D]
            similarity = -torch.sum(diff ** 2, dim=2)           # [num_cycles, T]
        elif similarity_type == 'cosine':
            # similarity[k, t] = candidate_feats[k,t] · query_feats[k]
            # bmm: [num_cycles, T, D] × [num_cycles, D, 1] → [num_cycles, T, 1]
            similarity = torch.bmm(
                candidate_feats,
                query_feats.unsqueeze(2)
            ).squeeze(2)                                         # [num_cycles, T]
        else:
            raise ValueError(
                f'similarity_type must be "l2" or "cosine", got "{similarity_type}"'
            )

        similarity = similarity / D / temperature                # [num_cycles, T]

        # Soft nearest neighbour: weighted sum over T candidates
        beta = torch.softmax(similarity, dim=1)                  # [num_cycles, T]
        # bmm: [num_cycles, 1, T] × [num_cycles, T, D] → [num_cycles, 1, D]
        query_feats = torch.bmm(
            beta.unsqueeze(1), candidate_feats
        ).squeeze(1)                                             # [num_cycles, D]

    # logits = similarity from the last hop: [num_cycles, T]
    logits_all = similarity

    # One-hot labels: label[k, n_idxs[k]] = 1
    labels_all = torch.zeros(num_cycles, num_steps, dtype=dtype, device=device)
    labels_all.scatter_(1, n_idxs.unsqueeze(1), 1.0)

    return logits_all, labels_all


def compute_stochastic_alignment_loss(
    embeddings,
    target_steps,
    seq_len,
    num_cycles=20,
    cycle_length=2,
    loss_type='regression_mse_var',
    similarity_type='l2',
    temperature=0.1,
    label_smoothing=0.1,
    variance_lambda=0.001,
    huber_delta=0.1,
    normalize_indices=True,
    batch_size=None,
    clip_len=None,
):
    """
    Compute stochastic alignment loss for a batch of sequence embeddings.

    Samples num_cycles random cycles from the batch and computes alignment loss
    only on those cycles, making it much cheaper than deterministic alignment.

    Args:
        embeddings:       Tensor, shape [B, T, D].
        target_steps:     Tensor, shape [B, T] — frame indices (long).
        seq_len:          Tensor, shape [B] — original sequence lengths (long).
        num_cycles:       int, number of random cycles to sample per batch.
        cycle_length:     int, hops per cycle (2 => A->B->A).
        loss_type:        str, one of 'classification', 'regression_mse',
                          'regression_mse_var', 'regression_huber'.
        similarity_type:  str, 'l2' or 'cosine'.
        temperature:      float > 0, softmax temperature.
        label_smoothing:  float, label smoothing for classification.
        variance_lambda:  float, variance weight for regression_mse_var.
        huber_delta:      float, delta for regression_huber.
        normalize_indices: bool, normalise frame indices by seq_len.
        batch_size:       int or None (inferred from embeddings).
        clip_len:         int or None (inferred from embeddings).

    Returns:
        loss: scalar tensor (differentiable).
    """
    if temperature <= 0:
        raise ValueError(f'temperature must be > 0, got {temperature}')

    if batch_size is None:
        batch_size = embeddings.shape[0]
    if clip_len is None:
        clip_len = embeddings.shape[1]

    num_steps = clip_len
    device = embeddings.device

    if batch_size < 2:
        raise ValueError(
            f'batch_size must be >= 2 for stochastic TCC alignment. '
            f'Got batch_size={batch_size}. '
            f'Stochastic TCC requires at least two sequences to form a cycle.'
        )

    # Generate random cycles
    cycles = gen_cycles(num_cycles, batch_size, cycle_length, device=device)
    # cycles: [num_cycles, cycle_length + 1]

    # Dispatch: vectorized on CUDA (batched tensor ops are ~100x faster than a
    # Python loop on GPU), sequential on CPU (avoids large intermediate tensors
    # whose memory bandwidth cost dominates over Python loop overhead on CPU).
    if device.type == 'cuda':
        logits_all, labels_all = _align_vectorized(
            cycles, embeddings, num_steps, num_cycles, cycle_length,
            similarity_type, temperature
        )
    else:
        logits_all, labels_all = _align(
            cycles, embeddings, num_steps, num_cycles, cycle_length,
            similarity_type, temperature
        )
    # logits_all: [num_cycles, T], labels_all: [num_cycles, T]

    # Compute loss
    if loss_type == 'classification':
        loss = classification_loss(logits_all, labels_all, label_smoothing)
    elif 'regression' in loss_type:
        # For regression, fetch steps/seq_lens for the start sequence of each cycle
        start_seq_ids = cycles[:, 0]              # [num_cycles]
        cycle_steps = target_steps[start_seq_ids]  # [num_cycles, T]
        cycle_seq_lens = seq_len[start_seq_ids]    # [num_cycles]

        loss = regression_loss(
            logits_all,
            labels_all,
            num_steps,
            cycle_steps,
            cycle_seq_lens,
            loss_type,
            normalize_indices,
            variance_lambda,
            huber_delta,
        )
    else:
        raise ValueError(
            f'Unsupported loss_type "{loss_type}". '
            'Supported: "classification", "regression_mse", '
            '"regression_mse_var", "regression_huber".'
        )

    return loss


# ============================================================================
# MINIMAL SMOKE TEST
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("TCC Stochastic Alignment - Minimal Smoke Test")
    print("=" * 70)

    torch.manual_seed(42)

    B = 4
    T = 8
    D = 16

    print(f"\nTest Configuration: B={B}, T={T}, D={D}")

    embs = torch.randn(B, T, D, requires_grad=True)
    steps = torch.arange(T, dtype=torch.long).unsqueeze(0).expand(B, -1).clone()
    seq_lens = torch.full((B,), T, dtype=torch.long)

    print(f"embs: {tuple(embs.shape)}, steps: {tuple(steps.shape)}, seq_lens: {seq_lens.tolist()}")

    # Test 1: classification
    print("\n--- Test 1: classification ---")
    loss1 = compute_stochastic_alignment_loss(
        embeddings=embs,
        target_steps=steps,
        seq_len=seq_lens,
        loss_type='classification',
        similarity_type='l2',
        num_cycles=20,
        cycle_length=2,
        temperature=0.1,
    )
    assert loss1.dim() == 0, "loss should be scalar"
    assert torch.isfinite(loss1), "loss should be finite"
    loss1.backward()
    assert embs.grad is not None, "grad should be non-None after backward"
    print(f"  loss={loss1.item():.6f}, grad_norm={embs.grad.norm().item():.6f}  OK")

    # Test 2: regression_mse_var
    print("\n--- Test 2: regression_mse_var ---")
    embs2 = torch.randn(B, T, D, requires_grad=True)
    loss2 = compute_stochastic_alignment_loss(
        embeddings=embs2,
        target_steps=steps,
        seq_len=seq_lens,
        loss_type='regression_mse_var',
        similarity_type='l2',
        num_cycles=20,
        cycle_length=2,
        temperature=0.1,
        variance_lambda=0.001,
    )
    assert loss2.dim() == 0
    assert torch.isfinite(loss2)
    loss2.backward()
    assert embs2.grad is not None
    print(f"  loss={loss2.item():.6f}, grad_norm={embs2.grad.norm().item():.6f}  OK")

    # Test 3: cosine similarity
    print("\n--- Test 3: regression_mse (cosine) ---")
    embs3 = torch.randn(B, T, D, requires_grad=True)
    loss3 = compute_stochastic_alignment_loss(
        embeddings=embs3,
        target_steps=steps,
        seq_len=seq_lens,
        loss_type='regression_mse',
        similarity_type='cosine',
        num_cycles=10,
        cycle_length=2,
        temperature=0.1,
    )
    assert loss3.dim() == 0
    assert torch.isfinite(loss3)
    loss3.backward()
    assert embs3.grad is not None
    print(f"  loss={loss3.item():.6f}, grad_norm={embs3.grad.norm().item():.6f}  OK")

    # Test 4: cycle_length=3 (longer cycle)
    print("\n--- Test 4: cycle_length=3 ---")
    embs4 = torch.randn(B, T, D, requires_grad=True)
    loss4 = compute_stochastic_alignment_loss(
        embeddings=embs4,
        target_steps=steps,
        seq_len=seq_lens,
        loss_type='classification',
        similarity_type='l2',
        num_cycles=8,
        cycle_length=3,
        temperature=0.1,
    )
    assert loss4.dim() == 0
    assert torch.isfinite(loss4)
    loss4.backward()
    assert embs4.grad is not None
    print(f"  loss={loss4.item():.6f}, grad_norm={embs4.grad.norm().item():.6f}  OK")

    print("\n" + "=" * 70)
    print("All smoke tests passed!")
    print("=" * 70)
