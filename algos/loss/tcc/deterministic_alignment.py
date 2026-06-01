"""
Deterministic Alignment Loss for Temporal Cycle-Consistency (TCC).

Implements deterministic alignment between all pairs of sequences in a batch.
In deterministic mode, all sequence pairs (i, j) with i != j are aligned.

Reference: Original TensorFlow implementation in
google-research/tcc/tcc/deterministic_alignment.py

Mathematical Pipeline:
1. For each pair of sequences (seq_i, seq_j) with i != j:
   a. Compute similarity matrix: seq_i -> seq_j
   b. Apply softmax to get soft nearest-neighbor weights
   c. Compute soft nearest neighbors in seq_j embedding space
   d. Cycle back: soft_nn -> seq_i
   e. Construct one-hot labels for cycle-consistency
2. Aggregate losses across all pairs
3. Return scalar loss
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Handle both module import and direct script execution
try:
    from .loss_head import classification_loss, regression_loss
except ImportError:
    # If relative import fails, try adding parent directory to path
    _current_dir = Path(__file__).parent
    _parent_dir = _current_dir.parent.parent.parent
    if str(_parent_dir) not in sys.path:
        sys.path.insert(0, str(_parent_dir))
    from algos.loss.tcc.loss_head import classification_loss, regression_loss


def pairwise_l2_distance(embs1, embs2):
    """
    Computes pairwise L2 distances between all rows of embs1 and embs2.
    
    Uses the formula: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b
    
    Args:
        embs1: Tensor, embeddings of shape [M, D] where M is number of embeddings,
              D is embedding dimensionality.
        embs2: Tensor, embeddings of shape [N, D].
    
    Returns:
        dist: Tensor, pairwise squared distances of shape [M, N].
              dist[i, j] = ||embs1[i] - embs2[j]||^2
    """
    # Compute squared norms: ||x||^2 for each row
    # norm1: [M, 1]
    norm1 = torch.sum(torch.square(embs1), dim=1, keepdim=True)
    # norm2: [1, N]
    norm2 = torch.sum(torch.square(embs2), dim=1, keepdim=True).t()
    
    # Compute dot product: embs1 @ embs2^T
    # dot_prod: [M, N]
    dot_prod = torch.matmul(embs1, embs2.t())
    
    # Distance: ||a||^2 + ||b||^2 - 2*a·b
    # [M, 1] + [1, N] - 2*[M, N]
    dist = norm1 + norm2 - 2.0 * dot_prod
    
    # Clamp to avoid negative values due to floating point errors
    dist = torch.clamp(dist, min=0.0)
    
    return dist


def get_scaled_similarity(embs1, embs2, similarity_type, temperature):
    """
    Computes scaled similarity between all rows of embs1 and embs2.
    
    The similarity is scaled by:
    1. Embedding dimension D (normalization by channels)
    2. Temperature (controls softness of alignment)
    
    Formula:
    - l2 similarity:     -distance / (D * temperature)
    - cosine similarity: dot_product / (D * temperature)

    Args:
        embs1: Tensor, embeddings of shape [M, D].
        embs2: Tensor, embeddings of shape [N, D].
        similarity_type: String, 'l2' or 'cosine'.
        temperature: Float, temperature for scaling (> 0).

    Returns:
        similarity: Tensor, scaled similarity of shape [M, N].
    """
    embedding_dim = embs1.shape[1]
    channels = float(embedding_dim)
    
    if similarity_type == 'cosine':
        # Cosine similarity: dot product (assumes normalized embeddings or raw dot product)
        similarity = torch.matmul(embs1, embs2.t())  # [M, N]
    elif similarity_type == 'l2':
        # L2-based similarity: negative distance
        distances = pairwise_l2_distance(embs1, embs2)  # [M, N]
        similarity = -1.0 * distances
    else:
        raise ValueError(
            f'similarity_type must be "l2" or "cosine", got "{similarity_type}"'
        )
    
    # Normalize by embedding dimension
    similarity = similarity / channels
    
    # Scale by temperature
    similarity = similarity / temperature
    
    return similarity


def align_pair_of_sequences(embs1, embs2, similarity_type, temperature):
    """
    Align a pair of embedding sequences using cycle-consistency.
    
    This implements the core TCC alignment pipeline:
    1. Compute similarity matrix from embs1 to embs2
    2. Apply softmax to get soft nearest-neighbor weights
    3. Compute soft nearest neighbors in embs2 space
    4. Compute similarity matrix from soft_nn back to embs1
    5. Construct ground truth one-hot labels
    
    Args:
        embs1: Tensor, embeddings of shape [T, D] (reference sequence).
        embs2: Tensor, embeddings of shape [T, D] (target sequence).
               Note: embs1 and embs2 typically have the same number of timesteps
               due to batch sampling, but this isn't strictly required.
        similarity_type: String, 'l2' or 'cosine'.
        temperature: Float, temperature for softmax scaling.
    
    Returns:
        logits: Tensor, pre-softmax similarity scores after cycling back.
               Shape: [T, T] where T = number of timesteps in embs1.
               logits[i, j] = similarity(soft_nn[i], embs1[j])
        labels: Tensor, one-hot ground truth labels.
               Shape: [T, T]
               labels[i, i] = 1 (cycle should return to original position i)
    """
    max_num_steps = embs1.shape[0]
    
    # Step 1: Compute similarity from embs1 to embs2
    # sim_12[i, j] = similarity(embs1[i], embs2[j])
    sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature)  # [T, T]
    
    # Step 2: Apply softmax to get soft nearest-neighbor weights
    # softmax_sim_12[i, j] = P(embs2[j] | embs1[i])
    softmaxed_sim_12 = F.softmax(sim_12, dim=1)  # [T, T], softmax over embs2 dimension
    
    # Step 3: Compute soft nearest neighbors
    # nn_embs[i] = sum_j softmax_sim_12[i, j] * embs2[j]
    nn_embs = torch.matmul(softmaxed_sim_12, embs2)  # [T, D]
    
    # Step 4: Compute similarity from soft_nn back to embs1
    # sim_21[i, j] = similarity(nn_embs[i], embs1[j])
    sim_21 = get_scaled_similarity(nn_embs, embs1, similarity_type, temperature)  # [T, T]
    
    # Step 5: Create one-hot labels for cycle-consistency ground truth
    # labels[i, i] = 1, all others = 0
    # This means the soft nearest neighbor at position i should cycle back to embs1[i]
    labels = torch.eye(max_num_steps, dtype=embs1.dtype, device=embs1.device)  # [T, T]
    
    logits = sim_21
    
    return logits, labels


def compute_deterministic_alignment_loss(
    embs,
    steps,
    seq_lens,
    similarity_type='l2',
    loss_type='classification',
    temperature=0.1,
    label_smoothing=0.1,
    variance_lambda=0.001,
    huber_delta=0.1,
    normalize_indices=True,
):
    """
    Compute deterministic alignment loss for all sequence pairs in batch.
    
    This function aligns each pair of videos in the batch except self-pairs.
    For N videos in a batch, there are N*(N-1) alignments.
    Example: batch of size 3 creates 6 pair alignments.
    
    Workflow:
    1. Iterate over all pairs (i, j) with i != j
    2. For each pair, call align_pair_of_sequences to get logits and labels
    3. Aggregate across all pairs
    4. Compute final loss using specified loss function

    Args:
        embs: Tensor, sequential embeddings of shape [B, T, D] where:
             B = batch size (number of sequences)
             T = number of timesteps per sequence
             D = embedding dimensionality
        steps: Tensor, step/frame indices of shape [B, T].
              Used for regression losses to map similarity to frame numbers.
        seq_lens: Tensor, lengths of original sequences of shape [B].
                 Used for normalize_indices and regression loss computation.
        similarity_type: String, 'l2' or 'cosine'. Default: 'l2'
        loss_type: String, loss type. Default: 'classification'
                  Options: 'classification', 'regression_mse', 'regression_mse_var', 'regression_huber'
        temperature: Float, temperature scaling for softmax. Default: 0.1
        label_smoothing: Float, label smoothing for classification. Default: 0.1
        variance_lambda: Float, variance weight for regression_mse_var. Default: 0.001
        huber_delta: Float, delta for huber regression. Default: 0.1
        normalize_indices: Boolean, whether to normalize indices by seq_lens. Default: True

    Returns:
        loss: Scalar tensor, the computed loss (differentiable).
    """
    batch_size = embs.shape[0]
    num_steps = embs.shape[1]
    
    # Require batch size >= 2 for meaningful alignment (at least one pair)
    if batch_size < 2:
        raise ValueError(
            f'batch_size must be >= 2 for deterministic alignment. '
            f'Got batch_size={batch_size}. '
            f'Cannot align sequence with itself.'
        )
    
    logits_list = []
    labels_list = []
    steps_list = []
    seq_lens_list = []
    
    device = embs.device
    
    # Iterate over all pairs of sequences
    for i in range(batch_size):
        for j in range(batch_size):
            # Skip self-alignment (i == j)
            if i != j:
                # Extract sequence embeddings: shape [T, D]
                embs_i = embs[i]  # [T, D]
                embs_j = embs[j]  # [T, D]
                
                # Align pair (i, j)
                logits, labels = align_pair_of_sequences(
                    embs_i, embs_j, 
                    similarity_type, 
                    temperature
                )  # logits, labels: [T, T]
                
                logits_list.append(logits)
                labels_list.append(labels)
                
                # Replicate steps and seq_lens for this pair
                # For regression loss, we need these repeated for each timestep
                steps_i = steps[i:i+1]  # [1, T] -> tile to [T, T]
                steps_tiled = steps_i.expand(num_steps, -1)  # [T, T]
                steps_list.append(steps_tiled)
                
                seq_lens_i = seq_lens[i:i+1]  # [1] -> tile to [T]
                seq_lens_tiled = seq_lens_i.expand(num_steps)  # [T]
                seq_lens_list.append(seq_lens_tiled)
    
    # Concatenate all pairs
    logits_all = torch.cat(logits_list, dim=0)  # [total_T, T]
    labels_all = torch.cat(labels_list, dim=0)  # [total_T, T]
    steps_all = torch.cat(steps_list, dim=0)    # [total_T, T]
    seq_lens_all = torch.cat(seq_lens_list, dim=0)  # [total_T]
    
    # Compute loss based on loss_type
    if loss_type == 'classification':
        loss = classification_loss(logits_all, labels_all, label_smoothing)
    elif 'regression' in loss_type:
        loss = regression_loss(
            logits_all, 
            labels_all, 
            num_steps, 
            steps_all, 
            seq_lens_all,
            loss_type, 
            normalize_indices, 
            variance_lambda, 
            huber_delta
        )
    else:
        raise ValueError(
            f'Unsupported loss_type "{loss_type}". '
            'Supported: "classification", "regression_mse", "regression_mse_var", "regression_huber"'
        )
    
    return loss


# ============================================================================
# MINIMAL TEST
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("TCC Deterministic Alignment - Minimal Test")
    print("=" * 70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    B = 3          # batch size
    clip_len = 5   # timesteps
    D = 8          # embedding dimension
    
    print(f"\nTest Configuration:")
    print(f"  Batch size (B): {B}")
    print(f"  Sequence length (clip_len): {clip_len}")
    print(f"  Embedding dim (D): {D}")
    
    # Construct random embeddings
    embs = torch.randn(B, clip_len, D, requires_grad=True)
    print(f"\n✓ Created random embeddings: shape {tuple(embs.shape)}")
    
    # Construct step indices: [B, clip_len]
    # Example: steps for each sequence = [0, 1, 2, 3, 4]
    steps = torch.zeros(B, clip_len, dtype=torch.long)
    for b in range(B):
        steps[b] = torch.arange(clip_len)
    print(f"✓ Created step indices: shape {tuple(steps.shape)}")
    print(f"  steps[0] = {steps[0].tolist()}")
    
    # Construct sequence lengths: [B]
    seq_lens = torch.tensor([clip_len, clip_len, clip_len], dtype=torch.long)
    print(f"✓ Created sequence lengths: {seq_lens.tolist()}")
    
    print(f"\n{'='*70}")
    print("Test 1: Classification Loss")
    print("=" * 70)
    try:
        loss_cls = compute_deterministic_alignment_loss(
            embs=embs,
            steps=steps,
            seq_lens=seq_lens,
            similarity_type='l2',
            loss_type='classification',
            temperature=0.1,
            label_smoothing=0.1,
            normalize_indices=True,
        )
        print(f"✓ Classification loss computed: {loss_cls.item():.6f}")
        
        # Check backward
        loss_cls.backward()
        print(f"✓ Backward pass successful")
        print(f"  Gradient shape: {embs.grad.shape}")
        print(f"  Gradient norm: {embs.grad.norm().item():.6f}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("Test 2: Regression MSE Loss")
    print("=" * 70)
    try:
        embs = torch.randn(B, clip_len, D, requires_grad=True)
        loss_mse = compute_deterministic_alignment_loss(
            embs=embs,
            steps=steps,
            seq_lens=seq_lens,
            similarity_type='cosine',
            loss_type='regression_mse',
            temperature=0.1,
            normalize_indices=True,
        )
        print(f"✓ Regression MSE loss computed: {loss_mse.item():.6f}")
        
        loss_mse.backward()
        print(f"✓ Backward pass successful")
        print(f"  Gradient norm: {embs.grad.norm().item():.6f}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("Test 3: Regression MSE with Variance Loss")
    print("=" * 70)
    try:
        embs = torch.randn(B, clip_len, D, requires_grad=True)
        loss_mse_var = compute_deterministic_alignment_loss(
            embs=embs,
            steps=steps,
            seq_lens=seq_lens,
            similarity_type='l2',
            loss_type='regression_mse_var',
            temperature=0.1,
            variance_lambda=0.001,
            normalize_indices=True,
        )
        print(f"✓ Regression MSE Var loss computed: {loss_mse_var.item():.6f}")
        
        loss_mse_var.backward()
        print(f"✓ Backward pass successful")
        print(f"  Gradient norm: {embs.grad.norm().item():.6f}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("Test 4: Regression Huber Loss")
    print("=" * 70)
    try:
        embs = torch.randn(B, clip_len, D, requires_grad=True)
        loss_huber = compute_deterministic_alignment_loss(
            embs=embs,
            steps=steps,
            seq_lens=seq_lens,
            similarity_type='cosine',
            loss_type='regression_huber',
            temperature=0.1,
            huber_delta=0.1,
            normalize_indices=False,
        )
        print(f"✓ Regression Huber loss computed: {loss_huber.item():.6f}")
        
        loss_huber.backward()
        print(f"✓ Backward pass successful")
        print(f"  Gradient norm: {embs.grad.norm().item():.6f}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("All tests completed!")
    print("=" * 70)
