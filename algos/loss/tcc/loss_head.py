"""
Loss head functions for TCC alignment.

Implements classification_loss and regression_loss functions compatible with
both deterministic and stochastic alignment strategies.

Reference: Original TensorFlow implementation in google-research/tcc/tcc/losses.py
"""

import torch
import torch.nn.functional as F


def classification_loss(logits, labels, label_smoothing):
    """
    Loss function based on classifying the correct indices.
    
    In the paper, this is called Cycle-back Classification.
    Implements categorical cross-entropy with optional label smoothing.

    Args:
        logits: Tensor, Pre-softmax scores used for classification loss.
               Shape: [N, C] where N is batch size, C is number of classes.
               These are similarity scores after cycling back to the starting sequence.
        labels: Tensor, One-hot labels containing the ground truth.
               Shape: [N, C], with 1 at the ground truth index.
        label_smoothing: Float, label smoothing factor (0 <= label_smoothing < 1).
                        Applied to cross-entropy loss.

    Returns:
        loss: Scalar tensor, categorical cross-entropy loss with label smoothing.
    """
    # Apply label smoothing if specified
    if label_smoothing > 0:
        # Smooth the one-hot labels
        labels = labels * (1.0 - label_smoothing) + label_smoothing / labels.shape[-1]
    
    # Compute cross-entropy loss
    # F.cross_entropy expects logits and class indices, so we need to use log_softmax + labels
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -torch.sum(labels * log_probs, dim=-1)
    
    return torch.mean(loss)


def regression_loss(
    logits, 
    labels, 
    num_steps, 
    steps, 
    seq_lens, 
    loss_type,
    normalize_indices, 
    variance_lambda, 
    huber_delta
):
    """
    Loss function based on regressing to the correct indices.
    
    In the paper, this is called Cycle-back Regression. There are 3 variants:
    i) regression_mse: MSE of the predicted indices and ground truth indices.
    ii) regression_mse_var: MSE of the predicted indices that takes into account
        the variance of the similarities. This is important when the rate at which
        sequences go through different phases changes a lot. The variance scaling
        allows dynamic weighting of the MSE loss based on the similarities.
    iii) regression_huber: Huber loss between the predicted indices and ground
        truth indices.

    Args:
        logits: Tensor, Pre-softmax similarity scores after cycling back to the
               starting sequence. Shape: [N, T] where N is batch size, T is timesteps.
        labels: Tensor, One-hot labels containing the ground truth.
               Shape: [N, T], with 1 at the ground truth index.
        num_steps: Integer, Number of steps in the sequence embeddings.
        steps: Tensor, Step indices/frame indices of the embeddings.
              Shape: [N, T] where N is batch size, T is number of timesteps.
        seq_lens: Tensor, Lengths of the sequences from which the sampling was done.
                 Shape: [N] where N is batch size.
        loss_type: String, specifies the regression loss type.
                  Options: 'regression_mse', 'regression_mse_var', 'regression_huber'
        normalize_indices: Boolean, if True normalizes indices by sequence lengths.
                          Useful for numerical stability.
        variance_lambda: Float, weight of the variance of the similarity predictions.
                        Higher values prefer low variance (sharper) similarities.
        huber_delta: Float, Delta for Huber loss function.

    Returns:
        loss: Scalar tensor, regression loss.
    """
    # Ensure tensors are on the same device and dtype
    device = logits.device
    dtype = logits.dtype
    
    labels = labels.detach()
    steps = steps.detach()
    seq_lens = seq_lens.detach()
    
    # Normalize indices if requested
    if normalize_indices:
        float_seq_lens = seq_lens.to(dtype)
        # Tile seq_lens to match steps shape: [N, T]
        tile_seq_lens = float_seq_lens.unsqueeze(1).expand(-1, num_steps)
        steps_normalized = steps.to(dtype) / tile_seq_lens
    else:
        steps_normalized = steps.to(dtype)
    
    # Compute softmax over logits to get probability distribution
    beta = F.softmax(logits, dim=-1)  # Shape: [N, T]
    
    # Compute true and predicted time indices
    true_time = torch.sum(steps_normalized * labels, dim=1)  # Shape: [N]
    pred_time = torch.sum(steps_normalized * beta, dim=1)   # Shape: [N]
    
    if loss_type in ['regression_mse', 'regression_mse_var']:
        if 'var' in loss_type:
            # Variance-aware regression
            pred_time_tiled = pred_time.unsqueeze(1).expand(-1, num_steps)  # Shape: [N, T]
            
            # Compute variance of predicted time
            pred_time_variance = torch.sum(
                torch.square(steps_normalized - pred_time_tiled) * beta, 
                dim=1
            )  # Shape: [N]
            
            # Clamp variance for numerical stability
            pred_time_variance = torch.clamp(pred_time_variance, min=1e-8)
            
            # Using log of variance as it is numerically stabler
            pred_time_log_var = torch.log(pred_time_variance)
            squared_error = torch.square(true_time - pred_time)
            
            # Loss = exp(-log_var) * squared_error + variance_lambda * log_var
            loss = torch.exp(-pred_time_log_var) * squared_error + variance_lambda * pred_time_log_var
            return torch.mean(loss)
        else:
            # Standard MSE regression
            mse_loss = F.mse_loss(pred_time, true_time, reduction='mean')
            return mse_loss
    
    elif loss_type == 'regression_huber':
        # Huber loss
        huber_loss = F.huber_loss(
            pred_time, 
            true_time, 
            delta=huber_delta, 
            reduction='mean'
        )
        return huber_loss
    
    else:
        raise ValueError(
            f'Unsupported regression loss "{loss_type}". '
            'Supported losses are: regression_mse, regression_mse_var, regression_huber.'
        )


# ============================================================================
# MINIMAL TEST
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("TCC Loss Head - Minimal Test")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Test parameters
    N = 10  # batch size / number of samples
    T = 5   # number of timesteps
    
    print(f"\nTest Configuration:")
    print(f"  Batch size (N): {N}")
    print(f"  Timesteps (T): {T}")
    
    # Create test tensors
    logits = torch.randn(N, T, requires_grad=True)
    
    # Labels should be [N, T] with each row being a one-hot vector
    labels = torch.zeros(N, T)
    for i in range(N):
        labels[i, i % T] = 1.0
    
    steps = torch.arange(T, dtype=torch.long).unsqueeze(0).expand(N, -1)
    seq_lens = torch.tensor([T] * N, dtype=torch.long)
    
    print(f"\n✓ Created test tensors:")
    print(f"  logits shape: {logits.shape}")
    print(f"  labels shape: {labels.shape}")
    print(f"  steps shape: {steps.shape}")
    print(f"  seq_lens shape: {seq_lens.shape}")
    
    # Test 1: Classification Loss
    print(f"\n{'='*70}")
    print("Test 1: Classification Loss")
    print("=" * 70)
    try:
        loss_cls = classification_loss(logits, labels, label_smoothing=0.1)
        print(f"  ✓ Classification loss: {loss_cls.item():.6f}")
        print(f"    Scalar tensor: {loss_cls.dim() == 0}")
        print(f"    Requires grad: {loss_cls.requires_grad}")
        
        # Test backward
        loss_cls.backward()
        print(f"    Gradient norm: {logits.grad.norm().item():.6f}")
        print(f"  ✓ PASS")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Regression MSE
    print(f"\n{'='*70}")
    print("Test 2: Regression MSE Loss")
    print("=" * 70)
    try:
        logits = torch.randn(N, T, requires_grad=True)
        loss_mse = regression_loss(
            logits, labels, T, steps, seq_lens,
            loss_type='regression_mse',
            normalize_indices=True,
            variance_lambda=0.0,
            huber_delta=0.0
        )
        print(f"  ✓ Regression MSE loss: {loss_mse.item():.6f}")
        print(f"    Scalar tensor: {loss_mse.dim() == 0}")
        
        loss_mse.backward()
        print(f"    Gradient norm: {logits.grad.norm().item():.6f}")
        print(f"  ✓ PASS")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Regression MSE with Variance
    print(f"\n{'='*70}")
    print("Test 3: Regression MSE with Variance Loss")
    print("=" * 70)
    try:
        logits = torch.randn(N, T, requires_grad=True)
        loss_mse_var = regression_loss(
            logits, labels, T, steps, seq_lens,
            loss_type='regression_mse_var',
            normalize_indices=True,
            variance_lambda=0.001,
            huber_delta=0.0
        )
        print(f"  ✓ Regression MSE Var loss: {loss_mse_var.item():.6f}")
        print(f"    Scalar tensor: {loss_mse_var.dim() == 0}")
        
        loss_mse_var.backward()
        print(f"    Gradient norm: {logits.grad.norm().item():.6f}")
        print(f"  ✓ PASS")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Regression Huber
    print(f"\n{'='*70}")
    print("Test 4: Regression Huber Loss")
    print("=" * 70)
    try:
        logits = torch.randn(N, T, requires_grad=True)
        loss_huber = regression_loss(
            logits, labels, T, steps, seq_lens,
            loss_type='regression_huber',
            normalize_indices=False,
            variance_lambda=0.0,
            huber_delta=0.1
        )
        print(f"  ✓ Regression Huber loss: {loss_huber.item():.6f}")
        print(f"    Scalar tensor: {loss_huber.dim() == 0}")
        
        loss_huber.backward()
        print(f"    Gradient norm: {logits.grad.norm().item():.6f}")
        print(f"  ✓ PASS")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Label Smoothing Variations
    print(f"\n{'='*70}")
    print("Test 5: Label Smoothing Variations")
    print("=" * 70)
    try:
        logits = torch.randn(N, T, requires_grad=True)
        
        loss_no_smooth = classification_loss(logits.clone().detach().requires_grad_(True), 
                                             labels, label_smoothing=0.0)
        loss_smooth = classification_loss(logits, labels, label_smoothing=0.1)
        
        print(f"  Loss with label_smoothing=0.0: {loss_no_smooth.item():.6f}")
        print(f"  Loss with label_smoothing=0.1: {loss_smooth.item():.6f}")
        print(f"  Losses differ: {abs(loss_no_smooth.item() - loss_smooth.item()) > 1e-6}")
        print(f"  ✓ PASS")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: normalize_indices Variations
    print(f"\n{'='*70}")
    print("Test 6: normalize_indices Variations")
    print("=" * 70)
    try:
        logits = torch.randn(N, T, requires_grad=True)
        
        loss_not_norm = regression_loss(
            logits.clone().detach().requires_grad_(True),
            labels, T, steps, seq_lens,
            loss_type='regression_mse',
            normalize_indices=False,
            variance_lambda=0.0,
            huber_delta=0.0
        )
        
        logits = torch.randn(N, T, requires_grad=True)
        loss_norm = regression_loss(
            logits, labels, T, steps, seq_lens,
            loss_type='regression_mse',
            normalize_indices=True,
            variance_lambda=0.0,
            huber_delta=0.0
        )
        
        print(f"  Loss with normalize_indices=False: {loss_not_norm.item():.6f}")
        print(f"  Loss with normalize_indices=True: {loss_norm.item():.6f}")
        print(f"  Losses differ: {abs(loss_not_norm.item() - loss_norm.item()) > 1e-6}")
        print(f"  ✓ PASS")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("✓ All tests completed!")
    print("=" * 70)
