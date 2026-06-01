# -*- coding: utf-8 -*-
"""
Utility functions for TCC PyTorch project.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR
import numpy as np


def setup_logging(logdir: str, name: str = 'tcc'):
    """
    Setup logging configuration.
    
    Args:
        logdir: Directory to save logs
        name: Logger name
    
    Returns:
        Logger instance
    """
    os.makedirs(logdir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(os.path.join(logdir, 'log.txt'))
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def get_device(device_name: Optional[str] = None) -> torch.device:
    """
    Get PyTorch device.
    
    Args:
        device_name: 'cuda', 'cpu', or None (auto-detect)
    
    Returns:
        torch.device instance
    """
    if device_name is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    
    return device


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(
    model: nn.Module,
    optimizer_type: str = 'Adam',
    learning_rate: float = 0.0001,
    weight_decay: float = 1e-4,
    **kwargs
):
    """
    Create optimizer.
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer (Adam, SGD, AdamW)
        learning_rate: Initial learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
    
    Returns:
        Optimizer instance
    """
    if optimizer_type == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    decay_type: str = 'fixed',
    total_steps: int = 150000,
    **kwargs
):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        decay_type: Type of decay (fixed, exp_decay, manual, cosine)
        total_steps: Total training steps
        **kwargs: Additional scheduler arguments
    
    Returns:
        Scheduler instance or None
    """
    if decay_type == 'fixed':
        return None
    
    elif decay_type == 'exp_decay':
        decay_rate = kwargs.get('exp_decay_rate', 0.97)
        decay_steps = kwargs.get('exp_decay_steps', 1000)
        
        def lr_lambda(step):
            return decay_rate ** (step / decay_steps)
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif decay_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=kwargs.get('eta_min', 0)
        )
    
    elif decay_type == 'manual':
        boundaries = kwargs.get('manual_lr_step_boundaries', [5000, 10000])
        decay_rate = kwargs.get('manual_lr_decay_rate', 0.1)
        
        def lr_lambda(step):
            for boundary in boundaries:
                if step < boundary:
                    return 1.0
                decay_rate *= decay_rate
            return decay_rate
        
        return LambdaLR(optimizer, lr_lambda)
    
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    global_step: int = 0,
    checkpoint_path: str = 'checkpoint.pth',
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        global_step: Current global step
        checkpoint_path: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'global_step': global_step,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    checkpoint_path: str = 'checkpoint.pth',
    device: str = 'cuda',
) -> int:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load into
        optimizer: Optimizer to restore state
        scheduler: Scheduler to restore state
        checkpoint_path: Path to checkpoint
        device: Device to load to
    
    Returns:
        Global step from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    global_step = checkpoint.get('global_step', 0)
    return global_step


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_module(module: nn.Module):
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module):
    """Unfreeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = True


def freeze_bn_module(module: nn.Module):
    """Freeze batch normalization parameters (keep only BN stats fixed)."""
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            child.eval()
        else:
            freeze_bn_module(child)


def compute_l2_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 distance between two tensors.
    
    Args:
        x1: Tensor of shape (N, D)
        x2: Tensor of shape (M, D)
    
    Returns:
        Distance matrix of shape (N, M)
    """
    # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2 * x1 @ x2.T
    x1_sqnorm = torch.sum(x1 ** 2, dim=1, keepdim=True)  # (N, 1)
    x2_sqnorm = torch.sum(x2 ** 2, dim=1, keepdim=True)  # (M, 1)
    x1_x2 = torch.mm(x1, x2.t())  # (N, M)
    
    distances = x1_sqnorm + x2_sqnorm.t() - 2 * x1_x2
    distances = torch.clamp(distances, min=0.0)  # Numerical stability
    distances = torch.sqrt(distances)
    
    return distances


def compute_cosine_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine distance between two tensors.
    
    Args:
        x1: Tensor of shape (N, D)
        x2: Tensor of shape (M, D)
    
    Returns:
        Distance matrix of shape (N, M)
    """
    # Normalize
    x1_norm = torch.nn.functional.normalize(x1, p=2, dim=1)
    x2_norm = torch.nn.functional.normalize(x2, p=2, dim=1)
    
    # Cosine similarity: x1_norm @ x2_norm.T
    similarities = torch.mm(x1_norm, x2_norm.t())
    
    # Cosine distance: 1 - similarity
    distances = 1 - similarities
    
    return distances
