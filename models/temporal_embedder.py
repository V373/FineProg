"""
Minimal Temporal Embedder for TCC (Temporal Cycle-Consistency Loss).

This module implements the embedding network from Table 1 of the TCC paper.

KEY CHANGE: This embedder NO LONGER performs temporal stacking internally.
Instead, it directly consumes context features that are pre-constructed by the data layer
(H5VideoDataset). The temporal context window is already provided in the input.

Architecture:
- Input: [B, clip_len, context_size, 1024, 14, 14]
        - Temporal context features pre-grouped by H5VideoDataset
        - Each (b, t) location has a window of context_size frames
- Processing:
  - Reshape to [B*clip_len, 1024, context_size, 14, 14] for 3D convolution
  - Two 3D Conv layers (1024 → 512 → 512)
  - Global 3D max pooling over [context_size, 14, 14]
  - Two FC layers (512 → 512 → 512)
  - Linear projection to 128-dim embeddings
- Output: [B, clip_len, 128]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TCCTemporalEmbedder(nn.Module):
    """
    Temporal Embedder Network for TCC.
    
    Expects pre-constructed temporal context features from the data layer.
    The temporal context window is already provided as a separate dimension.
    
    This embedder simply:
    1. Takes the grouped context features
    2. Applies 3D convolutions to aggregate within the context window
    3. Produces per-target-frame embeddings
    """
    
    def __init__(
        self,
        in_channels: int = 1024,
        hidden_channels: int = 512,
        embed_dim: int = 128,
        debug: bool = False
    ):
        """
        Initialize TCCTemporalEmbedder.
        
        Args:
            in_channels: Number of input channels (1024 for ResNet50 Conv4c)
            hidden_channels: Number of channels in 3D conv layers (512)
            embed_dim: Output embedding dimension (128)
            debug: If True, print shape information during forward pass
        """
        super(TCCTemporalEmbedder, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.embed_dim = embed_dim
        self.debug = debug
        
        # Two 3D convolutional layers
        # Input: [B*clip_len, 1024, context_size, 14, 14]
        # Conv3d 1: 1024 -> 512, kernel=3, padding=1 (preserve spatial/temporal size)
        self.conv3d_1 = nn.Conv3d(
            in_channels, hidden_channels,
            kernel_size=3, padding=1, bias=True
        )
        self.relu_1 = nn.ReLU(inplace=True)
        
        # Conv3d 2: 512 -> 512, kernel=3, padding=1
        self.conv3d_2 = nn.Conv3d(
            hidden_channels, hidden_channels,
            kernel_size=3, padding=1, bias=True
        )
        self.relu_2 = nn.ReLU(inplace=True)
        
        # Global 3D max pooling will be applied in forward()
        # Pools over [context_size, 14, 14] -> [1]
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.relu_fc1 = nn.ReLU(inplace=True)
        
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.relu_fc2 = nn.ReLU(inplace=True)
        
        # Final linear projection to embedding dimension
        self.proj = nn.Linear(hidden_channels, embed_dim)
    
    def forward(self, cnn_feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of temporal embedder.
        
        Args:
            cnn_feats: Pre-constructed temporal context features
                      Shape: [B, clip_len, context_size, 1024, 14, 14]
                      - B: batch size
                      - clip_len: number of target time steps
                      - context_size: size of temporal context window
                      - 1024: channels from ResNet50 Conv4c
                      - 14, 14: spatial dimensions
        
        Returns:
            Embeddings of shape [B, clip_len, 128]
        """
        B, clip_len, context_size, C, H, W = cnn_feats.shape
        
        if self.debug:
            print(f"[TCCTemporalEmbedder] Input shape: {cnn_feats.shape}")
        
        # Step 1: Permute + reshape for 3D convolution processing.
        #
        # Memory layout of cnn_feats (C-order):
        #   [B, clip_len, context_size, C, H, W]
        #   inner block per (b,t): [context_size, C, H, W]
        #
        # Conv3d expects: [N, C_in, D, H, W]  (D = temporal/context depth)
        # So we need:     [B*clip_len, C, context_size, H, W]
        #   inner block per (b,t): [C, context_size, H, W]  ← must permute first!
        #
        # Step 1a: permute(0,1,3,2,4,5)
        #   [B, clip_len, context_size, C, H, W]
        #   -> [B, clip_len, C, context_size, H, W]
        # Step 1b: reshape
        #   [B, clip_len, C, context_size, H, W]
        #   -> [B*clip_len, C, context_size, H, W]
        cnn_feats_reshaped = cnn_feats.permute(0, 1, 3, 2, 4, 5).reshape(
            B * clip_len, C, context_size, H, W
        ).contiguous(memory_format=torch.channels_last_3d)
        
        if self.debug:
            print(f"[TCCTemporalEmbedder] After reshape for Conv3D: {cnn_feats_reshaped.shape}")
        
        # Step 2: First 3D convolution + ReLU
        # [B*clip_len, 1024, context_size, 14, 14]
        # -> [B*clip_len, 512, context_size, 14, 14]
        x = self.conv3d_1(cnn_feats_reshaped)
        x = self.relu_1(x)
        
        if self.debug:
            print(f"[TCCTemporalEmbedder] After Conv3D_1 + ReLU: {x.shape}")
        
        # Step 3: Second 3D convolution + ReLU
        # [B*clip_len, 512, context_size, 14, 14]
        # -> [B*clip_len, 512, context_size, 14, 14]
        x = self.conv3d_2(x)
        x = self.relu_2(x)
        
        if self.debug:
            print(f"[TCCTemporalEmbedder] After Conv3D_2 + ReLU: {x.shape}")
        
        # Step 4: Global 3D max pooling
        # Pool over temporal (context_size) and spatial (14, 14) dimensions
        # [B*clip_len, 512, context_size, 14, 14]
        # -> [B*clip_len, 512, 1, 1, 1]
        # -> [B*clip_len, 512]
        x = F.adaptive_max_pool3d(x, output_size=1)
        x = x.reshape(B * clip_len, self.hidden_channels)
        
        if self.debug:
            print(f"[TCCTemporalEmbedder] After global 3D max pool: {x.shape}")
        
        # Step 5: First fully connected layer + ReLU
        # [B*clip_len, 512] -> [B*clip_len, 512]
        x = self.fc1(x)
        x = self.relu_fc1(x)
        
        if self.debug:
            print(f"[TCCTemporalEmbedder] After FC1 + ReLU: {x.shape}")
        
        # Step 6: Second fully connected layer + ReLU
        # [B*clip_len, 512] -> [B*clip_len, 512]
        x = self.fc2(x)
        x = self.relu_fc2(x)
        
        if self.debug:
            print(f"[TCCTemporalEmbedder] After FC2 + ReLU: {x.shape}")
        
        # Step 7: Final linear projection to embedding dimension
        # [B*clip_len, 512] -> [B*clip_len, 128]
        embeddings = self.proj(x)
        
        if self.debug:
            print(f"[TCCTemporalEmbedder] After projection: {embeddings.shape}")
        
        # Step 8: Reshape back to [B, clip_len, 128]
        embeddings = embeddings.reshape(B, clip_len, self.embed_dim)
        
        if self.debug:
            print(f"[TCCTemporalEmbedder] Final output shape: {embeddings.shape}")
        
        return embeddings


def sanity_check(
    h5_path: str,
    config_path: Optional[str] = None,
    video_idx: int = 0,
    batch_size: int = 2
):
    """
    Sanity check: load data from H5VideoDataset and verify embedder output shape.
    
    This function tests the temporal embedder with real data from the H5 file,
    using the new input format where context is pre-constructed by the dataset.
    
    Args:
        h5_path: Path to the H5 file containing video data
        config_path: Path to YAML config file (optional)
        video_idx: Index of video to start loading from
        batch_size: Number of videos to load as batch
    """
    from pathlib import Path
    import sys
    from pathlib import Path as PathlibPath
    
    # Import backbone and dataset
    models_dir = PathlibPath(__file__).parent
    sys.path.insert(0, str(models_dir))
    
    from backbone import ResNet50Conv4cBackbone
    
    dataset_prep_dir = PathlibPath(__file__).parent.parent / "dataset_preparation"
    sys.path.insert(0, str(dataset_prep_dir))
    
    try:
        from h5vid_dataset import H5VideoDataset
    except ImportError as e:
        print(f"[Embedder Test] Error: Cannot import H5VideoDataset: {e}")
        return
    
    # Check if file exists
    if not PathlibPath(h5_path).exists():
        print(f"\n[Embedder Test] Error: H5 file not found at {h5_path}")
        return
    
    print(f"\n[Embedder Test] Loading data from: {h5_path}")
    if config_path:
        print(f"[Embedder Test] Config file: {config_path}")
    
    # Initialize H5VideoDataset
    try:
        dataset = H5VideoDataset(
            h5_path=h5_path,
            config_path=config_path
        )
    except Exception as e:
        print(f"[Embedder Test] Error initializing dataset: {e}")
        return
    
    if len(dataset) == 0:
        print("[Embedder Test] Error: Dataset is empty")
        return
    
    print(f"[Embedder Test] Dataset config:")
    print(f"  - clip_len: {dataset.clip_len}")
    print(f"  - context_size: {dataset.context_size}")
    print(f"  - context_stride: {dataset.context_stride}")
    
    # Load batch of samples
    try:
        batch_frames = []
        for idx in range(min(batch_size, len(dataset))):
            sample = dataset[video_idx + idx]
            frames = sample["frames"]  # [clip_len, context_size, 3, 224, 224]
            batch_frames.append(frames)
        
        # Stack into batch: [batch_size, clip_len, context_size, 3, 224, 224]
        batch_frames = torch.stack(batch_frames, dim=0)
        print(f"[Embedder Test] Loaded batch frames shape: {batch_frames.shape}")
    except Exception as e:
        print(f"[Embedder Test] Error loading sample: {e}")
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Embedder Test] Device: {device}")
    
    # Initialize backbone
    backbone = ResNet50Conv4cBackbone(pretrained=True)
    backbone.to(device)
    backbone.eval()
    
    # Process frames through backbone
    print(f"[Embedder Test] Extracting backbone features...")
    batch_size_actual, clip_len, context_size, _, _, _ = batch_frames.shape
    
    # Reshape batch frames to process all frames together
    # [batch_size, clip_len, context_size, 3, 224, 224]
    # -> [batch_size * clip_len * context_size, 3, 224, 224]
    batch_frames_flat = batch_frames.reshape(
        batch_size_actual * clip_len * context_size, 3, 224, 224
    )
    batch_frames_flat = batch_frames_flat.to(device)
    
    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    batch_frames_flat = (batch_frames_flat - mean) / std
    
    # Extract backbone features
    with torch.no_grad():
        backbone_features_flat = backbone(batch_frames_flat)
    # [batch_size * clip_len * context_size, 1024, 14, 14]
    
    # Reshape back to grouped format
    # [batch_size * clip_len * context_size, 1024, 14, 14]
    # -> [batch_size, clip_len, context_size, 1024, 14, 14]
    cnn_feats = backbone_features_flat.reshape(
        batch_size_actual, clip_len, context_size, 1024, 14, 14
    )
    print(f"[Embedder Test] CNN features shape: {cnn_feats.shape}")
    
    # Initialize temporal embedder
    embedder = TCCTemporalEmbedder(
        in_channels=1024,
        hidden_channels=512,
        embed_dim=128,
        debug=True
    )
    embedder.to(device)
    embedder.eval()
    
    print(f"[Embedder Test] Running temporal embedder...")
    
    # Forward pass through embedder
    with torch.no_grad():
        embeddings = embedder(cnn_feats)
    
    print(f"[Embedder Test] Embeddings shape: {embeddings.shape}")
    print(f"[Embedder Test] Expected shape: [{batch_size_actual}, {clip_len}, 128]")
    
    # Verify output shape
    expected_shape = (batch_size_actual, clip_len, 128)
    if embeddings.shape == expected_shape:
        print(f"✓ [Embedder Test] PASS: Output shape matches expected {expected_shape}")
    else:
        print(f"✗ [Embedder Test] FAIL: Output shape {embeddings.shape} does not match expected {expected_shape}")
    
    print(f"[Embedder Test] Embedding value range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    
    return embeddings


if __name__ == "__main__":
    # Path to the processed pouring dataset
    h5_path = "/home/user/zhangzk/projects/fineprog/datasets/processed/pouring_processed.h5"
    from pathlib import Path as _Path
    import sys as _sys
    _proj = str(_Path(__file__).resolve().parent.parent)
    if _proj not in _sys.path:
        _sys.path.insert(0, _proj)
    from utils.config_v2 import ConfigV2 as _ConfigV2
    config_path = str(_ConfigV2()._root / "train.yaml")  # [v2] configs_v2/train.yaml
    
    # Run sanity check with batch_size=2
    # This loads temporal context features from H5VideoDataset
    # passes them through backbone + embedder
    # and verifies output shape
    sanity_check(h5_path, config_path=config_path, batch_size=2)
