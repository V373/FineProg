"""
Minimal ResNet50 Backbone for TCC (Temporal Cycle-Consistency Loss).

This backbone extracts features using the Conv4c stage from ResNet50,
aligned with the TCC paper's setup when using ImageNet pretrained ResNet50.

Architecture:
- Input: [N, 3, 224, 224]
- Feature extraction: conv1 + layer1 + layer2 + layer3 (Conv4c)
- Output: [N, 1024, 14, 14]
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNet50Conv4cBackbone(nn.Module):
    """
    ResNet50 backbone that extracts Conv4c (layer3) features.
    
    This implementation strictly aligns with the TCC paper's feature extraction
    setup when using ImageNet pretrained ResNet50.
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize ResNet50 backbone.
        
        Args:
            pretrained: Whether to load ImageNet pretrained weights.
        """
        super(ResNet50Conv4cBackbone, self).__init__()
        
        # Load pretrained ResNet50 from torchvision
        if pretrained:
            # For newer torchvision versions, use weights parameter
            try:
                resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            except TypeError:
                # Fallback for older torchvision versions
                resnet50 = models.resnet50(pretrained=True)
        else:
            resnet50 = models.resnet50(pretrained=False)
        
        # Extract backbone layers: conv1, bn1, relu, maxpool, layer1, layer2, layer3
        # layer3 corresponds to Conv4c in the TCC paper
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3  # Conv4c stage - extracts [N, 1024, 14, 14]
        
        # Do NOT include layer4, avgpool, fc (following TCC paper setup)

    def set_train_base_mode(self, mode: str) -> None:
        """
        Set the training mode of the backbone, aligned with TCC's TRAIN_BASE semantics.

        Args:
            mode: One of "frozen", "only_bn", or "train_all".
                - "frozen"   : All params frozen, backbone in eval mode.
                - "train_all": All params trainable, backbone in train mode.
                - "only_bn"  : Only BatchNorm learnable params (weight/bias)
                               are trainable; backbone stays in train mode so
                               BN running stats continue to update.
        """
        _VALID_MODES = ("frozen", "only_bn", "train_all")
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unsupported train_base mode '{mode}'. "
                f"Choose from: {_VALID_MODES}"
            )

        if mode == "frozen":
            for param in self.parameters():
                param.requires_grad = False
            self.eval()

        elif mode == "train_all":
            for param in self.parameters():
                param.requires_grad = True
            self.train()

        elif mode == "only_bn":
            # First freeze everything
            for param in self.parameters():
                param.requires_grad = False
            # Then unfreeze BN learnable params (weight & bias)
            _BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
            for module in self.modules():
                if isinstance(module, _BN_TYPES):
                    if module.weight is not None:
                        module.weight.requires_grad = True
                    if module.bias is not None:
                        module.bias.requires_grad = True
            # Keep train mode so BN running stats are updated
            self.train()

    def get_trainable_parameter_count(self) -> int:
        """
        Return the number of parameters with requires_grad=True.

        Useful for debugging after calling set_train_base_mode().
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract Conv4c features.
        
        Args:
            x: Input tensor of shape [N, 3, 224, 224]
            
        Returns:
            Feature tensor of shape [N, 1024, 14, 14]
        """
        # Initial convolution block
        x = self.conv1(x)  # [N, 3, 224, 224] -> [N, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [N, 64, 112, 112] -> [N, 64, 56, 56]
        
        # Residual blocks
        x = self.layer1(x)  # [N, 64, 56, 56] -> [N, 256, 56, 56]
        x = self.layer2(x)  # [N, 256, 56, 56] -> [N, 512, 28, 28]
        x = self.layer3(x)  # [N, 512, 28, 28] -> [N, 1024, 14, 14] (Conv4c)
        
        return x


def check_pretrained_weights():
    """
    Check if pretrained ResNet50 weights are correctly loaded.
    
    Returns:
        bool: True if weights are loaded, False otherwise
    """
    print("\n[Weight Check] Verifying pretrained ResNet50 weights...")
    
    # Load model with pretrained weights
    try:
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        print("[Weight Check] ✓ Successfully loaded ResNet50 with IMAGENET1K_V1 weights")
    except TypeError:
        # Fallback for older torchvision versions
        resnet50 = models.resnet50(pretrained=True)
        print("[Weight Check] ✓ Successfully loaded ResNet50 with pretrained=True (older torchvision)")
    except Exception as e:
        print(f"[Weight Check] ✗ Failed to load pretrained weights: {e}")
        return False
    
    # Check if conv1 weights are non-zero (sanity check)
    conv1_weight = resnet50.conv1.weight.data
    has_nonzero_weights = (conv1_weight.abs().sum() > 0)
    
    if has_nonzero_weights:
        print(f"[Weight Check] ✓ Conv1 layer has non-zero weights (sum: {conv1_weight.abs().sum():.2f})")
        print(f"[Weight Check] ✓ Conv1 weight shape: {conv1_weight.shape}")
    else:
        print("[Weight Check] ✗ Conv1 layer has zero weights - weights may not be loaded")
        return False
    
    # Check layer3 (Conv4c) weights
    layer3_weight = resnet50.layer3[0].conv1.weight.data
    layer3_nonzero = (layer3_weight.abs().sum() > 0)
    
    if layer3_nonzero:
        print(f"[Weight Check] ✓ Layer3 (Conv4c) has non-zero weights (sum: {layer3_weight.abs().sum():.2f})")
    else:
        print("[Weight Check] ✗ Layer3 (Conv4c) has zero weights")
        return False
    
    print("[Weight Check] ✓ Pretrained weights verification PASSED")
    return True


def sanity_check_backbone(h5_path: str, config_path: Optional[str] = None, video_idx: int = 0):
    """
    Minimal sanity check: load frames from H5 using H5VideoDataset and verify backbone output.
    
    This test:
    1. Loads temporal context frames [clip_len, context_size, 3, 224, 224] from H5VideoDataset
    2. Extracts the current frame from context (last frame) to get [clip_len, 3, 224, 224]
    3. Runs backbone forward pass
    4. Verifies output shape is [clip_len, 1024, 14, 14]
    
    The backbone only performs per-frame visual feature extraction (Conv4c features).
    Temporal modeling is handled by the temporal embedder, not here.
    
    Args:
        h5_path: Path to the H5 file containing video data
        config_path: Path to YAML config file (optional, uses default if None)
        video_idx: Index of video to load from dataset
    """
    from pathlib import Path
    import sys
    from pathlib import Path as PathlibPath
    
    # Import H5VideoDataset from the same project
    dataset_prep_dir = PathlibPath(__file__).parent.parent / "dataset_preparation"
    sys.path.insert(0, str(dataset_prep_dir))
    
    try:
        from h5vid_dataset import H5VideoDataset
    except ImportError as e:
        print(f"[Backbone Test] Error: Cannot import H5VideoDataset: {e}")
        return
    
    # Check if H5 file exists
    if not PathlibPath(h5_path).exists():
        print(f"[Backbone Test] Error: H5 file not found at {h5_path}")
        return
    
    print(f"\n[Backbone Test] H5 file: {h5_path}")
    if config_path:
        print(f"[Backbone Test] Config file: {config_path}")
    
    # Initialize H5VideoDataset
    try:
        dataset = H5VideoDataset(
            h5_path=h5_path,
            config_path=config_path
        )
    except Exception as e:
        print(f"[Backbone Test] Error initializing dataset: {e}")
        return
    
    if len(dataset) == 0:
        print("[Backbone Test] Error: Dataset is empty")
        return
    
    # Load a sample from the dataset
    try:
        sample = dataset[video_idx]
        frames = sample["frames"]  # [clip_len, context_size, 3, 224, 224] float32 in [0, 1]
        target_steps = sample["target_steps"]
        video_id = sample["video_id"]
        action_id = sample["action_id"]
        
        print(f"[Backbone Test] Loaded video {video_id} (action_id: {action_id})")
        print(f"[Backbone Test] Frames shape: {frames.shape}")
        print(f"[Backbone Test] Target steps: {target_steps.tolist()[:5]}...")  # Show first 5
    except Exception as e:
        print(f"[Backbone Test] Error loading sample: {e}")
        return
    
    # Extract current frame from temporal context
    # frames shape: [clip_len, context_size, 3, 224, 224]
    # Extract the last frame in context (current frame): [clip_len, 3, 224, 224]
    current_frames = frames[:, -1, :, :, :]
    clip_len = current_frames.shape[0]
    
    print(f"[Backbone Test] Extracted current frames from context: {current_frames.shape}")
    print(f"[Backbone Test]   Expected input to backbone: [{clip_len}, 3, 224, 224]")
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Backbone Test] Device: {device}")
    
    # Create backbone (loads ImageNet pretrained weights)
    backbone = ResNet50Conv4cBackbone(pretrained=True)
    backbone.to(device)
    backbone.eval()
    
    # Move frames to device
    current_frames = current_frames.to(device)
    
    # Apply ImageNet normalization (required for pretrained ResNet50)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    current_frames = (current_frames - mean) / std
    
    # Forward pass through backbone
    print(f"[Backbone Test] Running forward pass...")
    with torch.no_grad():
        features = backbone(current_frames)
    
    print(f"[Backbone Test] Output shape: {features.shape}")
    print(f"[Backbone Test]   Expected output: [{clip_len}, 1024, 14, 14]")
    print(f"[Backbone Test]   Feature range: [{features.min():.3f}, {features.max():.3f}]")
    
    # Verify output shape
    expected_shape = (clip_len, 1024, 14, 14)
    if features.shape == expected_shape:
        print(f"✓ [Backbone Test] PASS: Output shape matches expected {expected_shape}")
        return features
    else:
        print(f"✗ [Backbone Test] FAIL: Output shape {features.shape} does not match expected {expected_shape}")
        return None


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. train_base mode tests
    # ------------------------------------------------------------------
    print("=" * 60)
    print("[Mode Test] Testing set_train_base_mode()")
    print("=" * 60)

    backbone = ResNet50Conv4cBackbone(pretrained=False)
    total_params = sum(p.numel() for p in backbone.parameters())

    for mode in ("frozen", "only_bn", "train_all"):
        backbone.set_train_base_mode(mode)
        trainable = backbone.get_trainable_parameter_count()
        print(f"\n  mode           : {mode}")
        print(f"  total params   : {total_params:,}")
        print(f"  trainable params: {trainable:,}")
        print(f"  backbone.training: {backbone.training}")

    print("\n" + "=" * 60)

    # ------------------------------------------------------------------
    # 2. Original backbone sanity check (H5 dataset)
    # ------------------------------------------------------------------
    h5_path = "/home/user/zhangzk/projects/fineprog/datasets/processed/pouring-2vid.h5"
    from pathlib import Path as _Path
    import sys as _sys
    _proj = str(_Path(__file__).resolve().parent.parent)
    if _proj not in _sys.path:
        _sys.path.insert(0, _proj)
    from utils.config_v2 import ConfigV2 as _ConfigV2
    config_path = str(_ConfigV2()._root / "train.yaml")  # [v2] configs_v2/train.yaml

    # This loads temporal context frames from H5VideoDataset, extracts current frames,
    # and verifies backbone produces [clip_len, 1024, 14, 14] features
    sanity_check_backbone(h5_path, config_path=config_path, video_idx=0)
