"""
Minimal TCCEncoder for TCC (Temporal Cycle-Consistency Loss).

This encoder orchestrates the complete pipeline:
1. Receive pre-grouped temporal context frames from the data layer
2. Extract per-frame Conv4c features using ResNet50 backbone
3. Aggregate features within context window using 3D convolutions
4. Output per-target-frame embeddings

Data Flow:
- Input: [B, clip_len, context_size, 3, 224, 224]
  (frames already sampled and grouped by H5VideoDataset)
- Backbone: Per-frame Conv4c feature extraction
  [B*clip_len*context_size, 3, 224, 224] -> [B*clip_len*context_size, 1024, 14, 14]
- Regroup: [B, clip_len, context_size, 1024, 14, 14]
- Temporal Embedder: Context aggregation
  [B, clip_len, context_size, 1024, 14, 14] -> [B, clip_len, D]
- Embedding normalization (output contract): optional L2 normalization
  applied to every embedding the encoder hands out.

Latent normalization is part of the encoder's public output contract and is
controlled by `embedding_normalization`:
- "none" (default): raw projection output, unchanged legacy behaviour.
- "l2": unit-norm embeddings, computed in FP32 and returned as FP32.

The temporal embedder itself always produces the raw projection; the encoder
is the single place where the normalization contract is applied, so the
regular forward path, the backbone-cache path and the dict-returning path all
share exactly one normalization step.

The encoder does NOT:
- Sample frames (done by dataset)
- Construct temporal context (done by dataset)
- Compute losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pathlib import Path
from typing import Dict, Optional, Union

# Allowed values for the encoder embedding normalization contract.
EMBEDDING_NORMALIZATION_MODES = ("none", "l2")

# Sentinel: lets us tell "caller did not specify" apart from an explicit "none".
_UNSET = object()

try:
    from .backbone import ResNet50Conv4cBackbone
    from .temporal_embedder import TCCTemporalEmbedder
except ImportError:
    # Allow running this file directly: python models/encoder.py
    from backbone import ResNet50Conv4cBackbone
    from temporal_embedder import TCCTemporalEmbedder


class TCCEncoder(nn.Module):
    """
    Temporal Cycle-Consistency Encoder for video understanding.
    
    Combines ResNet50 backbone for visual feature extraction with
    3D convolutions for temporal context aggregation.
    
    Input structure:
    - Batch of videos with pre-sampled target time steps
    - Pre-constructed causal temporal context for each target step
    - Output: Per-target-frame embeddings for TCC loss computation
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        train_config_path: Optional[str] = None,
        clip_len: int = 20,
        context_size: int = 2,
        context_stride: int = 15,
        pretrained: bool = True,
        return_backbone_feats: bool = False,
        embedding_dim: int = 128,
        embedding_normalization=_UNSET,
        debug: bool = False
    ):
        """
        Initialize TCCEncoder.
        
        Args:
            config_path: Path to data/architecture YAML config
                        (e.g., configs_v2/train.yaml or configs_v2/extract.yaml).
                        Overrides clip_len /
                        context_size / context_stride / embedding_dim if provided.
            train_config_path: Path to training YAML config
                        (e.g., configs_v2/train.yaml). Controls
                        train_base, train_embedding, pretrained, backbone_name.
                        If not provided, falls back to constructor defaults.
            clip_len: Number of target time steps per clip.
            context_size: Size of temporal context window (causal).
            context_stride: Stride between frames in context window.
            pretrained: Whether to load ImageNet pretrained ResNet50 backbone.
                        Overridden by train_config_path if that file is provided.
            return_backbone_feats: If True, also return grouped backbone features.
            embedding_dim: Positive output embedding dimension. Overridden by
                        config_path when that YAML defines embedding_dim.
            embedding_normalization: Encoder output contract for the latent
                        embeddings. Either "none" (raw projection) or "l2"
                        (unit-norm, FP32). Resolution order:
                        explicit argument > train_config_path YAML > "none".
                        An explicit value always wins, so a checkpoint loader
                        can later pass the authoritative mode recorded with the
                        weights. Legacy checkpoints carry no such metadata and
                        must therefore be interpreted as "none".
            debug: If True, print debug information during forward pass.
        """
        super(TCCEncoder, self).__init__()
        
        # ------------------------------------------------------------------ #
        # 1. Load data / architecture config YAML
        # ------------------------------------------------------------------ #
        if config_path is not None and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.clip_len = config.get('clip_len', clip_len)
            self.context_size = config.get('context_size', context_size)
            self.context_stride = config.get('context_stride', context_stride)
            embedding_dim = config.get('embedding_dim', embedding_dim)
            print(f"[TCCEncoder] Loaded data config from {config_path}")
        else:
            self.clip_len = clip_len
            self.context_size = context_size
            self.context_stride = context_stride

        self.embedding_dim = self._resolve_embedding_dim(embedding_dim)
        
        # ------------------------------------------------------------------ #
        # 2. Load training config YAML
        # ------------------------------------------------------------------ #
        # Defaults (aligned with TCC paper: only_bn + train embedding)
        _train_base = 'only_bn'
        _train_embedding = True
        _pretrained = pretrained
        _backbone_name = 'resnet50_conv4c'
        _embedding_normalization = _UNSET

        if train_config_path is not None and Path(train_config_path).exists():
            with open(train_config_path, 'r') as f:
                train_cfg = yaml.safe_load(f)
            _train_base = train_cfg.get('train_base', _train_base)
            _train_embedding = train_cfg.get('train_embedding', _train_embedding)
            _pretrained = train_cfg.get('pretrained', _pretrained)
            _backbone_name = train_cfg.get('backbone_name', _backbone_name)
            _embedding_normalization = train_cfg.get(
                'embedding_normalization', _embedding_normalization
            )
            print(f"[TCCEncoder] Loaded train config from {train_config_path}")
        else:
            if train_config_path is not None:
                print(f"[TCCEncoder] Warning: train_config_path not found: {train_config_path}")

        # Explicit constructor argument wins over the YAML value.
        if embedding_normalization is not _UNSET:
            _embedding_normalization = embedding_normalization

        # Store training config for configure_trainability()
        self._train_base = _train_base
        self._train_embedding = _train_embedding
        self.pretrained = _pretrained
        self._backbone_name = _backbone_name
        self.return_backbone_feats = return_backbone_feats
        self.debug = debug

        # Encoder output contract. Kept as plain attributes (not parameters or
        # buffers) so that state_dict() stays byte-compatible with existing
        # checkpoints and strict loading of legacy weights keeps working.
        self.embedding_normalization = self._resolve_embedding_normalization(
            _embedding_normalization
        )
        self.embedding_normalization_eps = 1e-12
        
        print(f"[TCCEncoder] Config:")
        print(f"  - clip_len: {self.clip_len}")
        print(f"  - context_size: {self.context_size}")
        print(f"  - context_stride: {self.context_stride}")
        print(f"  - embedding_dim: {self.embedding_dim}")
        print(f"  - backbone_name: {self._backbone_name}")
        print(f"  - pretrained: {self.pretrained}")
        print(f"  - train_base: {self._train_base}")
        print(f"  - train_embedding: {self._train_embedding}")
        print(f"  - embedding_normalization: {self.embedding_normalization}")
        
        # ------------------------------------------------------------------ #
        # 3. Build sub-modules
        # ------------------------------------------------------------------ #
        # Initialize backbone (ResNet50 Conv4c feature extractor)
        self.backbone = ResNet50Conv4cBackbone(pretrained=self.pretrained)
        
        # Initialize temporal embedder (3D conv context aggregator)
        self.temporal_embedder = TCCTemporalEmbedder(
            in_channels=1024,
            hidden_channels=512,
            embed_dim=self.embedding_dim,
            debug=debug
        )
        self.backbone.to(memory_format=torch.channels_last)
        self.temporal_embedder.to(memory_format=torch.channels_last_3d)
        
        print(f"[TCCEncoder] Initialized backbone and temporal embedder")

        # ------------------------------------------------------------------ #
        # 4. Apply trainability rules immediately after construction
        # ------------------------------------------------------------------ #
        self.configure_trainability()

    @staticmethod
    def _resolve_embedding_dim(value) -> int:
        """Validate the configured output embedding dimension."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                "embedding_dim must be a positive integer, "
                f"got {value!r} of type {type(value).__name__}"
            )
        return value

    @staticmethod
    def _resolve_embedding_normalization(value) -> str:
        """
        Validate and normalize the embedding_normalization setting.

        Accepts only the exact strings in EMBEDDING_NORMALIZATION_MODES.
        Anything else (bool, None, unknown string) is rejected immediately at
        construction time, so a misconfigured run fails before training starts.

        Args:
            value: Raw value from the constructor or the train config YAML,
                   or the _UNSET sentinel when nothing was specified.

        Returns:
            The resolved mode string ("none" when nothing was specified).
        """
        if value is _UNSET:
            return "none"

        # bool is a subclass of int, but it is never a valid mode here.
        if isinstance(value, bool) or not isinstance(value, str):
            raise ValueError(
                f"embedding_normalization must be one of "
                f"{list(EMBEDDING_NORMALIZATION_MODES)}, got {value!r} "
                f"of type {type(value).__name__}"
            )

        if value not in EMBEDDING_NORMALIZATION_MODES:
            raise ValueError(
                f"embedding_normalization must be one of "
                f"{list(EMBEDDING_NORMALIZATION_MODES)}, got {value!r}"
            )

        return value

    def _apply_embedding_normalization(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply the encoder output contract to raw projection outputs.

        This is the single place where latent normalization happens; every
        public embedding path routes through it exactly once.

        Args:
            embeddings: Raw temporal-embedder output, shape [B, clip_len, D].

        Returns:
            "none": the input tensor unchanged (same values, same dtype).
            "l2":   unit-norm embeddings computed and returned in FP32.
                    The cast and the normalization stay inside the autograd
                    graph, so AMP training backpropagates normally.
        """
        if self.embedding_normalization == "none":
            return embeddings

        return F.normalize(
            embeddings.float(),
            p=2,
            dim=-1,
            eps=self.embedding_normalization_eps,
        )

    def configure_trainability(self) -> None:
        """
        (Re-)apply backbone and embedder trainability rules.

        Call this method after encoder.train() or encoder.eval() to restore
        the intended trainability state, because PyTorch's Module.train()
        recursively sets every submodule to training mode, which can undo
        the per-module freezing set up at construction time.

        Uses the stored self._train_base and self._train_embedding values.
        """
        # -- Backbone --
        self.backbone.set_train_base_mode(self._train_base)

        # -- Temporal Embedder --
        if self._train_embedding:
            for param in self.temporal_embedder.parameters():
                param.requires_grad = True
            self.temporal_embedder.train()
        else:
            for param in self.temporal_embedder.parameters():
                param.requires_grad = False
            self.temporal_embedder.eval()

    def get_trainable_parameter_groups_summary(self) -> Dict[str, int]:
        """
        Return a summary of total and trainable parameter counts per sub-module.

        Returns:
            Dict with keys:
                - backbone_total
                - backbone_trainable
                - embedder_total
                - embedder_trainable
        """
        def _count(module: nn.Module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        bb_total, bb_trainable = _count(self.backbone)
        emb_total, emb_trainable = _count(self.temporal_embedder)
        return {
            "backbone_total": bb_total,
            "backbone_trainable": bb_trainable,
            "embedder_total": emb_total,
            "embedder_trainable": emb_trainable,
        }
    
    def forward(self, frames: torch.Tensor, return_backbone_feats: bool = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of TCCEncoder.
        
        Args:
            frames: Input video frames
                   Shape: [B, clip_len, context_size, 3, 224, 224]
                   - B: batch size
                   - clip_len: number of target time steps
                   - context_size: size of temporal context window (causal)
                   - 3: RGB channels
                   - 224, 224: spatial dimensions
            return_backbone_feats: Override default return_backbone_feats setting for this forward pass
                                   If True, returns both embeddings and grouped backbone features
                                   If False or None, returns only embeddings
        
        Returns:
            If return_backbone_feats is False/None (default):
                embeddings: Tensor of shape [B, clip_len, D]
                           Per-target-frame embeddings.
                           FP32 unit-norm when embedding_normalization="l2".
            
            If return_backbone_feats is True:
                Dict with keys:
                - "embeddings": [B, clip_len, D] - per-target-frame embeddings
                                (normalized per the encoder output contract)
                - "grouped_backbone_feats": [B, clip_len, context_size, 1024, 14, 14]
                                           grouped Conv4c features from backbone
                                           (raw, never normalized)
        """
        # Determine whether to return backbone features
        if return_backbone_feats is None:
            return_backbone_feats = self.return_backbone_feats
        
        # Step 0: Parse input shape and validate
        assert frames.ndim == 6, f"Expected 6D input [B, clip_len, context_size, 3, H, W], got shape {frames.shape}"
        B, clip_len, context_size, C, H, W = frames.shape
        
        assert C == 3, f"Expected 3 channels, got {C}"
        assert H == 224 and W == 224, f"Expected 224x224 spatial size, got {H}x{W}"
        assert clip_len == self.clip_len, \
            f"Input clip_len {clip_len} does not match config clip_len {self.clip_len}"
        assert context_size == self.context_size, \
            f"Input context_size {context_size} does not match config context_size {self.context_size}"
        
        if self.debug:
            print(f"[TCCEncoder.forward] Input shape: {frames.shape}")
        
        # Step 1: Reshape for backbone processing
        # [B, clip_len, context_size, 3, 224, 224]
        # -> [B*clip_len*context_size, 3, 224, 224]
        frames_flat = frames.reshape(
            B * clip_len * context_size, C, H, W
        ).contiguous(memory_format=torch.channels_last)
        
        if self.debug:
            print(f"[TCCEncoder.forward] After flatten for backbone: {frames_flat.shape}")
        
        # Step 2: Extract Conv4c features using backbone
        # [B*clip_len*context_size, 3, 224, 224]
        # -> [B*clip_len*context_size, 1024, 14, 14]
        backbone_feats_flat = self.backbone(frames_flat)
        
        if self.debug:
            print(f"[TCCEncoder.forward] After backbone: {backbone_feats_flat.shape}")
        
        # Step 3: Regroup backbone features back to grouped context format
        # [B*clip_len*context_size, 1024, 14, 14]
        # -> [B, clip_len, context_size, 1024, 14, 14]
        _, num_channels, feat_h, feat_w = backbone_feats_flat.shape
        grouped_backbone_feats = backbone_feats_flat.reshape(
            B, clip_len, context_size, num_channels, feat_h, feat_w
        )
        
        if self.debug:
            print(f"[TCCEncoder.forward] After regroup: {grouped_backbone_feats.shape}")
        
        # Step 4: Temporal embedding - aggregate context and produce per-frame embeddings
        # [B, clip_len, context_size, 1024, 14, 14]
        # -> [B, clip_len, D]
        # Routed through forward_from_feats() so that the regular path and the
        # backbone-cache path share exactly one normalization step.
        embeddings = self.forward_from_feats(grouped_backbone_feats) # here is temporal_embedder(backbone_feats) + _apply_embedding_normalization(embeddings)
        
        if self.debug:
            print(f"[TCCEncoder.forward] After temporal embedder: {embeddings.shape}")
        
        # Step 5: Return embeddings (and optionally backbone features)
        if return_backbone_feats:
            return {
                "embeddings": embeddings,
                "grouped_backbone_feats": grouped_backbone_feats
            }
        else:
            return embeddings

    def forward_from_feats(self, backbone_feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using pre-computed backbone features (cache path).

        Skips the ResNet50 backbone entirely and runs only the temporal embedder.
        Used when extract_backbone_cache=True and train_base=only_bn so that
        the expensive backbone forward/backward is replaced by a cached lookup.

        Also used internally by forward() so that both paths apply the encoder
        embedding normalization contract exactly once.

        Args:
            backbone_feats: Pre-computed Conv4c features.
                            Shape: [B, clip_len, context_size, 1024, 14, 14]

        Returns:
            embeddings: Tensor of shape [B, clip_len, D].
                        FP32 unit-norm when embedding_normalization="l2".
        """
        embeddings = self.temporal_embedder(backbone_feats)
        return self._apply_embedding_normalization(embeddings)


def sanity_check(
    h5_path: str,
    config_path: Optional[str] = None,
    video_idx: int = 0,
    batch_size: int = 2,
    pretrained: bool = False
):
    """
    Sanity check: load data from H5VideoDataset and verify encoder output.
    
    This function tests the complete TCCEncoder pipeline with real data.
    
    Args:
        h5_path: Path to the H5 file containing video data
        config_path: Path to YAML config file (optional)
        video_idx: Index of video to start loading from
        batch_size: Number of videos to load as batch
        pretrained: Whether to use pretrained backbone (set to False for faster testing)
    """
    from pathlib import Path as PathlibPath
    import sys
    
    # Import dataset
    dataset_prep_dir = PathlibPath(__file__).parent.parent / "dataset_preparation"
    sys.path.insert(0, str(dataset_prep_dir))
    
    try:
        from h5vid_dataset import H5VideoDataset
    except ImportError as e:
        print(f"[Encoder Test] Error: Cannot import H5VideoDataset: {e}")
        return
    
    # Check if H5 file exists
    if not PathlibPath(h5_path).exists():
        print(f"\n[Encoder Test] Error: H5 file not found at {h5_path}")
        return
    
    print(f"\n[Encoder Test] Loading data from: {h5_path}")
    if config_path:
        print(f"[Encoder Test] Config file: {config_path}")
    
    # Initialize H5VideoDataset
    try:
        dataset = H5VideoDataset(
            h5_path=h5_path,
            config_path=config_path
        )
    except Exception as e:
        print(f"[Encoder Test] Error initializing dataset: {e}")
        return
    
    if len(dataset) == 0:
        print("[Encoder Test] Error: Dataset is empty")
        return
    
    print(f"[Encoder Test] Dataset config:")
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
        print(f"[Encoder Test] Loaded batch frames shape: {batch_frames.shape}")
    except Exception as e:
        print(f"[Encoder Test] Error loading sample: {e}")
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Encoder Test] Device: {device}")
    
    # Move data to device
    batch_frames = batch_frames.to(device)
    
    # Apply ImageNet normalization (required for pretrained backbone)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3, 1, 1).to(device)
    batch_frames = (batch_frames - mean) / std
    
    # Initialize encoder
    print(f"[Encoder Test] Initializing TCCEncoder...")
    encoder = TCCEncoder(
        config_path=config_path,
        pretrained=pretrained,
        debug=True
    )
    encoder.to(device)
    encoder.eval()
    
    # Forward pass (default: return only embeddings)
    print(f"[Encoder Test] Running forward pass (return_backbone_feats=False)...")
    with torch.no_grad():
        embeddings = encoder(batch_frames, return_backbone_feats=False)
    
    print(f"[Encoder Test] Embeddings shape: {embeddings.shape}")
    print(
        f"[Encoder Test] Expected shape: "
        f"[{batch_size}, {dataset.clip_len}, {encoder.embedding_dim}]"
    )
    
    # Verify output shape
    expected_shape = (batch_size, dataset.clip_len, encoder.embedding_dim)
    if embeddings.shape == expected_shape:
        print(f"✓ [Encoder Test] PASS: Embeddings shape matches expected {expected_shape}")
    else:
        print(f"✗ [Encoder Test] FAIL: Embeddings shape {embeddings.shape} does not match expected {expected_shape}")
        return
    
    print(f"[Encoder Test] Embedding value range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    
    # Forward pass with backbone features
    print(f"\n[Encoder Test] Running forward pass (return_backbone_feats=True)...")
    with torch.no_grad():
        output_dict = encoder(batch_frames, return_backbone_feats=True)
    
    embeddings_2 = output_dict["embeddings"]
    backbone_feats = output_dict["grouped_backbone_feats"]
    
    print(f"[Encoder Test] Embeddings shape: {embeddings_2.shape}")
    print(f"[Encoder Test] Grouped backbone feats shape: {backbone_feats.shape}")
    print(f"[Encoder Test] Expected backbone feats shape: [{batch_size}, {dataset.clip_len}, {dataset.context_size}, 1024, 14, 14]")
    
    # Verify backbone features shape
    expected_backbone_shape = (batch_size, dataset.clip_len, dataset.context_size, 1024, 14, 14)
    if backbone_feats.shape == expected_backbone_shape:
        print(f"✓ [Encoder Test] PASS: Backbone feats shape matches expected {expected_backbone_shape}")
    else:
        print(f"✗ [Encoder Test] FAIL: Backbone feats shape {backbone_feats.shape} does not match expected {expected_backbone_shape}")
        return
    
    print(f"[Encoder Test] Backbone feats value range: [{backbone_feats.min():.4f}, {backbone_feats.max():.4f}]")
    
    print(f"\n✓ [Encoder Test] All tests PASSED!")
    return embeddings, backbone_feats


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    _proj = str(_Path(__file__).resolve().parent.parent)
    if _proj not in sys.path:
        sys.path.insert(0, _proj)
    from utils.config_v2 import ConfigV2 as _ConfigV2
    _v2_train_yaml = str(_ConfigV2()._root / "train.yaml")

    config_path       = _v2_train_yaml   # [v2] configs_v2/train.yaml
    train_config_path = _v2_train_yaml   # [v2] same file has both data and train params
    h5_path = "/home/user/zhangzk/projects/fineprog/datasets/processed/pouring_processed.h5"

    # ------------------------------------------------------------------
    # 1. Init encoder with both config files
    # ------------------------------------------------------------------
    print("=" * 60)
    print("[Main] Initializing TCCEncoder with train config...")
    print("=" * 60)

    encoder = TCCEncoder(
        config_path=config_path,
        train_config_path=train_config_path,
        pretrained=False,  # Skip downloading weights for this test
        debug=False
    )

    # ------------------------------------------------------------------
    # 2. Print trainability summary
    # ------------------------------------------------------------------
    summary = encoder.get_trainable_parameter_groups_summary()
    print("\n[Main] Trainability summary:")
    print(f"  train_base      : {encoder._train_base}")
    print(f"  train_embedding : {encoder._train_embedding}")
    print(f"  backbone_total     : {summary['backbone_total']:,}")
    print(f"  backbone_trainable : {summary['backbone_trainable']:,}")
    print(f"  embedder_total     : {summary['embedder_total']:,}")
    print(f"  embedder_trainable : {summary['embedder_trainable']:,}")

    # ------------------------------------------------------------------
    # 3. Verify configure_trainability() survives encoder.train() call
    # ------------------------------------------------------------------
    print("\n[Main] Calling encoder.train() then configure_trainability()...")
    encoder.train()
    encoder.configure_trainability()
    summary2 = encoder.get_trainable_parameter_groups_summary()
    assert summary2 == summary, "configure_trainability() must restore the same param counts"
    print("  OK - trainability restored correctly after encoder.train()")

    # ------------------------------------------------------------------
    # 4. Forward pass with a synthetic batch to verify output shape
    # ------------------------------------------------------------------
    print("\n[Main] Running synthetic forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    B = 2
    dummy = torch.randn(
        B, encoder.clip_len, encoder.context_size, 3, 224, 224,
        device=device
    )
    encoder.eval()
    encoder.configure_trainability()  # Restore after eval()
    with torch.no_grad():
        embeddings = encoder(dummy)

    expected_shape = (B, encoder.clip_len, encoder.embedding_dim)
    print(f"  Embeddings shape : {tuple(embeddings.shape)}")
    print(f"  Expected shape   : {expected_shape}")
    assert tuple(embeddings.shape) == expected_shape, \
        f"Shape mismatch: {tuple(embeddings.shape)} != {expected_shape}"
    print("  OK - output shape matches")

    print("\n" + "=" * 60)
    print("[Main] All checks PASSED")
    print("=" * 60)
