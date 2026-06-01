
"""
Unified Encoder Loss Interface Layer

Provides a unified interface for loss computation:
- BaseEncoderLoss: Base class defining the unified forward interface
- build_loss(): Factory function to instantiate specific loss modules

This is an "interface layer" that does NOT implement TCC alignment details,
but only:
1. Defines the forward(embeddings, batch) -> {loss, metrics} interface
2. Reads hyperparameters from YAML configuration
3. Dispatches to concrete loss implementations (e.g., TCCLoss)

Concrete loss implementations (e.g., algos/loss/tcc/loss_tcc.py) handle
the mathematical details.
"""

from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import yaml
from pathlib import Path


class BaseEncoderLoss(nn.Module):
    """
    Base class for encoder-based loss functions.
    
    All encoder loss implementations should inherit from this class and
    implement the forward method.
    
    Interface specification:
    - forward(embeddings, batch) -> Dict[str, Any]
    - embeddings: [B, clip_len, D] - encoder output embeddings
      - B: batch size
      - clip_len: number of target frames sampled per clip
      - D: embedding dimension (typically 128)
    
    - batch: dictionary containing at least these fields:
      - target_steps: [B, clip_len] - time step indices of target frames
      - seq_len: [B] - sequence length for each batch
      - other fields depend on specific loss type
    
    - return value: dictionary containing:
      - "loss": scalar tensor (already reduced)
      - "metrics": dict - auxiliary metrics for logging
    """
    
    def __init__(self):
        """Initialize base encoder loss."""
        super(BaseEncoderLoss, self).__init__()
    
    def forward(
        self,
        embeddings: torch.Tensor,
        batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss from encoder embeddings and batch metadata.
        
        Args:
            embeddings: tensor of shape [B, clip_len, D]
                - B: batch size
                - clip_len: number of target frames per clip
                - D: embedding dimension
            
            batch: dictionary containing batch metadata, at minimum:
                - target_steps: [B, clip_len] int tensor - target frame time steps
                - seq_len: [B] int tensor - sequence length
                - other optional fields depend on specific loss type
        
        Returns:
            dictionary containing:
                - "loss": scalar tensor (usually already reduced)
                - "metrics": dict - auxiliary metrics for logging
        
        Raises:
            NotImplementedError: subclasses must implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward() method"
        )


def build_loss(
    loss_name: str,
    config_path: Optional[str] = None
) -> BaseEncoderLoss:
    """
    Factory function to build and instantiate loss modules.
    
    Reads hyperparameters from YAML configuration file and returns the
    appropriate loss module instance.
    
    Supported loss types:
    - "tcc": Temporal Cycle-Consistency Loss
    - "temporal_infonce" / "temporal_contrastive_infonce": Temporal InfoNCE Loss
    - "composite": Weighted-sum of multiple losses (e.g. TCC + TemporalInfoNCE)
    - "temporal_triplet": Intra-video temporal triplet hinge loss

    Future extensions planned:
    - "tcn": TCC with Normalization
    
    Args:
        loss_name: name of the loss (case-insensitive)
            supported: "tcc", "temporal_infonce", "temporal_contrastive_infonce", "composite", "temporal_triplet"
        
        config_path: path to YAML config file (e.g., "configs_v2/loss/loss_tcc.yaml")
            if None, uses default configuration
    
    Returns:
        instance of a BaseEncoderLoss subclass
    
    Raises:
        NotImplementedError: if loss_name is not yet implemented
        FileNotFoundError: if config_path is provided but does not exist
        ValueError: if loss_name is not recognized
    """
    loss_name = loss_name.lower().strip()
    
    # Load configuration from YAML file
    loss_cfg = {}
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                loss_cfg = yaml.safe_load(f) or {}
            print(f"[build_loss] Loaded config from {config_path}")
        else:
            print(f"[build_loss] Warning: config file not found {config_path}, using defaults")
    
    # Dispatch to concrete implementation based on loss_name
    if loss_name == "tcc":
        # Delayed import to avoid circular dependencies
        # Try relative import first, then absolute import
        try:
            from .tcc.loss_tcc import TCCLoss
        except ImportError:
            try:
                from algos.loss.tcc.loss_tcc import TCCLoss
            except ImportError as e:
                raise ImportError(
                    f"Could not import TCCLoss from either relative or absolute import paths. "
                    f"Error: {e}"
                )
        
        # Pass loss_cfg (from YAML) to TCCLoss
        # TCCLoss will merge its defaults with YAML config_path and explicit loss_cfg.
        loss_module = TCCLoss(loss_cfg=loss_cfg)
        print(f"[build_loss] Successfully created TCCLoss instance")
        return loss_module
    
    elif loss_name in ("temporal_infonce", "temporal_contrastive_infonce"):
        try:
            from .contrastive.loss_temporal_infonce import TemporalInfoNCELoss
        except ImportError:
            try:
                from algos.loss.contrastive.loss_temporal_infonce import TemporalInfoNCELoss
            except ImportError as e:
                raise ImportError(
                    f"Could not import TemporalInfoNCELoss. Error: {e}"
                )
        loss_module = TemporalInfoNCELoss(loss_cfg=loss_cfg)
        print(f"[build_loss] Successfully created TemporalInfoNCELoss instance")
        return loss_module

    elif loss_name == "composite":
        try:
            from .composite.loss_composite import CompositeEncoderLoss
        except ImportError:
            try:
                from algos.loss.composite.loss_composite import CompositeEncoderLoss
            except ImportError as e:
                raise ImportError(
                    f"Could not import CompositeEncoderLoss. Error: {e}"
                )
        # Pass config_path so CompositeEncoderLoss can resolve child config paths
        # relative to the composite YAML directory.
        loss_module = CompositeEncoderLoss(loss_cfg=loss_cfg, config_path=config_path)
        print(f"[build_loss] Successfully created CompositeEncoderLoss instance")
        return loss_module

    elif loss_name == "temporal_triplet":
        try:
            from .contrastive.loss_temporal_triplet import TemporalTripletLoss
        except ImportError:
            try:
                from algos.loss.contrastive.loss_temporal_triplet import TemporalTripletLoss
            except ImportError as e:
                raise ImportError(
                    f"Could not import TemporalTripletLoss. Error: {e}"
                )
        loss_module = TemporalTripletLoss(loss_cfg=loss_cfg)
        print(f"[build_loss] Successfully created TemporalTripletLoss instance")
        return loss_module

    elif loss_name == "tcn":
        # Reserved for future implementation
        raise NotImplementedError(
            f'Loss "{loss_name}" is not yet implemented.\n'
            "Planned: Temporal Cycle-Consistency with Normalization (TCN)"
        )

    else:
        raise ValueError(
            f'Unknown loss name: "{loss_name}"\n'
            f'Supported losses: "tcc", "temporal_infonce", "temporal_contrastive_infonce", '
            f'"composite", "temporal_triplet", "tcn" (planned)'
        )


if __name__ == "__main__":
    """
    Minimal test: verify loss interface layer functionality.
    
    Test steps:
    1. Load configuration files
    2. Build dataloader and get a batch
    3. Build encoder and generate embeddings
    4. Create loss module via build_loss
    5. Compute loss and verify output format
    """
    import sys
    
    # ============================================================================
    # Path setup
    # ============================================================================
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent  # Navigate to mytcc root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    print("=" * 80)
    print("Encoder Loss Interface Layer - Minimal Functionality Test")
    print("=" * 80)
    
    try:
        # ====================================================================
        # Step 1: Load configurations
        # ====================================================================
        from utils.config_v2 import ConfigV2 as _CfgV2
        _v2 = _CfgV2()
        train_config_path = str(_v2._root / "train.yaml")
        loss_config_path  = str(_v2._root / "loss" / "loss_tcc.yaml")

        if not Path(train_config_path).exists():
            print(f"  ERROR: {train_config_path} not found")
            sys.exit(1)

        print(f"  ✓ Found {train_config_path}")

        if not Path(loss_config_path).exists():
            print(f"  WARNING: {loss_config_path} not found, using default config")
            loss_config_path = None
        else:
            print(f"  ✓ Found {loss_config_path}")
        
        # ====================================================================
        # Step 2: Import required modules
        # ====================================================================
        print("\n[Step 2] Importing required modules...")
        
        try:
            from dataset_preparation.h5vid_dataset import build_dataloader
            print("  ✓ Imported build_dataloader")
        except ImportError as e:
            print(f"  ERROR: Could not import build_dataloader: {e}")
            sys.exit(1)
        
        try:
            from models.encoder import TCCEncoder
            print("  ✓ Imported TCCEncoder")
        except ImportError as e:
            print(f"  ERROR: Could not import TCCEncoder: {e}")
            sys.exit(1)
        
        # ====================================================================
        # Step 3: Build dataloader
        # ====================================================================
        print("\n[Step 3] Building dataloader...")
        
        h5_full_path = project_root / "datasets" / "processed" / "pouring_processed.h5"
        if not h5_full_path.exists():
            print(f"  ERROR: H5 file not found at {h5_full_path}")
            sys.exit(1)
        
        print(f"  ✓ Found H5 file")
        
        try:
            dataloader = build_dataloader(
                config_path=str(train_config_path),
                h5_path=str(h5_full_path),
                batch_size=2,
                num_workers=0,
                shuffle=False
            )
            print(f"  ✓ Created DataLoader with {len(dataloader)} batches")
        except Exception as e:
            print(f"  ERROR: Could not build dataloader: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # ====================================================================
        # Step 4: Get one batch
        # ====================================================================
        print("\n[Step 4] Getting first batch from dataloader...")
        
        batch_iter = iter(dataloader)
        batch = next(batch_iter)
        
        print(f"  Batch structure:")
        print(f"    - frames: {batch['frames'].shape} (B, clip_len, context_size, 3, H, W)")
        print(f"    - target_steps: {batch['target_steps'].shape} (B, clip_len)")
        print(f"    - seq_len: {batch['seq_len'].shape} (B,)")
        print(f"    - action_id: {batch['action_id'].shape} (B,)")
        print(f"    - video_id: {len(batch['video_id'])} videos")
        print(f"  ✓ Batch loaded successfully")
        
        # ====================================================================
        # Step 5: Compute encoder embeddings
        # ====================================================================
        print("\n[Step 5] Computing encoder embeddings...")
        
        try:
            encoder = TCCEncoder(config_path=train_config_path, pretrained=True)
            print(f"  ✓ Encoder initialized")
            
            # Move batch to device (use CPU for testing)
            device = next(encoder.parameters()).device
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in batch.items()}
            
            # Forward pass
            with torch.no_grad():
                embeddings = encoder(batch_device["frames"])
            
            print(f"  ✓ Encoder forward pass completed")
            print(f"    - embeddings shape: {embeddings.shape}")
            print(f"    - expected shape: [B={batch['frames'].shape[0]}, "
                  f"clip_len={batch['frames'].shape[1]}, D=128]")
            
            assert embeddings.shape[0] == batch['frames'].shape[0], \
                f"Batch size mismatch: {embeddings.shape[0]} vs {batch['frames'].shape[0]}"
            assert embeddings.shape[1] == batch['frames'].shape[1], \
                f"Clip length mismatch: {embeddings.shape[1]} vs {batch['frames'].shape[1]}"
            
        except Exception as e:
            print(f"  WARNING: Could not compute embeddings: {e}")
            print("  This is expected if model weights are not available.")
            print("  Creating dummy embeddings for loss interface test...")
            
            B = batch['frames'].shape[0]
            clip_len = batch['frames'].shape[1]
            D = 128  # Default embedding dimension
            embeddings = torch.randn(B, clip_len, D)
        
        # ====================================================================
        # Step 6: Build loss module
        # ====================================================================
        print("\n[Step 6] Building loss module using unified interface...")
        
        try:
            loss_module = build_loss("tcc", config_path=str(loss_config_path) 
                                     if loss_config_path else None)
            print(f"  ✓ Successfully created {loss_module.__class__.__name__}")
        except NotImplementedError as e:
            print(f"  NOTE: {e}")
            print(f"  Using dummy loss for interface test...")
            
            # class DummyLoss(BaseEncoderLoss):
            #     def forward(self, embeddings, batch):
            #         loss = embeddings.mean()
            #         return {
            #             "loss": loss,
            #             "metrics": {
            #                 "dummy_metric": loss.item(),
            #                 "batch_size": embeddings.shape[0]
            #             }
            #         }
            
            # loss_module = DummyLoss()
            # print(f"  ✓ Created DummyLoss")
        
        # ====================================================================
        # Step 7: Compute loss
        # ====================================================================
        print("\n[Step 7] Computing loss using unified interface...")
        
        # Get device
        if list(loss_module.parameters()):
            loss_device = next(loss_module.parameters()).device
        else:
            loss_device = torch.device("cpu")
        
        # Move to device
        embeddings_device = embeddings.to(loss_device)
        batch_device = {
            k: v.to(loss_device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        # Compute loss
        with torch.no_grad():
            output = loss_module(embeddings_device, batch_device)
        
        print(f"  ✓ Loss computed successfully")
        
        # Verify output format
        assert isinstance(output, dict), "Output must be a dict"
        assert "loss" in output, "Output must contain 'loss' key"
        assert "metrics" in output, "Output must contain 'metrics' key"
        assert isinstance(output["loss"], torch.Tensor), "'loss' must be a tensor"
        assert isinstance(output["metrics"], dict), "'metrics' must be a dict"
        
        print(f"\n  Output structure:")
        print(f"    - loss type:    {type(output['loss'])}")
        print(f"    - loss value:   {output['loss'].item():.6f}")
        print(f"    - metrics keys: {list(output['metrics'].keys())}")
        print(f"    - metrics:      {output['metrics']}")
        
        # ====================================================================
        # Summary
        # ====================================================================
        print("\n" + "=" * 80)
        print("✓ Minimal functionality test completed!")
        print("=" * 80)
        print("\nInterface Layer Summary:")
        print("  - BaseEncoderLoss: base class for all loss implementations")
        print("  - build_loss(): factory function to instantiate loss modules")
        print("  - forward() signature: (embeddings, batch) -> {loss, metrics}")
        print("\nUsage in training:")
        print("  loss_module = build_loss('tcc', 'configs_v2/loss/loss_tcc.yaml')")
        print("  output = loss_module(embeddings, batch)")
        print("  loss = output['loss']")
        print("  metrics = output['metrics']")
        print("=" * 80 + "\n")
    
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
