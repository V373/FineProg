"""
Temporal Cycle-Consistency (TCC) Loss - Unified Entry Point

This module implements the unified entry point for TCC loss computation.
It follows the original TCC alignment.py design pattern:
  - Reads configuration from YAML or loss_cfg dict
  - Dispatches to deterministic or stochastic alignment implementations
  - Returns loss in the unified format

This file is responsible ONLY for:
  1. Interfacing with BaseEncoderLoss
  2. Configuration handling and defaults
  3. Dispatching to alignment implementations
  
Actual mathematical implementations are delegated to:
  - deterministic_alignment.py
  - stochastic_alignment.py
"""

import sys
from typing import Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn
import yaml

# Add project root to path to enable module imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from algos.loss.encoder_loss import BaseEncoderLoss


# Default configuration values (from original TCC repository)
DEFAULT_CONFIG = {
    "stochastic_matching": False,
    "normalize_embeddings": False,
    "loss_type": "classification",
    "similarity_type": "l2",
    "num_cycles": 20,
    "num_cycles_fraction": 0.25,
    "cycle_length": 2,
    "softmax_temperature": 0.1,
    "label_smoothing": 0.1,
    "variance_lambda": 0.001,
    "huber_delta": 0.1,
    "normalize_indices": True,
}


class TCCLoss(BaseEncoderLoss):
    """
    Temporal Cycle-Consistency Loss - Unified Entry Point.
    
    This class wraps the TCC alignment loss computation and provides a unified
    interface compatible with the encoder loss framework.
    
    Configuration precedence (highest to lowest):
    1. Explicitly passed loss_cfg dict
    2. YAML config file (for example configs_v2/loss/loss_tcc.yaml)
    3. Default hardcoded values from DEFAULT_CONFIG
    
    Configuration keys:
    - stochastic_matching (bool): Use stochastic or deterministic matching
    - normalize_embeddings (bool): L2 normalize embeddings before loss computation
    - loss_type (str): 'classification', 'regression_mse', 'regression_mse_var', 'regression_huber'
    - similarity_type (str): 'l2' or 'cosine'
    - num_cycles_fraction (float): Fraction of cycles for stochastic matching
    - cycle_length (int): Length of cycles (e.g., 2 = A->B->A)
    - softmax_temperature (float): Temperature for softmax scaling
    - label_smoothing (float): Label smoothing for classification loss
    - variance_lambda (float): Variance weight for regression_mse_var
    - huber_delta (float): Delta for huber regression loss
    - normalize_indices (bool): Normalize indices by sequence lengths
    """
    
    def __init__(
        self,
        loss_cfg: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize TCCLoss with configuration.
        
        Args:
            loss_cfg: Configuration dictionary from encoder_loss.py
                     Overrides values from config_path and defaults
            
            config_path: Path to YAML config file
                        If None, uses only DEFAULT_CONFIG plus explicit loss_cfg.
        """
        super(TCCLoss, self).__init__()
        
        # Start with default configuration
        config = DEFAULT_CONFIG.copy()
        
        # Merge with YAML config if available
        yaml_config = self._load_yaml_config(config_path)
        if yaml_config:
            config.update(yaml_config)
        
        # Override with explicitly passed loss_cfg
        if loss_cfg:
            config.update(loss_cfg)
        
        # Extract configuration
        self.stochastic_matching = config.get("stochastic_matching")
        self.normalize_embeddings = config.get("normalize_embeddings")
        self.loss_type = config.get("loss_type")
        self.similarity_type = config.get("similarity_type")
        self.num_cycles = config.get("num_cycles", 20)
        self.num_cycles_fraction = config.get("num_cycles_fraction")
        self.cycle_length = config.get("cycle_length")
        self.softmax_temperature = config.get("softmax_temperature")
        self.label_smoothing = config.get("label_smoothing")
        self.variance_lambda = config.get("variance_lambda")
        self.huber_delta = config.get("huber_delta")
        self.normalize_indices = config.get("normalize_indices")
        
        self._last_num_cycles = self.num_cycles or 0

        print(f"[TCCLoss] Initialized with merged config:")
        print(f"  - stochastic_matching: {self.stochastic_matching}")
        print(f"  - normalize_embeddings: {self.normalize_embeddings}")
        print(f"  - loss_type: {self.loss_type}")
        print(f"  - similarity_type: {self.similarity_type}")
        print(f"  - num_cycles: {self.num_cycles}")
        print(f"  - num_cycles_fraction: {self.num_cycles_fraction}")
        print(f"  - cycle_length: {self.cycle_length}")
        print(f"  - softmax_temperature: {self.softmax_temperature}")
        print(f"  - normalize_indices: {self.normalize_indices}")
    
    @staticmethod
    def _load_yaml_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load YAML configuration file.
        
        Args:
            config_path: Path to YAML config file. If None, tries default location.
        
        Returns:
            Configuration dictionary (empty dict if file not found)
        """
        # Use provided path or default location
        if config_path is None:
            from pathlib import Path as _Path
            _v2_path = _Path(__file__).resolve().parent.parent.parent.parent / "configs_v2" / "loss" / "loss_tcc.yaml"
            if _v2_path.exists():
                config_path = str(_v2_path)
        
        config_file = Path(config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    result = yaml.safe_load(f) or {}
                    print(f"[TCCLoss] Loaded config from {config_path}")
                    return result
            except Exception as e:
                print(f"[TCCLoss] Warning: Failed to load {config_path}: {e}")
        else:
            print(f"[TCCLoss] Warning: Config file not found: {config_path}")
        
        print(f"[TCCLoss] Using hardcoded defaults")
        return {}
    
    def forward(
        self,
        embeddings: torch.Tensor,
        batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute TCC loss from embeddings and batch metadata.
        
        Args:
            embeddings: Tensor of shape [B, clip_len, D]
                - B: batch size
                - clip_len: number of target frames per clip
                - D: embedding dimension
            
            batch: Dictionary containing batch metadata:
                - target_steps: [B, clip_len] int tensor - target frame indices
                - seq_len: [B] int tensor - sequence length for each sample
        
        Returns:
            Dictionary containing:
                - "loss": scalar tensor (the computed TCC loss)
                - "metrics": dict with auxiliary metrics
                    - "loss_total": same as loss (scalar)
                    - "loss_tcc": same as loss (scalar)
        
        Raises:
            ValueError: If required batch fields are missing
            NotImplementedError: If alignment functions not yet implemented
        """
        # Validate inputs
        if "target_steps" not in batch:
            raise ValueError("batch must contain 'target_steps'")
        if "seq_len" not in batch:
            raise ValueError("batch must contain 'seq_len'")
        
        target_steps = batch["target_steps"]
        seq_len = batch["seq_len"]
        
        # Get batch info
        batch_size = embeddings.shape[0]
        clip_len = embeddings.shape[1]
        
        # Dispatch to appropriate alignment implementation
        if self.stochastic_matching:
            loss = self._compute_stochastic_loss(
                embeddings, target_steps, seq_len, batch_size, clip_len
            )
        else:
            loss = self._compute_deterministic_loss(
                embeddings, target_steps, seq_len, batch_size, clip_len
            )
        
        # Return loss in unified format
        metrics = {
            "loss_total": loss.detach(),
            "loss_tcc": loss.detach(),
            "tcc/stochastic_matching": 1.0 if self.stochastic_matching else 0.0,
        }
        if self.stochastic_matching:
            metrics["tcc/num_cycles"] = float(self._last_num_cycles)
            metrics["tcc/cycle_length"] = float(self.cycle_length)
        return {"loss": loss, "metrics": metrics}
    
    def _compute_deterministic_loss(
        self,
        embeddings: torch.Tensor,
        target_steps: torch.Tensor,
        seq_len: torch.Tensor,
        batch_size: int,
        clip_len: int
    ) -> torch.Tensor:
        """
        Compute deterministic alignment loss (align all pairs).
        
        This aligns each pair of embeddings in the batch.
        
        Args:
            embeddings: [B, clip_len, D] batch of embeddings
            target_steps: [B, clip_len] indices of target frames
            seq_len: [B] sequence lengths
            batch_size: Size of batch
            clip_len: Number of timesteps per sequence
        
        Returns:
            Scalar loss tensor
        """
        try:
            from .deterministic_alignment import compute_deterministic_alignment_loss
        except ImportError:
            # If relative import fails, try absolute import
            try:
                from algos.loss.tcc.deterministic_alignment import compute_deterministic_alignment_loss
            except ImportError:
                raise NotImplementedError(
                    "deterministic_alignment.compute_deterministic_alignment_loss "
                    "is not yet implemented. Please implement the alignment module."
                )
        
        loss = compute_deterministic_alignment_loss(
            embs=embeddings,
            steps=target_steps,
            seq_lens=seq_len,
            similarity_type=self.similarity_type,
            loss_type=self.loss_type,
            temperature=self.softmax_temperature,
            label_smoothing=self.label_smoothing,
            variance_lambda=self.variance_lambda,
            huber_delta=self.huber_delta,
            normalize_indices=self.normalize_indices
        )
        
        return loss
    
    def _compute_stochastic_loss(
        self,
        embeddings: torch.Tensor,
        target_steps: torch.Tensor,
        seq_len: torch.Tensor,
        batch_size: int,
        clip_len: int
    ) -> torch.Tensor:
        """
        Compute stochastic alignment loss (sample cycles).
        
        This samples random cycles from the batch for alignment.
        
        Args:
            embeddings: [B, clip_len, D] batch of embeddings
            target_steps: [B, clip_len] indices of target frames
            seq_len: [B] sequence lengths
            batch_size: Size of batch
            clip_len: Number of timesteps per sequence
        
        Returns:
            Scalar loss tensor
        """
        try:
            from .stochastic_alignment import compute_stochastic_alignment_loss
        except ImportError:
            # If relative import fails, try absolute import
            try:
                from algos.loss.tcc.stochastic_alignment import compute_stochastic_alignment_loss
            except ImportError:
                raise NotImplementedError(
                    "stochastic_alignment.compute_stochastic_alignment_loss "
                    "is not yet implemented. Please implement the alignment module."
                )
        
        # Determine num_cycles: prefer explicit num_cycles, fall back to fraction
        if self.num_cycles is not None:
            num_cycles = int(self.num_cycles)
        else:
            num_cycles = int(batch_size * clip_len * self.num_cycles_fraction)
        num_cycles = max(num_cycles, 1)
        self._last_num_cycles = num_cycles

        loss = compute_stochastic_alignment_loss(
            embeddings=embeddings,
            target_steps=target_steps,
            seq_len=seq_len,
            batch_size=batch_size,
            clip_len=clip_len,
            num_cycles=num_cycles,
            loss_type=self.loss_type,
            similarity_type=self.similarity_type,
            cycle_length=self.cycle_length,
            temperature=self.softmax_temperature,
            label_smoothing=self.label_smoothing,
            variance_lambda=self.variance_lambda,
            huber_delta=self.huber_delta,
            normalize_indices=self.normalize_indices,
        )

        return loss


if __name__ == "__main__":
    """
    Minimal test for TCCLoss integration.
    
    This test:
    1. Creates synthetic batch data
    2. Creates embeddings tensor
    3. Instantiates TCCLoss with default config loading
    4. Runs forward pass
    5. Prints results
    """
    import sys
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    print("=" * 80)
    print("Minimal TCCLoss Test")
    print("=" * 80)
    
    # Create synthetic batch data
    batch_size = 2
    clip_len = 4
    embedding_dim = 128
    
    print(f"\n[Test] Creating synthetic data:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Clip length: {clip_len}")
    print(f"  - Embedding dim: {embedding_dim}")
    
    # Create embeddings
    embeddings = torch.randn(batch_size, clip_len, embedding_dim)
    
    # Create batch dictionary
    batch = {
        "target_steps": torch.arange(clip_len).unsqueeze(0).expand(batch_size, -1),
        "seq_len": torch.full((batch_size,), clip_len, dtype=torch.long)
    }
    
    print(f"\n[Test] Created tensors:")
    print(f"  - embeddings shape: {embeddings.shape}")
    print(f"  - target_steps shape: {batch['target_steps'].shape}")
    print(f"  - seq_len shape: {batch['seq_len'].shape}")
    
    # Instantiate loss - will load config from default path
    print(f"\n[Test] Instantiating TCCLoss (will auto-load config from YAML)...")
    loss_fn = TCCLoss()
    
    # Try forward pass (will fail with NotImplementedError since alignment modules aren't implemented)
    print(f"\n[Test] Running forward pass...")
    try:
        output = loss_fn(embeddings, batch)
        print(f"\n[Test] Forward pass successful!")
        print(f"  - loss: {output['loss']}")
        print(f"  - metrics: {output['metrics']}")
    except NotImplementedError as e:
        print(f"\n[Test] Expected NotImplementedError (alignment modules not yet implemented):")
        print(f"  {e}")
    except Exception as e:
        print(f"\n[Test] Unexpected error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test completed")
    print("=" * 80)
