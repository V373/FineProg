"""
H5 Video Dataset for PyTorch with Temporal Context.

This dataset loads video clips from HDF5 files and constructs temporal context windows
aligned with the google-research/tcc framework.

Key features:
- Samples clip_len target time steps from full video
- Constructs causal temporal context (no future frames) for each target step
- Returns explicit [clip_len, context_size, 3, 224, 224] tensor
- All sampling parameters read from config YAML file
"""

import h5py
import torch
import numpy as np
import argparse
import yaml
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Any, Optional


class H5VideoDataset(Dataset):
    """PyTorch Dataset for video frames stored in HDF5 format with temporal context."""
    
    def __init__(
        self,
        h5_path: str,
        config_path: Optional[str] = None,
        clip_len: int = 20,
        context_size: int = 2,
        context_stride: int = 15,
        sampling_strategy: str = "offset_uniform",
        random_offset: int = 0,
        sample_all: bool = False,
        sample_all_stride: int = 1
    ):
        """
        Initialize H5 Video Dataset.
        
        All sampling parameters can be provided via config YAML file or as arguments.
        Config file takes precedence over arguments.
        
        Args:
            h5_path: Path to HDF5 file containing video data
            config_path: Path to YAML config file (overrides other arguments if provided)
            clip_len: Number of target time steps to sample from video
            context_size: Number of frames in temporal context window (causal)
            context_stride: Stride between frames in context window
            sampling_strategy: One of 'stride' or 'offset_uniform'
                - 'stride': Random offset + stride-based sampling (TCC official method)
                - 'offset_uniform': Random sampling from [random_offset, seq_len-1]
            random_offset: Minimum offset for sampling (for 'offset_uniform')
            sample_all: If True, export all frames at fixed stride (for extract_embeddings)
            sample_all_stride: Stride used when sample_all=True
        """
        self.h5_path = h5_path
        self.config_path = config_path
        
        # Load config from YAML if provided
        if config_path is not None and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.clip_len = config.get('clip_len', clip_len)
            self.context_size = config.get('context_size', context_size)
            self.context_stride = config.get('context_stride', context_stride)
            self.sampling_strategy = config.get('sampling_strategy', sampling_strategy)
            self.random_offset = config.get('random_offset', random_offset)
            self.sample_all = config.get('sample_all', sample_all)
            self.sample_all_stride = config.get('sample_all_stride', sample_all_stride)
            print(f"[H5VideoDataset] Loaded config from {config_path}")
        else:
            self.clip_len = clip_len
            self.context_size = context_size
            self.context_stride = context_stride
            self.sampling_strategy = sampling_strategy
            self.random_offset = random_offset
            self.sample_all = sample_all
            self.sample_all_stride = sample_all_stride
        
        assert self.sampling_strategy in ["stride", "offset_uniform"], \
            f"Unknown sampling strategy: {self.sampling_strategy}"
        
        # Load all video frames into memory at init time to avoid repeated H5 I/O
        self._frames_cache: Dict[str, Any] = {}
        with h5py.File(self.h5_path, 'r') as f:
            self.video_ids = sorted(list(f['videos'].keys()))
            for vid in self.video_ids:
                self._frames_cache[vid] = {
                    'frames': f['videos'][vid]['frames'][:],
                    'action_id': int(f['videos'][vid].attrs.get('action_id', -1)),
                }
        
        print(f"[H5VideoDataset] Loaded {len(self.video_ids)} videos from {h5_path} (all frames cached in memory)")
        print(f"[H5VideoDataset] Sampling config:")
        print(f"  - sample_all: {self.sample_all}")
        if self.sample_all:
            print(f"  - sample_all_stride: {self.sample_all_stride}")
        else:
            print(f"  - clip_len: {self.clip_len}")
            print(f"  - context_size: {self.context_size}")
            print(f"  - context_stride: {self.context_stride}")
            print(f"  - sampling_strategy: {self.sampling_strategy}")
            print(f"  - random_offset: {self.random_offset}")
    
    def __len__(self) -> int:
        """Return number of videos in dataset."""
        return len(self.video_ids)
    
    def _sample_offset_uniform(self, seq_len: int) -> np.ndarray:
        """
        Sample clip_len target time steps using offset_uniform strategy.
        
        Samples from [random_offset, seq_len-1] and sorts the results.
        If not enough frames available, pads with the last valid index.
        
        Args:
            seq_len: Total number of frames in video
            
        Returns:
            Array of target time step indices, sorted, shape [clip_len]
        """
        # Valid range for sampling
        min_idx = self.random_offset
        max_idx = seq_len - 1
        
        if max_idx < min_idx:
            # Not enough frames, use what we have
            available_indices = np.arange(0, seq_len)
            if len(available_indices) < self.clip_len:
                # Pad with last index
                target_steps = np.concatenate([
                    available_indices,
                    np.full(self.clip_len - len(available_indices), seq_len - 1)
                ])
            else:
                # Sample from available
                target_steps = np.random.choice(available_indices, size=self.clip_len, replace=True)
        else:
            available_range = max_idx - min_idx + 1
            
            if available_range < self.clip_len:
                # Not enough unique samples, pad with last index
                unique_samples = np.arange(min_idx, max_idx + 1)
                target_steps = np.concatenate([
                    unique_samples,
                    np.full(self.clip_len - len(unique_samples), max_idx)
                ])
            else:
                # Sample clip_len indices from [min_idx, max_idx]
                valid_indices = np.arange(min_idx, max_idx + 1)
                target_steps = np.random.choice(valid_indices, size=self.clip_len, replace=False)
        
        # Sort to maintain temporal order
        target_steps = np.sort(target_steps[:self.clip_len])
        
        return target_steps.astype(np.int32)
    
    def _sample_stride(self, seq_len: int) -> np.ndarray:
        """
        Sample clip_len target time steps using stride strategy (TCC official method).
        
        This method implements the stride-based sampling from the official TCC repository:
        - Randomly choose an offset from [0, max(1, seq_len - stride * clip_len)]
        - Sample clip_len indices starting from offset with step size = 1
        - All indices are clamped to [0, seq_len-1]
        
        Args:
            seq_len: Total number of frames in video
            
        Returns:
            Array of target time step indices, shape [clip_len]
        """
        # Calculate max offset to ensure we can sample clip_len frames without exceeding seq_len
        # The required range is: offset + clip_len - 1 < seq_len
        # => offset < seq_len - clip_len + 1
        max_offset = max(1, seq_len - self.clip_len + 1)
        
        # Randomly choose an offset
        offset = np.random.randint(0, max_offset)
        
        # Generate clip_len consecutive indices starting from offset
        target_steps = np.arange(offset, offset + self.clip_len, dtype=np.int32)
        
        # Clamp all indices to valid range [0, seq_len-1]
        target_steps = np.minimum(target_steps, seq_len - 1)
        
        return target_steps.astype(np.int32)
    
    def _construct_causal_context(self, target_step: int) -> np.ndarray:
        """
        Construct causal temporal context indices for a target time step.
        
        Creates a window of context_size indices ending at target_step,
        looking only at past and current frames (no future frames).
        
        For example:
          - context_size=2, stride=1: [target_step-1, target_step]
          - context_size=3, stride=1: [target_step-2, target_step-1, target_step]
          - context_size=5, stride=2: [target_step-8, target_step-6, target_step-4, target_step-2, target_step]
        
        All indices are clamped to [0, seq_len-1].
        
        Args:
            target_step: The current time step
            
        Returns:
            Array of context time indices, shape [context_size]
        """
        context_indices = []
        
        for i in range(self.context_size):
            # i-th position in the context window (0 = oldest, context_size-1 = current)
            # Compute offset from target_step
            offset = (self.context_size - 1 - i) * self.context_stride
            idx = target_step - offset
            
            # Clamp to valid range [0, ...]
            idx = max(0, idx)
            context_indices.append(idx)
        
        return np.array(context_indices, dtype=np.int32)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a video sample with temporal context.
        
        Returns:
            Dict with keys:
                - "frames": Tensor [clip_len, context_size, 3, 224, 224] float32 in [0, 1]
                - "target_steps": Tensor [clip_len] int32, the target time steps
                - "seq_len": int, total frames in video
                - "action_id": int, action class (or -1 if not available)
                - "video_id": str, video identifier
        """
        video_id = self.video_ids[idx]
        
        # Read from in-memory cache (loaded once at __init__)
        cached = self._frames_cache[video_id]
        frames_np = cached['frames']   # [T, 224, 224, 3] uint8
        seq_len = frames_np.shape[0]
        action_id = cached['action_id']
        
        # Step 1: Sample target time steps from full video
        if self.sample_all:
            # Export mode: stride-subsample the entire video (aligned with TCC sample_all)
            # target_steps = [0, sample_all_stride, 2*sample_all_stride, ...]
            target_steps = np.arange(0, seq_len, self.sample_all_stride, dtype=np.int32)
        elif self.sampling_strategy == "stride":
            target_steps = self._sample_stride(seq_len)
        else:  # "offset_uniform"
            target_steps = self._sample_offset_uniform(seq_len)
        
        T_out = len(target_steps)  # clip_len (train) or T_out (export all frames)

        # Step 2: Construct causal context for each target step
        # Result: [T_out, context_size]
        context_indices = np.zeros((T_out, self.context_size), dtype=np.int32)
        
        for t in range(T_out):
            target_step = target_steps[t]
            context_indices[t, :] = self._construct_causal_context(target_step)
        
        # Step 3: Gather frames based on indices
        # Result: [T_out, context_size, 224, 224, 3]
        # Clamp all indices at once, then use numpy fancy indexing (vectorized)
        idx_clamped = np.minimum(context_indices, seq_len - 1)  # [T_out, context_size]
        output_frames = frames_np[idx_clamped].astype(np.float32) / 255.0  # [T_out, context_size, 224, 224, 3]
        
        # Step 4: Convert to torch tensor and transpose to [T_out, context_size, 3, 224, 224]
        # From [T_out, context_size, 224, 224, 3] to [T_out, context_size, 3, 224, 224]
        output_tensor = torch.from_numpy(output_frames)  # [T_out, context_size, 224, 224, 3]
        output_tensor = output_tensor.permute(0, 1, 4, 2, 3)  # [T_out, context_size, 3, 224, 224]
        
        return {
            "frames": output_tensor,
            "target_steps": torch.from_numpy(target_steps),
            "seq_len": seq_len,
            "action_id": action_id,
            "video_id": video_id
        }


def collate_fn(batch: list) -> Dict[str, Any]:
    """
    Collate function for DataLoader to combine multiple samples into a batch.
    
    Supports two modes:
    - sample_all=False (train): all videos share clip_len steps, safe to stack.
      frames: [B, clip_len, context_size, 3, H, W]
    - sample_all=True (export): use batch_size=1; T_out may differ per video.
      frames: [1, T_out, context_size, 3, H, W]

    Args:
        batch: List of dicts from __getitem__, each containing:
            - "frames": [T_out, context_size, 3, 224, 224] tensor
            - "target_steps": [T_out] tensor
            - "seq_len": int
            - "action_id": int
            - "video_id": str
    
    Returns:
        Dict with batched outputs:
            - "frames": [B, T_out, context_size, 3, H, W] tensor
            - "target_steps": [B, T_out] tensor
            - "seq_len": [B] tensor
            - "action_id": [B] tensor
            - "video_id": list of strings (length B)
    """
    # Stack frames: [B, T_out, context_size, 3, 224, 224]
    frames = torch.stack([s["frames"] for s in batch])
    
    # Stack target_steps: [B, T_out]
    target_steps = torch.stack([s["target_steps"] for s in batch])
    
    # Stack seq_len: [B]
    seq_len = torch.tensor([s["seq_len"] for s in batch], dtype=torch.int32)
    
    # Stack action_id: [B]
    action_id = torch.tensor([s["action_id"] for s in batch], dtype=torch.int32)
    
    # Collect video_ids as list (no stacking needed)
    video_ids = [s["video_id"] for s in batch]
    
    return {
        "frames": frames,
        "target_steps": target_steps,
        "seq_len": seq_len,
        "action_id": action_id,
        "video_id": video_ids
    }


def build_dataloader(
    config_path: Optional[str] = None,
    h5_path: Optional[str] = None,
    h5_path_override: Optional[str] = None,
    batch_size: int = 2,
    num_workers: int = 0,
    shuffle: bool = True,
    split: str = "train",
    sample_all: bool = False,
    sample_all_stride: int = 1
) -> torch.utils.data.DataLoader:
    """
    Build PyTorch DataLoader for H5 video dataset.
    
    Configuration can be provided via YAML config file or direct arguments.
    Config file takes precedence over arguments, but h5_path_override always
    wins over everything (useful for CLI --h5_path flags).
    
    Args:
        config_path: Path to YAML config file (e.g., configs_v2/train.yaml)
        h5_path: Path to HDF5 file (used if config_path not provided or doesn't exist)
        h5_path_override: If provided, overrides h5_path from both config and h5_path arg.
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle data
        split: Data split name ("train", "val", "test") - for logging only
        sample_all: If True, export all frames at fixed stride (forces batch_size=1, shuffle=False)
        sample_all_stride: Stride used when sample_all=True
    
    Returns:
        PyTorch DataLoader instance with collate_fn configured
    
    Raises:
        ValueError: If neither config file nor h5_path is provided
    """
    # Load config from YAML if provided
    config_h5_path = h5_path
    config_batch_size = batch_size
    config_num_workers = num_workers
    config_shuffle = shuffle
    config_sample_all = sample_all
    config_sample_all_stride = sample_all_stride
    
    if config_path is not None and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Use extract_h5_path for extraction split, h5_path for all others
        if split == "extract":
            config_h5_path = config.get('extract_h5_path', config.get('h5_path', h5_path))
        else:
            config_h5_path = config.get('h5_path', h5_path)
        config_batch_size = config.get('batch_size', batch_size)
        config_num_workers = config.get('num_workers', num_workers)
        config_shuffle = config.get('shuffle', shuffle)
        config_sample_all = config.get('sample_all', sample_all)
        config_sample_all_stride = config.get('sample_all_stride', sample_all_stride)
        print(f"[build_dataloader] Loaded config from {config_path}")
    
    # h5_path_override always wins (e.g. from CLI --h5_path argument)
    if h5_path_override is not None:
        config_h5_path = h5_path_override
        print(f"[build_dataloader] h5_path overridden by caller: {config_h5_path}")
    
    if config_h5_path is None:
        raise ValueError("h5_path not found in config or arguments")
    
    # sample_all=True: force batch_size=1 and no shuffling
    if config_sample_all:
        config_batch_size = 1
        config_shuffle = False

    # Create dataset with config (dataset will read additional params from config)
    dataset = H5VideoDataset(
        h5_path=config_h5_path,
        config_path=config_path,
        sample_all=config_sample_all,
        sample_all_stride=config_sample_all_stride
    )
    
    # Create dataloader with custom collate_fn
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config_batch_size,
        num_workers=config_num_workers,
        shuffle=config_shuffle,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(config_num_workers > 0),
    )
    
    print(f"[build_dataloader] Created DataLoader ({split} split):")
    print(f"  - dataset size: {len(dataset)}")
    print(f"  - batch_size: {config_batch_size}")
    print(f"  - num_workers: {config_num_workers}")
    print(f"  - shuffle: {config_shuffle}")
    print(f"  - sample_all: {config_sample_all}")
    if config_sample_all:
        print(f"  - sample_all_stride: {config_sample_all_stride}")
    
    return dataloader


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load H5 video dataset with temporal context")
    parser.add_argument("h5_filename", type=str, help="H5 filename (e.g., pouring_processed.h5)")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to YAML config file (overrides command line args)")
    parser.add_argument("--clip_len", type=int, default=8, help="Number of target time steps")
    parser.add_argument("--context_size", type=int, default=2, help="Size of temporal context window")
    parser.add_argument("--context_stride", type=int, default=1, help="Stride in temporal context")
    parser.add_argument("--sampling_strategy", type=str, default="stride",
                        choices=["stride", "offset_uniform"],
                        help="Sampling strategy for target time steps")
    parser.add_argument("--random_offset", type=int, default=0, 
                        help="Minimum offset for offset_uniform strategy")
    
    args = parser.parse_args()
    
    # Determine config path
    config_path = args.config
    if config_path is None:
        # Try default location
        default_config = Path(__file__).parent.parent / "configs_v2" / "train.yaml"
        if default_config.exists():
            config_path = str(default_config)
    
    # Hardcode prefix directory
    h5_prefix = "/home/user/zhangzk/projects/fineprog/datasets/processed"
    h5_path = str(Path(h5_prefix) / args.h5_filename)
    
    # Create dataset
    dataset = H5VideoDataset(
        h5_path=h5_path,
        config_path=config_path,
        clip_len=args.clip_len,
        context_size=args.context_size,
        context_stride=args.context_stride,
        sampling_strategy=args.sampling_strategy,
        random_offset=args.random_offset
    )
    
    print(f"\n[Test] Dataset size: {len(dataset)}")
    
    # Test multiple samples
    for test_idx in range(min(3, len(dataset))):
        print(f"\n[Test] Sample {test_idx}:")
        sample = dataset[test_idx]
        
        frames = sample["frames"]
        target_steps = sample["target_steps"]
        seq_len = sample["seq_len"]
        action_id = sample["action_id"]
        video_id = sample["video_id"]
        
        print(f"  frames shape: {frames.shape}")
        print(f"  frames dtype: {frames.dtype}")
        print(f"  frames range: [{frames.min():.3f}, {frames.max():.3f}]")
        print(f"  target_steps: {target_steps.tolist()}")
        print(f"  target_steps shape: {target_steps.shape}")
        print(f"  seq_len: {seq_len}")
        print(f"  action_id: {action_id}")
        print(f"  video_id: {video_id}")
        
        # Verify shapes (use actual dataset parameters, not command-line args)
        # because config file values override command-line values
        expected_frames_shape = (dataset.clip_len, dataset.context_size, 3, 224, 224)
        expected_steps_shape = (dataset.clip_len,)
        
        assert frames.shape == expected_frames_shape, \
            f"Expected frames shape {expected_frames_shape}, got {frames.shape}"
        assert target_steps.shape == expected_steps_shape, \
            f"Expected target_steps shape {expected_steps_shape}, got {target_steps.shape}"
        
        print(f"  ✓ Shapes verified!")
    
    # ========== Test 1: Training mode (sample_all=False) ==========
    print(f"\n{'='*60}")
    print("[Test 1] Training mode DataLoader (sample_all=False)")
    print(f"{'='*60}")
    
    test_batch_size = min(4, len(dataset))  # Use smaller batch for testing
    dataloader_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=test_batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    batch = next(iter(dataloader_train))
    
    print(f"\n[Test 1] Batch content:")
    print(f"  batch['frames'].shape: {batch['frames'].shape}")
    print(f"  Expected: [{test_batch_size}, {dataset.clip_len}, {dataset.context_size}, 3, 224, 224]")
    print(f"  batch['target_steps'].shape: {batch['target_steps'].shape}")
    print(f"  Expected: [{test_batch_size}, {dataset.clip_len}]")
    print(f"  batch['seq_len'].shape: {batch['seq_len'].shape}")
    print(f"  batch['action_id'].shape: {batch['action_id'].shape}")
    print(f"  batch['video_id']: {batch['video_id']}")
    
    assert batch['frames'].shape == (test_batch_size, dataset.clip_len, dataset.context_size, 3, 224, 224), \
        f"Batch frames shape mismatch!"
    assert batch['target_steps'].shape == (test_batch_size, dataset.clip_len), \
        f"Batch target_steps shape mismatch!"
    assert batch['seq_len'].shape == (test_batch_size,)
    assert batch['action_id'].shape == (test_batch_size,)
    assert len(batch['video_id']) == test_batch_size
    unique_video_ids = set(batch['video_id'])
    assert len(unique_video_ids) == len(batch['video_id']), \
        f"Duplicate video_ids found in batch!"
    print(f"  ✓ All training-mode batch shape tests passed!")

    # ========== Test 2: Export mode single sample (sample_all=True) ==========
    print(f"\n{'='*60}")
    print("[Test 2] Export mode single sample (sample_all=True)")
    print(f"{'='*60}")
    
    export_dataset = H5VideoDataset(
        h5_path=h5_path,
        config_path=config_path,
        clip_len=dataset.clip_len,
        context_size=dataset.context_size,
        context_stride=dataset.context_stride,
        sampling_strategy=dataset.sampling_strategy,
        sample_all=True,
        sample_all_stride=1
    )
    
    export_sample = export_dataset[0]
    export_frames = export_sample["frames"]
    export_target_steps = export_sample["target_steps"]
    export_seq_len = export_sample["seq_len"]
    export_video_id = export_sample["video_id"]
    T_out = export_frames.shape[0]
    
    print(f"  frames.shape: {export_frames.shape}")
    print(f"  Expected: [T_out={T_out}, context_size={export_dataset.context_size}, 3, 224, 224]")
    print(f"  target_steps.shape: {export_target_steps.shape}")
    print(f"  Expected: [T_out={T_out}]")
    print(f"  seq_len: {export_seq_len}")
    print(f"  video_id: {export_video_id}")
    
    assert export_frames.shape == (T_out, export_dataset.context_size, 3, 224, 224), \
        f"Export frames shape mismatch!"
    assert export_target_steps.shape == (T_out,), \
        f"Export target_steps shape mismatch!"
    print(f"  ✓ Export mode single-sample shapes verified!")

    # ========== Test 3: Export mode DataLoader (sample_all=True, batch_size forced to 1) ==========
    print(f"\n{'='*60}")
    print("[Test 3] Export mode DataLoader (sample_all=True)")
    print(f"{'='*60}")
    
    export_dataloader = build_dataloader(
        h5_path=h5_path,
        batch_size=8,       # will be overridden to 1
        shuffle=True,       # will be overridden to False
        num_workers=0,
        split="export",
        sample_all=True,
        sample_all_stride=1
    )
    
    assert export_dataloader.batch_size == 1, \
        f"Expected batch_size=1 for sample_all=True, got {export_dataloader.batch_size}"
    
    export_batch = next(iter(export_dataloader))
    T_out_batch = export_batch['frames'].shape[1]
    
    print(f"\n[Test 3] Export batch content:")
    print(f"  batch['frames'].shape: {export_batch['frames'].shape}")
    print(f"  Expected: [1, T_out={T_out_batch}, {export_dataset.context_size}, 3, 224, 224]")
    print(f"  batch['target_steps'].shape: {export_batch['target_steps'].shape}")
    print(f"  batch['video_id']: {export_batch['video_id']}")
    
    assert export_batch['frames'].shape == (1, T_out_batch, export_dataset.context_size, 3, 224, 224), \
        f"Export batch frames shape mismatch!"
    assert export_batch['target_steps'].shape == (1, T_out_batch), \
        f"Export batch target_steps shape mismatch!"
    assert len(export_batch['video_id']) == 1
    print(f"  ✓ Export mode DataLoader batch shapes verified!")


# =============================================================================
# Backbone Feature Cache Dataset (independent branch — only_bn + cache mode)
# =============================================================================

class FeatureCacheDataset(Dataset):
    """
    Dataset that serves pre-computed backbone features instead of raw frames.

    Used exclusively when train_base=only_bn and extract_backbone_cache=True.
    Reuses the same temporal sampling logic as H5VideoDataset so the training
    distribution is identical — only the data source changes (features vs frames).

    Args:
        cache: Dict mapping video_id -> Tensor[T, 1024, 14, 14] (CPU, fp16).
               Built by extract_backbone_features() in train.py.
        h5_dataset: An existing H5VideoDataset instance whose sampling
                    parameters (clip_len, context_size, context_stride,
                    sampling_strategy, random_offset) are reused verbatim.
    """

    def __init__(self, cache: Dict[str, Any], h5_dataset: "H5VideoDataset"):
        self._cache = cache
        self.video_ids = h5_dataset.video_ids
        self._action_ids = {
            vid: h5_dataset._frames_cache[vid]["action_id"]
            for vid in self.video_ids
        }
        # Mirror sampling parameters from the source dataset
        self.clip_len = h5_dataset.clip_len
        self.context_size = h5_dataset.context_size
        self.context_stride = h5_dataset.context_stride
        self.sampling_strategy = h5_dataset.sampling_strategy
        self.random_offset = h5_dataset.random_offset
        # Bind sampling helpers directly from the source dataset instance
        self._sample_offset_uniform = h5_dataset._sample_offset_uniform
        self._sample_stride = h5_dataset._sample_stride
        self._construct_causal_context = h5_dataset._construct_causal_context

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_id = self.video_ids[idx]
        feats = self._cache[video_id]          # [T, 1024, 14, 14] fp16 CPU tensor
        seq_len = feats.shape[0]
        action_id = self._action_ids[video_id]

        # Sample clip_len target time steps (same logic as H5VideoDataset)
        if self.sampling_strategy == "stride":
            target_steps = self._sample_stride(seq_len)
        else:
            target_steps = self._sample_offset_uniform(seq_len)

        # Build causal context index array [clip_len, context_size]
        context_indices = np.zeros((self.clip_len, self.context_size), dtype=np.int32)
        for t in range(self.clip_len):
            context_indices[t, :] = self._construct_causal_context(target_steps[t])

        # Clamp and gather: [clip_len, context_size, 1024, 14, 14]
        idx_clamped = np.minimum(context_indices, seq_len - 1)
        # feats is a torch Tensor — use advanced indexing directly
        output_feats = feats[idx_clamped]      # [clip_len, context_size, 1024, 14, 14]

        return {
            "backbone_feats": output_feats,
            "target_steps": torch.from_numpy(target_steps),
            "seq_len": seq_len,
            "action_id": action_id,
            "video_id": video_id,
        }


def _collate_cache_fn(batch: list) -> Dict[str, Any]:
    """Collate function for FeatureCacheDataset batches."""
    backbone_feats = torch.stack([s["backbone_feats"] for s in batch])
    target_steps = torch.stack([s["target_steps"] for s in batch])
    seq_len = torch.tensor([s["seq_len"] for s in batch], dtype=torch.int32)
    action_id = torch.tensor([s["action_id"] for s in batch], dtype=torch.int32)
    video_ids = [s["video_id"] for s in batch]
    return {
        "backbone_feats": backbone_feats,
        "target_steps": target_steps,
        "seq_len": seq_len,
        "action_id": action_id,
        "video_id": video_ids,
    }


def build_feature_cache_dataloader(
    cache: Dict[str, Any],
    h5_dataset: "H5VideoDataset",
    batch_size: int,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader backed by pre-computed backbone features.

    Args:
        cache: Dict from extract_backbone_features() {video_id: Tensor[T,1024,14,14]}.
        h5_dataset: Source H5VideoDataset whose sampling params are reused.
        batch_size: Training batch size.
        num_workers: DataLoader worker count.

    Returns:
        DataLoader yielding batches with key "backbone_feats" instead of "frames".
    """
    dataset = FeatureCacheDataset(cache, h5_dataset)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_cache_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


# Command line examples:
# python h5vid_dataset.py pouring_processed.h5 --clip_len 8 --context_size 2
# python h5vid_dataset.py pouring_processed.h5 --config ../configs_v2/train.yaml
# python h5vid_dataset.py pouring_processed.h5 --clip_len 16 --context_size 5 --context_stride 2
# python h5vid_dataset.py pouring_processed.h5 --clip_len 16 --context_size 5 --context_stride 2 --sampling_strategy stride
# python h5vid_dataset.py pouring_processed.h5 --clip_len 16 --context_size 5 --context_stride 2 --sampling_strategy offset_uniform
# DataLoader examples:
#   from h5vid_dataset import build_dataloader
#   dataloader = build_dataloader(config_path="configs_v2/train.yaml", batch_size=32, shuffle=True)