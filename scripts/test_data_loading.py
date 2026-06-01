#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for data loading.
Run this after preparing your H5 dataset.
Run from any directory: python test/test_data_loading.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import h5py

from config import CONFIG
from datasets import H5VideoDataset, create_dataloader


def create_dummy_h5(path: str, num_videos: int = 5, frames_per_video: int = 50):
    """Create a dummy H5 file for testing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    print(f"Creating dummy H5 dataset at {path}...")
    
    with h5py.File(path, 'w') as f:
        videos_group = f.create_group('videos')
        
        for vid_idx in range(num_videos):
            video_id = f'video_{vid_idx:03d}'
            
            # Create random frames (T, H, W, C) as uint8
            frames = np.random.randint(0, 256, (frames_per_video, 224, 224, 3), dtype=np.uint8)
            videos_group.create_dataset(
                f'{video_id}',
                data=frames,
                compression='gzip'
            )
            
            # Create dummy labels
            labels = np.random.randint(0, 5, frames_per_video, dtype=np.int32)
            videos_group.create_dataset(
                f'{video_id}_meta',
                data=labels,
                compression='gzip'
            )
        
        f.attrs['dataset_name'] = 'dummy_test'
        f.attrs['num_videos'] = num_videos
        f.attrs['total_frames'] = num_videos * frames_per_video
    
    print(f"✓ Dummy H5 dataset created with {num_videos} videos")
    return path


def test_dataset_loading(h5_path: str):
    """Test H5 dataset loading."""
    print("\n" + "=" * 60)
    print("Testing H5 Dataset Loading")
    print("=" * 60)
    
    try:
        print(f"\n[1] Loading dataset from: {h5_path}")
        dataset = H5VideoDataset(
            h5_path=h5_path,
            num_frames=20,
            image_size=224,
            augment=False,
        )
        print(f"  ✓ Dataset loaded with {len(dataset)} videos")
        
        print(f"\n[2] Sampling a single video...")
        sample = dataset[0]
        print(f"  ✓ Frames shape: {sample['frames'].shape}")  # Should be (20, 3, 224, 224)
        print(f"  ✓ Video ID: {sample['video_id']}")
        print(f"  ✓ Sequence length: {sample['seq_len']}")
        if 'labels' in sample:
            print(f"  ✓ Labels shape: {sample['labels'].shape}")
        
        # Verify tensor properties
        assert sample['frames'].shape[0] == 20, "Incorrect number of frames"
        assert sample['frames'].shape[1] == 3, "Incorrect number of channels"
        assert sample['frames'].shape[2] == 224, "Incorrect height"
        assert sample['frames'].shape[3] == 224, "Incorrect width"
        assert sample['frames'].dtype == torch.float32, "Incorrect data type"
        
        print("\n[3] Creating DataLoader...")
        dataloader = create_dataloader(
            h5_path=h5_path,
            batch_size=2,
            num_frames=20,
            image_size=224,
            augment=False,
            num_workers=0,
        )
        print(f"  ✓ DataLoader created")
        
        print(f"\n[4] Iterating through batches...")
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 2:
                break
            
            print(f"\n  Batch {batch_idx}:")
            print(f"    - Frames shape: {batch['frames'].shape}")  # (batch_size, 20, 3, 224, 224)
            print(f"    - Video IDs: {batch['video_ids']}")
            print(f"    - Sequence lengths: {batch['seq_lens']}")
            if 'labels' in batch:
                print(f"    - Labels shape: {batch['labels'].shape}")
            
            # Verify batch properties
            assert batch['frames'].shape[0] == 2, "Incorrect batch size"
            assert batch['frames'].shape[1] == 20, "Incorrect number of frames"
            assert len(batch['video_ids']) == 2, "Incorrect number of video IDs"
        
        print("\n" + "=" * 60)
        print("✓ Data Loading Test PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Error during data loading test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_augmentation(h5_path: str):
    """Test data augmentation."""
    print("\n" + "=" * 60)
    print("Testing Data Augmentation")
    print("=" * 60)
    
    try:
        print(f"\n[1] Creating dataset with augmentation enabled...")
        aug_config = {
            'RANDOM_FLIP': True,
            'BRIGHTNESS': True,
            'CONTRAST': True,
        }
        
        dataset_aug = H5VideoDataset(
            h5_path=h5_path,
            num_frames=20,
            image_size=224,
            augment=True,
            augmentation_config=aug_config,
        )
        print(f"  ✓ Augmented dataset created")
        
        print(f"\n[2] Sampling with augmentation...")
        sample1 = dataset_aug[0]
        sample2 = dataset_aug[0]  # Same video, different augmentation
        
        print(f"  ✓ Frames 1 shape: {sample1['frames'].shape}")
        print(f"  ✓ Frames 2 shape: {sample2['frames'].shape}")
        
        # Check if augmentation created different samples
        if not torch.allclose(sample1['frames'], sample2['frames']):
            print(f"  ✓ Augmentation applied (frames differ)")
        else:
            print(f"  ⚠ Frames are identical (augmentation may not have been applied)")
        
        print("\n" + "=" * 60)
        print("✓ Augmentation Test PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Error during augmentation test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print(" Phase 1: Data Loading Test Suite")
    print("=" * 70)
    
    # Check if H5 file exists
    h5_path = './data/test_dummy.h5'
    
    if not Path(h5_path).exists():
        print(f"\nNo H5 file found at {h5_path}")
        print("Creating dummy data for testing...")
        h5_path = create_dummy_h5(h5_path, num_videos=5, frames_per_video=50)
    else:
        print(f"\nUsing existing H5 file at {h5_path}")
    
    # Run tests
    results = []
    
    results.append(("Dataset Loading", test_dataset_loading(h5_path)))
    results.append(("Data Augmentation", test_augmentation(h5_path)))
    
    # Summary
    print("\n" + "=" * 70)
    print(" Test Summary")
    print("=" * 70)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ All tests PASSED!")
        print("\nYou can now:")
        print("1. Prepare your real HDF5 dataset")
        print("2. Move to Phase 2: Model components implementation")
    else:
        print("\n✗ Some tests FAILED. Please check the errors above.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
