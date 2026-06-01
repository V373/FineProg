#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for Phase 2 model components.

Tests:
1. BaseModel - ResNet50 feature extraction
2. ConvEmbedder - Embedding network
3. Algorithm base class
Run from any directory: python test/test_models.py
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from models import BaseModel, ConvEmbedder, ConvGRUEmbedder, Classifier, get_model
from algos.algorithm import Algorithm


def test_base_model():
    """Test BaseModel for feature extraction."""
    print("\n" + "=" * 60)
    print("Test 1: BaseModel (ResNet50 Feature Extraction)")
    print("=" * 60)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_steps = CONFIG.TRAIN.NUM_FRAMES
        
        # Create model
        model = BaseModel(num_steps=num_steps, device=device)
        model = model.to(device)
        model.eval()
        
        print(f"✓ BaseModel created on {device}")
        
        # Create dummy input
        batch_size = 2
        h, w, c = 224, 224, 3
        dummy_frames = torch.randn(batch_size, num_steps, h, w, c, device=device)
        
        print(f"✓ Input shape: {dummy_frames.shape}")
        
        # Forward pass
        with torch.no_grad():
            features = model(dummy_frames, training=False)
        
        print(f"✓ Output shape: {features.shape}")
        print(f"✓ Output is tensor: {isinstance(features, torch.Tensor)}")
        print(f"✓ Output dtype: {features.dtype}")
        
        # Verify dimensions
        assert features.shape[0] == batch_size, f"Batch size mismatch: {features.shape[0]} vs {batch_size}"
        assert features.shape[1] == num_steps, f"Num frames mismatch: {features.shape[1]} vs {num_steps}"
        assert features.shape[-1] == 3 or features.shape[-1] == 2048, "Unexpected feature dimension"
        
        print("\n✓ BaseModel test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n✗ BaseModel test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conv_embedder():
    """Test ConvEmbedder for embedding generation."""
    print("\n" + "=" * 60)
    print("Test 2: ConvEmbedder (Embedding Network)")
    print("=" * 60)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_frames = CONFIG.TRAIN.NUM_FRAMES
        
        # Create embedder with ResNet50 output channels
        embedder = ConvEmbedder(input_channels=2048)
        embedder = embedder.to(device)
        embedder.eval()
        
        print(f"✓ ConvEmbedder created on {device} (input_channels=2048)")
        
        # Create dummy features (from BaseModel output)
        batch_size = 2
        num_context = CONFIG.DATA.NUM_STEPS
        h, w, c = 7, 7, 2048  # Typical ResNet50 output: (7, 7, 2048)
        
        # Total frames = num_frames * num_context
        total_frames = num_frames * num_context
        dummy_features = torch.randn(batch_size, total_frames, h, w, c, device=device)
        
        print(f"✓ Input shape: {dummy_features.shape}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Total frames: {total_frames} (num_frames={num_frames} × num_context={num_context})")
        
        # Forward pass
        with torch.no_grad():
            embeddings = embedder(dummy_features, num_frames)
        
        print(f"✓ Output shape: {embeddings.shape}")
        print(f"✓ Output is tensor: {isinstance(embeddings, torch.Tensor)}")
        print(f"✓ Output dtype: {embeddings.dtype}")
        
        # Verify dimensions
        embedding_dim = CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE
        assert embeddings.shape[0] == batch_size * num_frames, \
            f"Batch size mismatch: {embeddings.shape[0]} vs {batch_size * num_frames}"
        assert embeddings.shape[-1] == embedding_dim, \
            f"Embedding dim mismatch: {embeddings.shape[-1]} vs {embedding_dim}"
        
        print(f"✓ Embedding dimension: {embedding_dim}")
        
        print("\n✓ ConvEmbedder test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n✗ ConvEmbedder test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_dict():
    """Test get_model() factory function."""
    print("\n" + "=" * 60)
    print("Test 3: Model Factory (get_model)")
    print("=" * 60)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create models
        models = get_model(device=device)
        
        print(f"✓ Model dictionary created on {device}")
        print(f"✓ Keys: {list(models.keys())}")
        
        # Check structure
        assert 'cnn' in models, "Missing 'cnn' key"
        assert 'emb' in models, "Missing 'emb' key"
        
        cnn = models['cnn']
        emb = models['emb']
        
        print(f"✓ CNN model type: {type(cnn).__name__}")
        print(f"✓ Embedder model type: {type(emb).__name__}")
        
        # Test end-to-end
        batch_size = 2
        num_frames = CONFIG.TRAIN.NUM_FRAMES
        h, w, c = 224, 224, 3
        
        dummy_frames = torch.randn(batch_size, num_frames, h, w, c, device=device)
        
        cnn.eval()
        emb.eval()
        
        with torch.no_grad():
            features = cnn(dummy_frames, training=False)
            embeddings = emb(features, num_frames)
        
        print(f"\n✓ End-to-end test:")
        print(f"  - Input: {dummy_frames.shape}")
        print(f"  - Features: {features.shape}")
        print(f"  - Embeddings: {embeddings.shape}")
        
        print("\n✓ Model factory test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n✗ Model factory test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_algorithm_base_class():
    """Test Algorithm base class interface."""
    print("\n" + "=" * 60)
    print("Test 4: Algorithm Base Class")
    print("=" * 60)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a concrete implementation for testing
        class DummyAlgorithm(Algorithm):
            def compute_loss(self, embs, steps, seq_lens, global_step, 
                           training=True, frame_labels=None, seq_labels=None):
                return torch.tensor(0.0, device=device)
        
        algo = DummyAlgorithm(device=device)
        algo = algo.to(device)
        algo.eval()
        
        print(f"✓ Algorithm instance created on {device}")
        
        # Test forward pass
        batch_size = 2
        num_frames = CONFIG.TRAIN.NUM_FRAMES
        h, w, c = 224, 224, 3
        
        data = {'frames': torch.randn(batch_size, num_frames, h, w, c, device=device)}
        steps = torch.arange(num_frames, device=device).unsqueeze(0).expand(batch_size, -1)
        seq_lens = torch.full((batch_size,), num_frames, device=device)
        
        print(f"✓ Input prepared:")
        print(f"  - Frames: {data['frames'].shape}")
        print(f"  - Steps: {steps.shape}")
        print(f"  - Seq lens: {seq_lens.shape}")
        
        # Forward pass
        with torch.no_grad():
            embeddings = algo(data, steps, seq_lens, training=False)
        
        print(f"✓ Forward pass output: {embeddings.shape}")
        
        embedding_dim = CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE
        assert embeddings.shape == (batch_size, num_frames, embedding_dim), \
            f"Shape mismatch: {embeddings.shape} vs ({batch_size}, {num_frames}, {embedding_dim})"
        
        # Test parameter access
        params = algo.get_base_and_embedding_variables()
        print(f"✓ Trainable parameters: {len(params)}")
        
        all_params = algo.get_all_trainable_variables()
        print(f"✓ All parameters: {len(all_params)}")
        
        # Test model control
        algo.set_training_mode(True)
        print(f"✓ Training mode set")
        
        print("\n✓ Algorithm base class test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n✗ Algorithm base class test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_modes():
    """Test different model training modes."""
    print("\n" + "=" * 60)
    print("Test 5: Model Training Modes (frozen/only_bn/train_all)")
    print("=" * 60)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test each mode
        for mode in ['frozen', 'only_bn', 'train_all']:
            CONFIG.MODEL.TRAIN_BASE = mode
            
            class TestAlgo(Algorithm):
                def compute_loss(self, embs, steps, seq_lens, global_step, 
                               training=True, frame_labels=None, seq_labels=None):
                    return torch.tensor(0.0, device=device)
            
            algo = TestAlgo(device=device)
            algo = algo.to(device)
            
            params = algo.get_base_and_embedding_variables()
            
            if mode == 'frozen':
                assert len(params) > 0, "Should have embedding params at least"
                print(f"✓ Mode '{mode}': {len(params)} trainable params (embedding only)")
            elif mode == 'only_bn':
                print(f"✓ Mode '{mode}': {len(params)} trainable params (BN only + embedding)")
            elif mode == 'train_all':
                assert len(params) > 100, "Should have many trainable params"
                print(f"✓ Mode '{mode}': {len(params)} trainable params (all)")
        
        print("\n✓ Model modes test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n✗ Model modes test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print(" Phase 2: Model Components Test Suite")
    print("=" * 70)
    
    results = []
    
    results.append(("BaseModel Feature Extraction", test_base_model()))
    results.append(("ConvEmbedder Embedding", test_conv_embedder()))
    results.append(("Model Factory", test_model_dict()))
    results.append(("Algorithm Base Class", test_algorithm_base_class()))
    results.append(("Model Training Modes", test_model_modes()))
    
    # Summary
    print("\n" + "=" * 70)
    print(" Test Summary")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ All Phase 2 tests PASSED!")
        print("\nNext steps:")
        print("1. Implement loss functions in Phase 3")
        print("2. Implement specific algorithms (Alignment, TCN)")
        print("3. Build training loop in Phase 5")
        return 0
    else:
        print("\n✗ Some tests FAILED. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
