#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test to verify Phase 1 setup is correct.
Run from any directory: python test/test_setup.py
or: python -m pytest test/test_setup.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all Phase 1 components can be imported."""
    print("=" * 60)
    print("Testing Phase 1 Setup")
    print("=" * 60)
    
    try:
        print("\n[1/5] Testing config import...")
        from config import CONFIG
        print("  ✓ CONFIG imported successfully")
        print(f"  ✓ LOGDIR: {CONFIG.LOGDIR}")
        print(f"  ✓ TRAIN.BATCH_SIZE: {CONFIG.TRAIN.BATCH_SIZE}")
        
        print("\n[2/5] Testing datasets import...")
        from datasets import H5VideoDataset, create_dataloader
        print("  ✓ H5VideoDataset imported successfully")
        print("  ✓ create_dataloader imported successfully")
        
        print("\n[3/5] Testing utils import...")
        from utils import (
            setup_logging, get_device, set_seed, get_optimizer,
            save_checkpoint, load_checkpoint, count_parameters
        )
        print("  ✓ Utils functions imported successfully")
        
        print("\n[4/5] Testing device setup...")
        device = get_device()
        print(f"  ✓ Device: {device}")
        
        print("\n[5/5] Testing package structure...")
        from pathlib import Path
        
        required_dirs = [
            'algos',
            'tcc',
            'evaluation',
            'dataset_preparation',
            'configs',
        ]
        
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists() and dir_path.is_dir():
                print(f"  ✓ {dir_name}/ directory exists")
            else:
                print(f"  ✗ {dir_name}/ directory NOT found")
                return False
        
        print("\n" + "=" * 60)
        print("✓ Phase 1 Setup Test PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Prepare your HDF5 data using:")
        print("   python dataset_preparation/tfrecords_to_h5.py --help")
        print("\n2. Test data loading with:")
        print("   python test_data_loading.py")
        print("\n3. Move to Phase 2: Model components implementation")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import Error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected Error: {e}")
        return False


if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)
