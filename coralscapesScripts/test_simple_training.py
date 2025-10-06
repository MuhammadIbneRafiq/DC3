#!/usr/bin/env python
"""
Simple test script to verify the training pipeline works
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fine_tune_pipeline import CoralReefDataset, create_transforms

def test_dataset():
    """Test the dataset loading"""
    print("Testing dataset loading...")
    
    # Create simple config using the actual config loading
    from coralscapesScripts.io import setup_config
    
    # Load the actual config
    cfg = setup_config(config_path='configs/coral_bleaching_dpt_dinov2.yaml', config_base_path='configs/base.yaml')
    
    # Override with smaller image size for testing
    cfg.augmentation.train.Resize.height = 128
    cfg.augmentation.train.Resize.width = 128
    cfg.augmentation.val.Resize.height = 128
    cfg.augmentation.val.Resize.width = 128
    cfg.augmentation.test.Resize.height = 128
    cfg.augmentation.test.Resize.width = 128
    
    transforms = create_transforms(cfg)
    
    # Test dataset
    dataset = CoralReefDataset(
        root_dir="../coralscapes",
        transform=transforms["train"],
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a single sample
    if len(dataset) > 0:
        image, mask = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Image dtype: {image.dtype}")
        print(f"Mask dtype: {mask.dtype}")
        print(f"Mask unique values: {torch.unique(mask)}")
        
        # Test DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch = next(iter(dataloader))
        print(f"Batch images shape: {batch[0].shape}")
        print(f"Batch masks shape: {batch[1].shape}")
        
        return True
    else:
        print("No images found in dataset!")
        return False

if __name__ == "__main__":
    success = test_dataset()
    if success:
        print("✅ Dataset test passed!")
    else:
        print("❌ Dataset test failed!")
