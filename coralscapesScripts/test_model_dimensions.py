#!/usr/bin/env python
"""
Test script to verify model dimensions work correctly
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fine_tune_pipeline import CoralReefDataset, create_transforms
from coralscapesScripts.io import setup_config

def test_model_dimensions():
    """Test model input/output dimensions"""
    print("Testing model dimensions...")
    
    # Load config
    cfg = setup_config(config_path='configs/coral_bleaching_dpt_dinov2.yaml', config_base_path='configs/base.yaml')
    
    # Override with smaller image size for testing
    cfg.augmentation.train.Resize.height = 64
    cfg.augmentation.train.Resize.width = 64
    cfg.augmentation.val.Resize.height = 64
    cfg.augmentation.val.Resize.width = 64
    cfg.augmentation.test.Resize.height = 64
    cfg.augmentation.test.Resize.width = 64
    
    transforms = create_transforms(cfg)
    
    # Test dataset
    dataset = CoralReefDataset(
        root_dir="../coralscapes",
        transform=transforms["train"],
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.N_classes}")
    
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
        
        # Test model creation
        from coralscapesScripts.segmentation.model import Benchmark_Run
        
        device = torch.device("cpu")
        benchmark_run = Benchmark_Run(
            run_name="test_model",
            model_name=cfg.model.name,
            N_classes=dataset.N_classes,
            device=device,
            model_kwargs=cfg.model.kwargs,
            model_checkpoint=cfg.model.checkpoint,
            training_hyperparameters=cfg.training
        )
        
        print(f"Model created successfully!")
        print(f"Model device: {next(benchmark_run.model.parameters()).device}")
        
        # Test model forward pass
        try:
            with torch.no_grad():
                # Prepare input - batch[0] already has batch dimension
                pixel_values = batch[0]  # Shape: (batch_size, channels, height, width)
                labels = batch[1]  # Shape: (batch_size, height, width)
                
                print(f"Input pixel_values shape: {pixel_values.shape}")
                print(f"Input labels shape: {labels.shape}")
                
                # Test model forward pass
                outputs = benchmark_run.model(pixel_values=pixel_values, labels=labels)
                print(f"Model outputs type: {type(outputs)}")
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    print(f"Logits shape: {logits.shape}")
                    print(f"Labels shape: {labels.shape}")
                    print(f"Logits dtype: {logits.dtype}")
                    print(f"Labels dtype: {labels.dtype}")
                    print(f"Logits device: {logits.device}")
                    print(f"Labels device: {labels.device}")
                    print(f"Logits unique values: {torch.unique(logits.argmax(dim=1))}")
                    print(f"Labels unique values: {torch.unique(labels)}")
                    
                if hasattr(outputs, 'loss'):
                    print(f"\nLoss: {outputs.loss}")
                
                print("\n✅ Model forward pass successful!")
                return True
                
        except Exception as e:
            print(f"❌ Model forward pass failed: {e}")
            return False
        
    else:
        print("No images found in dataset!")
        return False

if __name__ == "__main__":
    success = test_model_dimensions()
    if success:
        print("✅ Model dimension test passed!")
    else:
        print("❌ Model dimension test failed!")
