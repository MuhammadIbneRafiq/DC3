#!/usr/bin/env python
"""
Memory-optimized script to run fine-tuning for coral bleaching detection
"""

import os
import sys
import argparse
import subprocess

def main():
    """Run the optimized fine-tuning pipeline"""
    parser = argparse.ArgumentParser(description="Run memory-optimized fine-tuning")
    parser.add_argument("--config", type=str, default="configs/dpt-dinov2-giant_lora.yaml", help="Path to config file")
    parser.add_argument("--dataset-dir", type=str, default="data/cluster_0", help="Path to dataset directory")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--batch-size-eval", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cuda:0, cuda:1, cpu)")
    
    args = parser.parse_args()
    
    # Set environment variables to optimize memory usage
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:64'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Construct command with memory-saving options
    cmd = [
        "python", "fine_tune_pipeline.py",
        "--config", args.config,
        "--dataset-dir", args.dataset_dir,
        "--n-folds", str(args.n_folds),
        "--batch-size", str(args.batch_size),
        "--batch-size-eval", "1",  # Force batch size 1 for evaluation
        "--epochs", str(args.epochs),
        "--device", args.device
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
