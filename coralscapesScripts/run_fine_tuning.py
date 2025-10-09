#!/usr/bin/env python
"""
Run script for coral bleaching fine-tuning
"""
# TODO:
# First make sure your data is stored in DC3/coralscapesScripts/data
# Then run DC3/color_normalization/color_normalization/clustering_for_normalization.py
# You might have to install `kneed`, but it sill show up
# the clusters will be saved in data folder
# in fine_tune_pipeline.py lines 884, 796 and 258, change the folder name to the cluster nr. you're running
# Now, in here, specify the cluster to run
# Happy running!
import os
import argparse
from datetime import datetime
import torch

# Set CUDA memory allocation configuration before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'


def main():
    parser = argparse.ArgumentParser(description="Run coral bleaching fine-tuning")
    # parser.add_argument("--config", type=str, default="configs/coral_bleaching_dpt_dinov2.yaml",
    #                     help="Path to config file")
    parser.add_argument("--config", type=str, default="configs/segformer-mit-b2_lora_dc3.yaml",
                        help="Path to config file")
    parser.add_argument("--dataset-dir", type=str, default="data/cluster_2",  #TODO: specify cluster
                        help="Path to dataset directory")
    parser.add_argument("--n-folds", type=int, default=5, 
                        help="Number of cross-validation folds")
    parser.add_argument("--batch-size", type=int, default=2, 
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--device", type=str, default="gpu1", 
                        choices=["cpu", "gpu1"], 
                        help="Device to use for training (cpu or gpu1)")
    
    args = parser.parse_args()
    
    # Set up device and print device information
    if args.device == "gpu1":
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            device = "cuda:1"  # Use GPU 1
            device_name = torch.cuda.get_device_name(1)
            print(f"Using GPU 1: {device_name}")
        elif torch.cuda.is_available():
            device = "cuda:0"  # Fallback to GPU 0 if only one GPU available
            device_name = torch.cuda.get_device_name(0)
            print(f"Only one GPU available, using GPU 0: {device_name}")
        else:
            device = "cpu"
            device_name = "CPU"
            print("CUDA not available, falling back to CPU")
    else:  # cpu
        device = "cpu"
        device_name = "CPU"
        print("Using CPU")
    
    # Create a unique run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"coral_bleaching_{timestamp}"
    
    # Build the command
    cmd = [
        "python", "fine_tune_pipeline.py",
        f"--config={args.config}",
        f"--dataset-dir={args.dataset_dir}",
        f"--n-folds={args.n_folds}",
        f"--run-name={run_name}",
        f"--batch-size={args.batch_size}",
        f"--epochs={args.epochs}",
        f"--device={device}"
    ]
    
    # Print and execute the command
    print("Running command:")
    print(" ".join(cmd))
    os.system(" ".join(cmd))

if __name__ == "__main__":
    main()
