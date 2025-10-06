#!/usr/bin/env python
"""
Coral Bleaching Detection Fine-Tuning Pipeline

This script creates a complete pipeline for fine-tuning the DPT-DINOv2-Giant model
on the coral reef dataset with cross-validation to detect coral bleaching.
"""

import os
import sys
import torch
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import glob
from sklearn.model_selection import KFold
import argparse
import yaml
import copy
import time
from datetime import datetime

# Add parent directory to path to import coralscapesScripts modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coralscapesScripts.datasets.preprocess import get_preprocessor
from coralscapesScripts.datasets.utils import calculate_weights
from coralscapesScripts.segmentation.model import Benchmark_Run
from coralscapesScripts.segmentation.benchmarking import launch_benchmark
from coralscapesScripts.logger import Logger, save_benchmark_run
from coralscapesScripts.io import setup_config, get_parser, update_config_with_args


class CoralReefDataset(Dataset):
    """
    Custom Dataset for Coral Reef images with bleaching masks
    """
    
    def __init__(self, root_dir, images=None, transform=None, transform_target=True, split='train'):
        """
        Args:
            root_dir (str): Root directory containing the dataset folders
            images (list): Optional list of image paths to use (for cross-validation)
            transform (callable): Optional transform to be applied on the images
            transform_target (bool): Whether to transform the target masks
            split (str): Dataset split ('train', 'val', 'test')
        """
        self.root_dir = root_dir
        self.transform = transform
        self.transform_target = transform_target
        self.split = split
        
        # Define class information - use original 40 classes for model training
        self.N_classes = 40  # Use original 40 classes
        self.id2label = {
            0: "background", 1: "seagrass", 2: "sand", 3: "other_coral_bleached", 4: "other_coral_bleached",
            5: "sand", 6: "rubble", 7: "algae_covered_substrate", 8: "algae_covered_substrate", 9: "fish",
            10: "algae_covered_substrate", 11: "algae_covered_substrate", 12: "algae_covered_substrate", 
            13: "background", 14: "dark", 15: "algae_covered_substrate", 16: "massive_meandering_bleached",
            17: "massive_meandering_alive", 18: "rubble", 19: "branching_bleached", 20: "branching_dead",
            21: "millepora", 22: "branching_alive", 23: "massive_meandering_dead", 24: "clam",
            25: "acropora_alive", 26: "sea_cucumber", 27: "turbinaria", 28: "table_acropora_alive",
            29: "sponge", 30: "anemone", 31: "pocillopora_alive", 32: "table_acropora_dead",
            33: "meandering_bleached", 34: "stylophora_alive", 35: "sea_urchin", 36: "meandering_alive",
            37: "meandering_dead", 38: "crown_of_thorn", 39: "dead_clam"
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # Define mapping from original 40 classes to our 3 classes
        # Based on the Coralscapes dataset class definitions
        self.class_mapping = {
            # Background classes -> 0
            0: 0,   # background
            1: 0,   # seagrass
            2: 0,   # sand
            5: 0,   # sand
            6: 0,   # rubble
            7: 0,   # algae_covered_substrate
            8: 0,   # algae_covered_substrate
            9: 0,   # fish
            10: 0,  # algae_covered_substrate
            11: 0,  # algae_covered_substrate
            12: 0,  # algae_covered_substrate
            13: 0,  # background
            14: 0,  # dark
            15: 0,  # algae_covered_substrate
            
            # Bleached coral classes -> 2
            16: 2,  # massive_meandering_bleached -> bleached
            17: 1,  # massive_meandering_alive
            18: 1,  # rubble (coral rubble)
            19: 2,  # branching_bleached -> bleached
            20: 1,  # branching_dead -> non-bleached (dead but not bleached)
            21: 1,  # millepora
            22: 1,  # branching_alive
            23: 1,  # massive_meandering_dead -> non-bleached (dead but not bleached)
            24: 1,  # clam
            25: 1,  # acropora_alive
            26: 1,  # sea_cucumber
            27: 1,  # turbinaria
            28: 1,  # table_acropora_alive
            29: 1,  # sponge
            30: 1,  # anemone
            31: 1,  # pocillopora_alive
            32: 1,  # table_acropora_dead -> non-bleached (dead but not bleached)
            33: 2,  # meandering_bleached -> bleached
            34: 1,  # stylophora_alive
            35: 1,  # sea_urchin
            36: 1,  # meandering_alive
            37: 1,  # meandering_dead -> non-bleached (dead but not bleached)
            38: 1,  # crown_of_thorn
            39: 1,  # dead_clam
            
            # Bleached coral classes -> 2
            3: 2,   # other_coral_bleached
            4: 2,   # other_coral_bleached
        }
        
        # Define paths
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_bleached_dir = os.path.join(root_dir, "masks_bleached")
        self.masks_non_bleached_dir = os.path.join(root_dir, "masks_non_bleached")
        
        # Get all image paths
        if images is None:
            self.images = sorted(glob.glob(os.path.join(self.images_dir, "*.jpg")))
        else:
            self.images = images
            
        print(f"Found {len(self.images)} images for {split} split")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        
        # Get image name without extension
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Create a 40-class mask from ground truth data
        # We'll simulate the original 40-class segmentation by assigning different
        # coral types to the bleached/non-bleached areas
        
        # Load ground truth masks
        bleached_mask_path = os.path.join(self.masks_bleached_dir, f"{img_name}_bleached.png")
        non_bleached_mask_path = os.path.join(self.masks_non_bleached_dir, f"{img_name}_non_bleached.png")
        
        # Initialize empty mask with background (class 0)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Load and assign coral classes
        if os.path.exists(non_bleached_mask_path):
            non_bleached = np.array(Image.open(non_bleached_mask_path))
            if non_bleached.ndim == 3:
                non_bleached = non_bleached[:, :, 0]  # Take first channel if RGB
            # Assign different non-bleached coral classes
            mask[non_bleached > 0] = 22  # branching_alive (most common)
        
        if os.path.exists(bleached_mask_path):
            bleached = np.array(Image.open(bleached_mask_path))
            if bleached.ndim == 3:
                bleached = bleached[:, :, 0]  # Take first channel if RGB
            # Assign different bleached coral classes
            mask[bleached > 0] = 19  # branching_bleached (most common)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            if self.transform_target:
                mask = transformed["mask"]
        
        # Convert to tensors
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).long()
        
        # Return in the format expected by the training pipeline
        return (image, mask)


def create_transforms(cfg):
    """Create data transformations based on config"""
    transforms = {}
    for split in cfg.augmentation:
        transforms[split] = A.Compose(
            [
                getattr(A, transform_name)(**transform_params) 
                for transform_name, transform_params in cfg.augmentation[split].items()
            ]
        )
    return transforms

def map_40_classes_to_3_classes(seg_mask, class_mapping):
    """Map 40-class segmentation mask to 3-class mask (background, non-bleached, bleached)"""
    mapped_mask = np.zeros(seg_mask.shape, dtype=np.uint8)
    
    # Apply class mapping
    for original_class, mapped_class in class_mapping.items():
        mapped_mask[seg_mask == original_class] = mapped_class
    
    return mapped_mask


def setup_cross_validation(dataset_dir, cfg, n_folds=5):
    """Setup cross-validation splits"""
    # Get all image paths
    images_dir = os.path.join(dataset_dir, "images")
    all_images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    
    # Create KFold object
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    return all_images, kfold


def train_fold(fold, train_images, val_images, dataset_dir, cfg, device):
    """Train a single fold"""
    print(f"\n{'='*50}")
    print(f"Training Fold {fold+1}")
    print(f"{'='*50}")
    
    # Create transforms
    transforms = create_transforms(cfg)
    
    # Create datasets
    train_dataset = CoralReefDataset(
        root_dir=dataset_dir,
        images=train_images,
        transform=transforms["train"],
        split='train'
    )
    
    val_dataset = CoralReefDataset(
        root_dir=dataset_dir,
        images=val_images,
        transform=transforms["val"],
        transform_target=cfg.training.eval.transform_target if cfg.training.eval is not None and cfg.training.eval.transform_target is not None else True,
        split='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.data.batch_size, 
        shuffle=True, 
        num_workers=2, 
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.data.batch_size_eval, 
        shuffle=False, 
        num_workers=2
    )
    
    # Calculate class weights - create a simple wrapper for calculate_weights
    if cfg.data.weight:
        # Extract labels from the dataset for weight calculation
        labels = []
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            labels.append(label.numpy())
        
        # Calculate weights manually
        all_labels = np.concatenate([label.flatten() for label in labels])
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        total_pixels = len(all_labels)
        
        # Calculate inverse frequency weights
        weights = []
        for label in range(train_dataset.N_classes):
            if label in unique_labels:
                weight = total_pixels / (len(unique_labels) * counts[unique_labels == label][0])
                weights.append(weight)
            else:
                weights.append(1.0)
        
        weight = torch.tensor(weights, dtype=torch.float32).to(device)
    else:
        weight = None
    
    # Create model
    run_name = f"{cfg.run_name}_fold{fold+1}"
    benchmark_run = Benchmark_Run(
        run_name=run_name,
        model_name=cfg.model.name,
        N_classes=train_dataset.N_classes,
        device=device,
        model_kwargs=cfg.model.kwargs,
        model_checkpoint=cfg.model.checkpoint,
        lora_kwargs=cfg.lora if hasattr(cfg, 'lora') else None,
        training_hyperparameters=cfg.training
    )
    
    # Print trainable parameters
    benchmark_run.print_trainable_parameters()
    
    # Create logger - disable wandb if project is null
    wandb_project = None
    if hasattr(cfg, 'logger') and hasattr(cfg.logger, 'wandb_project') and cfg.logger.wandb_project:
        wandb_project = cfg.logger.wandb_project
    
    logger = Logger(
        project=wandb_project,
        benchmark_run=benchmark_run,
        log_epochs=cfg.logger.log_epochs if hasattr(cfg, 'logger') and hasattr(cfg.logger, 'log_epochs') else 1,
        config=cfg,
        checkpoint_dir=f"./checkpoints/fold{fold+1}"
    )
    
    # Train the model
    benchmark_metrics = launch_benchmark(train_loader, val_loader, val_loader, benchmark_run, logger=logger)
    
    # Save the model
    save_path = f"./checkpoints/fold{fold+1}/{run_name}_final.pth"
    save_benchmark_run(benchmark_run, benchmark_metrics, save_path)
    
    return benchmark_metrics


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune DPT-DINOv2-Giant for coral bleaching detection")
    parser.add_argument("--config", type=str, default="configs/dpt-dinov2-giant.yaml", help="Path to config file")
    parser.add_argument("--dataset-dir", type=str, default="../coralscapes", help="Path to dataset directory")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--run-name", type=str, default=f"coral_bleaching_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Run name")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--batch-size-eval", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate")
    
    args = parser.parse_args()
    
    # Setup config
    base_config_path = os.path.join(os.path.dirname(args.config), "base.yaml")
    cfg = setup_config(config_path=args.config, config_base_path=base_config_path)
    
    # Update config with command line arguments
    # Create a mock args object with the expected attributes
    class MockArgs:
        def __init__(self, args):
            self.model = None
            self.data = None
            self.training = None
            self.run_name = args.run_name
            self.batch_size = args.batch_size
            self.batch_size_eval = args.batch_size
            self.epochs = args.epochs
            self.lr = args.lr if hasattr(args, 'lr') else None
            self.model_checkpoint = None
            self.weight = None
            self.r = None
            self.wandb_project = None
            self.log_epochs = None
    
    mock_args = MockArgs(args)
    cfg = update_config_with_args(cfg, mock_args)
    cfg_logger = copy.deepcopy(cfg)
    
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    # Create output directories
    os.makedirs("./checkpoints", exist_ok=True)
    
    # Setup cross-validation
    all_images, kfold = setup_cross_validation(args.dataset_dir, cfg, args.n_folds)
    
    # Train each fold
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_images)):
        train_images = [all_images[i] for i in train_idx]
        val_images = [all_images[i] for i in val_idx]
        
        metrics = train_fold(fold, train_images, val_images, args.dataset_dir, cfg, device)
        fold_metrics.append(metrics)
    
    # Calculate average metrics across folds
    print("\n" + "="*50)
    print("Cross-Validation Results")
    print("="*50)
    
    avg_metrics = {}
    for metric in fold_metrics[0].keys():
        if isinstance(fold_metrics[0][metric], (int, float)):
            avg_metrics[metric] = np.mean([fold[metric] for fold in fold_metrics])
            std_metrics = np.std([fold[metric] for fold in fold_metrics])
            print(f"{metric}: {avg_metrics[metric]:.4f} ± {std_metrics:.4f}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
