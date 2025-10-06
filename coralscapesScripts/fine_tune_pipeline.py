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
from sklearn.model_selection import KFold  # pyright: ignore[reportMissingImports]
import argparse
import yaml
import copy
import time
from datetime import datetime, timedelta
from tqdm import tqdm

# Add parent directory to path to import coralscapesScripts modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coralscapesScripts.datasets.preprocess import get_preprocessor
from coralscapesScripts.datasets.utils import calculate_weights
from coralscapesScripts.segmentation.model import Benchmark_Run
from coralscapesScripts.segmentation.benchmarking import launch_benchmark
from coralscapesScripts.segmentation.model import preprocess_batch, get_batch_predictions, get_windows
from coralscapesScripts.segmentation.eval import Evaluator
from coralscapesScripts.logger import Logger, save_benchmark_run, save_model_checkpoint
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


def train_epoch_with_progress(benchmark_run, train_loader, epoch, total_epochs, epoch_start_time):
    """Train one epoch with detailed progress tracking"""
    benchmark_run.model.train(True)
    if hasattr(benchmark_run.optimizer, 'train'):
        benchmark_run.optimizer.train()
    
    running_loss = 0.0
    total_batches = len(train_loader)
    batch_times = []
    
    # Create progress bar for batches
    batch_pbar = tqdm(enumerate(train_loader), total=total_batches, 
                      desc=f"Epoch {epoch+1}/{total_epochs} - Training", 
                      leave=False, unit="batch")
    
    for batch_idx, data in batch_pbar:
        batch_start_time = time.time()
        
        benchmark_run.optimizer.zero_grad()
        preprocessed_batch = preprocess_batch(data, benchmark_run.preprocessor)
        outputs, loss = get_batch_predictions(preprocessed_batch, benchmark_run.model, benchmark_run.device, loss_fn=benchmark_run.loss)
        loss.backward()
        benchmark_run.optimizer.step()
        
        running_loss += loss.item()
        
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        # Update progress bar with current metrics
        avg_loss = running_loss / (batch_idx + 1)
        avg_batch_time = np.mean(batch_times)
        images_processed = (batch_idx + 1) * train_loader.batch_size
        total_images = total_batches * train_loader.batch_size
        images_left = total_images - images_processed
        
        # Calculate time estimates
        epoch_elapsed = time.time() - epoch_start_time
        estimated_batch_time = avg_batch_time * images_left / train_loader.batch_size
        estimated_epoch_time = epoch_elapsed + estimated_batch_time
        
        batch_pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Imgs': f'{images_processed}/{total_images}',
            'Left': f'{images_left}',
            'Time/batch': f'{avg_batch_time:.2f}s',
            'ETA': f'{estimated_batch_time:.0f}s'
        })
    
    batch_pbar.close()
    return running_loss / total_batches


def val_epoch_with_progress(benchmark_run, val_loader, epoch, total_epochs):
    """Validate one epoch with detailed progress tracking"""
    benchmark_run.model.eval()
    if hasattr(benchmark_run.optimizer, 'train'):
        benchmark_run.optimizer.eval()
    
    running_loss = 0.0
    total_batches = len(val_loader)
    batch_times = []
    
    with torch.no_grad():
        # Create progress bar for validation
        val_pbar = tqdm(enumerate(val_loader), total=total_batches,
                        desc=f"Epoch {epoch+1}/{total_epochs} - Validation",
                        leave=False, unit="batch")
        
        for batch_idx, data in val_pbar:
            batch_start_time = time.time()
            
            if len(data[0]) == 1:  # Skip if batch size is one
                continue
                
            if benchmark_run.eval and benchmark_run.eval.sliding_window:
                input_windows, label_windows = get_windows(data, benchmark_run.eval.window, 
                                                         benchmark_run.eval.stride, 
                                                         benchmark_run.eval.window_target, 
                                                         benchmark_run.eval.stride_target)
                input_windows = input_windows.view(-1, *input_windows.shape[-3:])
                label_windows = label_windows.view(-1, *label_windows.shape[-2:])
                data = (input_windows, label_windows)
            
            preprocessed_batch = preprocess_batch(data, benchmark_run.preprocessor)
            outputs, loss = get_batch_predictions(preprocessed_batch, benchmark_run.model, 
                                                benchmark_run.device, loss_fn=benchmark_run.loss)
            running_loss += loss.item()
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            avg_batch_time = np.mean(batch_times)
            images_processed = (batch_idx + 1) * val_loader.batch_size
            total_images = total_batches * val_loader.batch_size
            
            val_pbar.set_postfix({
                'Val_Loss': f'{avg_loss:.4f}',
                'Imgs': f'{images_processed}/{total_images}',
                'Time/batch': f'{avg_batch_time:.2f}s'
            })
        
        val_pbar.close()
    
    return running_loss / total_batches


def train_with_progress_tracking(train_loader, val_loader, benchmark_run, logger, cfg):
    """Enhanced training loop with comprehensive progress tracking"""
    best_val_mean_iou = 0.
    best_epoch = -1
    
    total_epochs = benchmark_run.training_hyperparameters["epochs"]
    fold_start_time = time.time()
    
    # Create evaluator
    if hasattr(benchmark_run, "preprocessor"):
        evaluator = Evaluator(N_classes=benchmark_run.N_classes, device=benchmark_run.device, 
                             preprocessor=benchmark_run.preprocessor, eval_params=benchmark_run.eval)
    else:
        evaluator = Evaluator(N_classes=benchmark_run.N_classes, device=benchmark_run.device, 
                             eval_params=benchmark_run.eval)
    
    print(f"\n🚀 Starting training for {total_epochs} epochs")
    print(f"📊 Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")
    print(f"🖼️  Training images: {len(train_loader) * train_loader.batch_size} | Validation images: {len(val_loader) * val_loader.batch_size}")
    print(f"⏱️  Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Main training loop with progress tracking
    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        
        print(f"\n📈 EPOCH {epoch+1}/{total_epochs}")
        print("-" * 50)
        
        # Training phase
        train_loss = train_epoch_with_progress(benchmark_run, train_loader, epoch, total_epochs, epoch_start_time)
        
        # Scheduler step
        if hasattr(benchmark_run, "scheduler"):
            benchmark_run.scheduler.step()
        
        # Validation phase
        val_loss = val_epoch_with_progress(benchmark_run, val_loader, epoch, total_epochs)
        
        # Calculate epoch timing
        epoch_time = time.time() - epoch_start_time
        total_time_elapsed = time.time() - fold_start_time
        avg_epoch_time = total_time_elapsed / (epoch + 1)
        estimated_remaining_time = avg_epoch_time * (total_epochs - epoch - 1)
        
        # Print epoch summary
        print(f"✅ Epoch {epoch+1} completed in {epoch_time:.1f}s")
        print(f"📉 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"⏰ Time elapsed: {timedelta(seconds=int(total_time_elapsed))} | "
              f"ETA: {timedelta(seconds=int(estimated_remaining_time))}")
        
        # Calculate detailed metrics for every epoch
        print("\n🔍 Calculating detailed metrics...")
        
        # Train metrics
        train_metric_results = evaluator.evaluate_model(train_loader, benchmark_run.model, split="train")
        
        # Validation metrics
        metric_results = evaluator.evaluate_model(val_loader, benchmark_run.model, split="validation")
        
        # Print detailed metrics table
        print("\n📊 DETAILED METRICS:")
        print("-" * 80)
        print(f"{'Metric':<15} | {'Train':<15} | {'Validation':<15}")
        print("-" * 80)
        
        # Print common metrics in a table format
        for metric_name in sorted(metric_results.keys()):
            if metric_name in train_metric_results:
                train_value = train_metric_results[metric_name]
                val_value = metric_results[metric_name]
                if isinstance(train_value, (int, float)):
                    print(f"{metric_name:<15} | {train_value:<15.4f} | {val_value:<15.4f}")
        
        print("-" * 80)
        
        # Log to wandb if available
        if logger:
            logger.log({
                "train/loss": train_loss,
                "validation/loss": val_loss,
                "train/time_taken": epoch_time,
                "train/total_time_elapsed": total_time_elapsed,
                "train/estimated_remaining_time": estimated_remaining_time,
                **{f"train/{metric_name}": metric for metric_name, metric in train_metric_results.items()},
                **{f"validation/{metric_name}": metric for metric_name, metric in metric_results.items()}
            }, step=epoch)
            
            # Log images at specified intervals
            if epoch % logger.log_epochs == 0 or epoch == total_epochs - 1:
                if logger.logger:
                    logger.log_image_predictions(*evaluator.evaluate_image(train_loader, benchmark_run.model, split="train", epoch=epoch), epoch, split="train")
                    logger.log_image_predictions(*evaluator.evaluate_image(val_loader, benchmark_run.model, split="validation", epoch=epoch), epoch, split="validation")
            
            # Track best performance and save model
            if best_val_mean_iou < metric_results["mean_iou"]:
                best_vloss = val_loss
                best_val_mean_iou = metric_results["mean_iou"]
                best_val_mean_accuracy = metric_results["accuracy"]
                best_epoch = epoch
                
                print(f"🏆 New best model! IoU: {best_val_mean_iou:.4f} (Epoch {epoch+1})")
                save_model_checkpoint(benchmark_run, epoch, train_loss, val_loss, 
                                    best_val_mean_iou, best_val_mean_accuracy, logger)
                
            # Periodic checkpoints
            if (epoch % logger.log_checkpoint == 0) and epoch > 0:
                save_model_checkpoint(benchmark_run, epoch, train_loss, val_loss, 
                                    metric_results["mean_iou"], metric_results["accuracy"], 
                                    logger, final_checkpoint=False)
            
            # Final checkpoint
            if epoch == total_epochs - 1:
                save_model_checkpoint(benchmark_run, epoch, train_loss, val_loss, 
                                    metric_results["mean_iou"], metric_results["accuracy"], 
                                    logger, final_checkpoint=True)
        
        print("-" * 50)
    
    # Training completion summary
    total_training_time = time.time() - fold_start_time
    print(f"\n🎉 Training completed!")
    print(f"⏱️  Total training time: {timedelta(seconds=int(total_training_time))}")
    print(f"🏆 Best validation IoU: {best_val_mean_iou:.4f} (Epoch {best_epoch+1})")
    print(f"📈 Best validation accuracy: {best_val_mean_accuracy:.4f}")
    print("="*80)
    
    results_dict = {
        "validation_loss": best_vloss,
        "validation_mean_iou": best_val_mean_iou,
        "validation_mean_accuracy": best_val_mean_accuracy,
        "best_epoch": best_epoch,
        "total_training_time": total_training_time
    }
    
    return results_dict


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
    
    # Train the model with enhanced progress tracking
    benchmark_metrics = train_with_progress_tracking(train_loader, val_loader, benchmark_run, logger, cfg)
    
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
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cuda:0, cuda:1, cpu)")
    
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
    if args.device.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(args.device)
            if args.device == "cuda:1" and torch.cuda.device_count() < 2:
                print("Warning: GPU 1 requested but only one GPU available, using cuda:0")
                device = torch.device("cuda:0")
            print(f"Using device: {device}")
            if ":" in args.device:
                gpu_id = args.device.split(":")[1]
                print(f"GPU {gpu_id} name: {torch.cuda.get_device_name(int(gpu_id))}")
        else:
            print("CUDA not available, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
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
    
    # Train each fold with cross-validation progress tracking
    fold_metrics = []
    total_folds = args.n_folds
    cv_start_time = time.time()
    
    print(f"\n🔄 Starting {total_folds}-fold cross-validation")
    print(f"📁 Total images: {len(all_images)}")
    print(f"⏱️  Cross-validation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_images)):
        train_images = [all_images[i] for i in train_idx]
        val_images = [all_images[i] for i in val_idx]
        
        print(f"\n🔄 Fold {fold+1}/{total_folds}")
        print(f"📊 Train images: {len(train_images)} | Validation images: {len(val_images)}")
        
        fold_start_time = time.time()
        metrics = train_fold(fold, train_images, val_images, args.dataset_dir, cfg, device)
        fold_time = time.time() - fold_start_time
        
        fold_metrics.append(metrics)
        
        # Calculate remaining time for cross-validation
        elapsed_cv_time = time.time() - cv_start_time
        avg_fold_time = elapsed_cv_time / (fold + 1)
        remaining_folds = total_folds - fold - 1
        estimated_remaining_cv_time = avg_fold_time * remaining_folds
        
        print(f"✅ Fold {fold+1} completed in {timedelta(seconds=int(fold_time))}")
        print(f"⏰ CV Time elapsed: {timedelta(seconds=int(elapsed_cv_time))} | "
              f"CV ETA: {timedelta(seconds=int(estimated_remaining_cv_time))}")
        
        if fold < total_folds - 1:
            print(f"🔄 Starting next fold in 2 seconds...")
            time.sleep(2)
    
    # Calculate average metrics across folds
    total_cv_time = time.time() - cv_start_time
    print("\n" + "="*80)
    print("🎉 CROSS-VALIDATION RESULTS")
    print("="*80)
    
    # Collect all metrics
    avg_metrics = {}
    std_metrics = {}
    for metric in fold_metrics[0].keys():
        if isinstance(fold_metrics[0][metric], (int, float)):
            avg_metrics[metric] = np.mean([fold[metric] for fold in fold_metrics])
            std_metrics[metric] = np.std([fold[metric] for fold in fold_metrics])
    
    # Print metrics in a well-formatted table
    print("\n📊 PERFORMANCE METRICS:")
    print("-" * 60)
    print(f"{'Metric':<25} | {'Mean':<15} | {'Std Dev':<15}")
    print("-" * 60)
    
    # Sort metrics for better readability
    sorted_metrics = sorted(avg_metrics.keys())
    for metric in sorted_metrics:
        print(f"{metric:<25} | {avg_metrics[metric]:<15.4f} | {std_metrics[metric]:<15.4f}")
    
    print("-" * 60)
    
    # Print timing information
    print("\n⏱️  TIMING INFORMATION:")
    print("-" * 60)
    print(f"Total cross-validation time: {timedelta(seconds=int(total_cv_time))}")
    print(f"Average time per fold: {timedelta(seconds=int(total_cv_time / total_folds))}")
    print(f"Average time per epoch: {timedelta(seconds=int(total_cv_time / (total_folds * args.epochs)))}")
    print("-" * 60)
    
    # Print key performance indicators
    print("\n🏆 KEY PERFORMANCE INDICATORS:")
    print("-" * 60)
    print(f"Best validation IoU: {avg_metrics.get('validation_mean_iou', 0):.4f} ± {std_metrics.get('validation_mean_iou', 0):.4f}")
    print(f"Best validation accuracy: {avg_metrics.get('validation_mean_accuracy', 0):.4f} ± {std_metrics.get('validation_mean_accuracy', 0):.4f}")
    if 'precision' in avg_metrics:
        print(f"Precision: {avg_metrics.get('precision', 0):.4f} ± {std_metrics.get('precision', 0):.4f}")
    if 'recall' in avg_metrics:
        print(f"Recall: {avg_metrics.get('recall', 0):.4f} ± {std_metrics.get('recall', 0):.4f}")
    if 'f1' in avg_metrics:
        print(f"F1 Score: {avg_metrics.get('f1', 0):.4f} ± {std_metrics.get('f1', 0):.4f}")
    print("-" * 60)
    
    print("\n" + "="*80)
    print("✅ Training complete!")
    print("="*80)


if __name__ == "__main__":
    main()
