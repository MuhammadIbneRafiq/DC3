#!/usr/bin/env python
"""
Inference script for testing the trained coral bleaching detection model
"""

import os
import sys
import torch
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader
from PIL import Image
import glob
import argparse
import yaml
import copy
from datetime import datetime
import matplotlib.pyplot as plt
import cv2

# Add parent directory to path to import coralscapesScripts modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coralscapesScripts.datasets.preprocess import get_preprocessor
from coralscapesScripts.segmentation.model import Benchmark_Run
from coralscapesScripts.segmentation.model import preprocess_batch, get_batch_predictions
from coralscapesScripts.io import setup_config


class CoralReefDataset:
    """
    Dataset class for inference (simplified version)
    """
    
    def __init__(self, root_dir, images=None, transform=None, split='test'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Define class information - use original 40 classes for model training
        self.N_classes = 40
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
        
        # Define mapping from original 40 classes to our 3 classes
        self.class_mapping = {
            # no bleached classes -> 0
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
            
            # Bleached coral classes -> 1     
            3: 1,   # other_coral_bleached
            4: 1,   # other_coral_bleached
            16: 1,  # massive_meandering_bleached -> bleached
            17: 0,  # massive_meandering_alive
            18: 0,  # rubble (coral rubble)
            19: 1,  # branching_bleached -> bleached
            20: 0,  # branching_dead -> non-bleached (dead but not bleached)
            21: 0,  # millepora
            22: 0,  # branching_alive
            23: 0,  # massive_meandering_dead -> non-bleached (dead but not bleached)
            24: 0,  # clam
            25: 0,  # acropora_alive
            26: 0,  # sea_cucumber
            27: 0,  # turbinaria
            28: 0,  # table_acropora_alive
            29: 0,  # sponge
            30: 0,  # anemone
            31: 0,  #pocillopora_alive
            32: 0,  # table_acropora_dead -> non-bleached (dead but not bleached)
            33: 1,  # meandering_bleached -> bleached
            34: 0,  # stylophora_alive
            35: 0,  # sea_urchin
            36: 0,  # meandering_alive
            37: 0,  # meandering_dead -> non-bleached (dead but not bleached)
            38: 0,  # crown_of_thorn
            39: 0,  # dead_clam
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
        
        # Load ground truth masks for comparison
        bleached_mask_path = os.path.join(self.masks_bleached_dir, f"{img_name}_bleached.png")
        non_bleached_mask_path = os.path.join(self.masks_non_bleached_dir, f"{img_name}_non_bleached.png")
        
        # Create ground truth mask
        gt_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if os.path.exists(non_bleached_mask_path):
            non_bleached = np.array(Image.open(non_bleached_mask_path))
            if non_bleached.ndim == 3:
                non_bleached = non_bleached[:, :, 0]
            gt_mask[non_bleached > 0] = 0  # Non-bleached coral
        
        if os.path.exists(bleached_mask_path):
            bleached = np.array(Image.open(bleached_mask_path))
            if bleached.ndim == 3:
                bleached = bleached[:, :, 0]
            gt_mask[bleached > 0] = 1  # Bleached coral
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=gt_mask)
            image = transformed["image"]
            gt_mask = transformed["mask"]
        
        # Convert to tensors
        image = torch.from_numpy(image.transpose(2, 0, 1)).float().contiguous()
        gt_mask = torch.from_numpy(gt_mask).long().contiguous()
        
        return (image, gt_mask, img_name)


def map_40_classes_to_3_classes(seg_mask, class_mapping):
    """Map 40-class segmentation mask to 3-class mask (background, non-bleached, bleached)"""
    mapped_mask = np.zeros(seg_mask.shape, dtype=np.uint8)
    
    # Apply class mapping
    for original_class, mapped_class in class_mapping.items():
        mapped_mask[seg_mask == original_class] = mapped_class
    
    return mapped_mask


def create_inference_transforms():
    """Create transforms for inference"""
    return A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_trained_model(checkpoint_path, device):
    """Load the trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    else:
        model_state_dict = checkpoint
    
    # Create a simple config object for the model
    class SimpleConfig:
        def __init__(self):
            self.epochs = 1
            self.optimizer = {"type": "torch.optim.AdamW", "lr": 0.00005}
            self.scheduler = {"type": "torch.optim.lr_scheduler.PolynomialLR", "power": 1}
            self.loss = {"type": "cross_entropy"}
            self.preprocessor = "dpt"
            self.eval = {"transform_target": True}
    
    config = SimpleConfig()
    
    # Create model instance
    model = Benchmark_Run(
        run_name="inference_model",
        model_name="dpt-dinov2-giant",
        N_classes=40,
        device=device,
        model_kwargs={"num_labels": 40},
        training_hyperparameters=config
    )
    
    # Load the trained weights
    model.model.load_state_dict(model_state_dict)
    model.model.eval()
    
    return model


def run_inference(model, dataloader, device, output_dir, dataset):
    """Run inference on the test set"""
    print(f"Running inference on {len(dataloader)} batches...")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)
    
    all_predictions = []
    all_ground_truths = []
    all_image_names = []
    
    with torch.no_grad():
        for batch_idx, (images, gt_masks, img_names) in enumerate(dataloader):
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
            
            # Preprocess batch
            preprocessed_batch = preprocess_batch((images, gt_masks), model.preprocessor)
            
            if isinstance(preprocessed_batch, dict) and "pixel_values" in preprocessed_batch:
                # For DPT model
                inputs = preprocessed_batch["pixel_values"].to(device)
                outputs = model.model(inputs)
                
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                    # Resize to match input size
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=gt_masks.shape[-2:], mode="bilinear", align_corners=False
                    )
                
                # Get predictions
                predictions = outputs.argmax(dim=1)
            else:
                # For other models
                inputs = images.to(device)
                outputs = model.model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                predictions = outputs.argmax(dim=1)
            
            # Convert to numpy
            predictions = predictions.cpu().numpy()
            gt_masks = gt_masks.numpy()
            
            # Process each image in the batch
            for i in range(len(img_names)):
                img_name = img_names[i]
                pred = predictions[i]
                gt = gt_masks[i]
                
                # Map 40 classes to 3 classes
                pred_3class = map_40_classes_to_3_classes(pred, dataset.class_mapping)
                gt_3class = gt  # Ground truth is already in 3 classes
                
                # Save predictions
                pred_path = os.path.join(output_dir, "predictions", f"{img_name}_pred.png")
                cv2.imwrite(pred_path, (pred_3class * 127).astype(np.uint8))  # Scale for visualization
                
                # Create overlay
                original_img = images[i].permute(1, 2, 0).numpy()
                original_img = (original_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                original_img = np.clip(original_img, 0, 255).astype(np.uint8)
                
                # Resize to original size
                original_img = cv2.resize(original_img, (gt.shape[1], gt.shape[0]))
                
                # Create colored overlay
                overlay = original_img.copy()
                overlay[pred_3class == 1] = [255, 0, 0]  # Red for bleached coral
                overlay[pred_3class == 0] = [0, 255, 0]  # Green for non-bleached coral
                
                # Blend with original
                blended = cv2.addWeighted(original_img, 0.7, overlay, 0.3, 0)
                
                # Save overlay
                overlay_path = os.path.join(output_dir, "overlays", f"{img_name}_overlay.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                
                all_predictions.append(pred_3class)
                all_ground_truths.append(gt_3class)
                all_image_names.append(img_name)
    
    return all_predictions, all_ground_truths, all_image_names


def calculate_metrics(predictions, ground_truths):
    """Calculate evaluation metrics"""
    from sklearn.metrics import accuracy_score, jaccard_score
    
    # Flatten all predictions and ground truths
    all_preds = np.concatenate([pred.flatten() for pred in predictions])
    all_gts = np.concatenate([gt.flatten() for gt in ground_truths])
    
    # Calculate metrics
    accuracy = accuracy_score(all_gts, all_preds)
    iou = jaccard_score(all_gts, all_preds, average='macro', zero_division=0)
    
    # Per-class IoU
    iou_per_class = jaccard_score(all_gts, all_preds, average=None, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'mean_iou': iou,
        'iou_per_class': iou_per_class
    }


def main():
    parser = argparse.ArgumentParser(description="Run inference on trained coral bleaching model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset-dir", type=str, default="./test_data", help="Path to test dataset")
    parser.add_argument("--output-dir", type=str, default="./inference_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create transforms
    transforms = create_inference_transforms()
    
    # Create dataset
    dataset = CoralReefDataset(
        root_dir=args.dataset_dir,
        transform=transforms,
        split='test'
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Load trained model
    model = load_trained_model(args.checkpoint, device)
    
    # Run inference
    predictions, ground_truths, image_names = run_inference(model, dataloader, device, args.output_dir, dataset)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truths)
    
    # Print results
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"IoU per class:")
    print(f"  Background: {metrics['iou_per_class'][0]:.4f}")
    print(f"  Non-bleached coral: {metrics['iou_per_class'][1]:.4f}")
    print(f"  Bleached coral: {metrics['iou_per_class'][2]:.4f}")
    print(f"\nResults saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
