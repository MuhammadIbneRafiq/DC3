import os
import sys
import torch
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader
from PIL import Image
import glob
import argparse
import cv2
from sklearn.metrics import accuracy_score, jaccard_score  # pyright: ignore[reportMissingImports]

# Add parent directory to path to import coralscapesScripts modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coralscapesScripts.segmentation.model import Benchmark_Run

class SimpleCoralDataset:
    """Simple dataset for inference"""
    
    def __init__(self, root_dir, images=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Define paths
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_bleached_dir = os.path.join(root_dir, "masks_bleached")
        self.masks_non_bleached_dir = os.path.join(root_dir, "masks_non_bleached")
        
        # Get all image paths
        if images is None:
            self.images = sorted(glob.glob(os.path.join(self.images_dir, "*.jpg")))
        else:
            self.images = images
            
        print(f"Found {len(self.images)} images for inference")
    
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


def create_inference_transforms():
    """Create transforms for inference"""
    return A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint with proper handling of LoRA weights"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create a simple config for the model
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
    
    # Try to load the state dict with strict=False to handle LoRA weights
    try:
        if "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
        else:
            model_state_dict = checkpoint
        
        # Load with strict=False to ignore missing keys
        missing_keys, unexpected_keys = model.model.load_state_dict(model_state_dict, strict=False)
        
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        
        if len(missing_keys) > 0:
            print("Some keys were missing, but continuing with inference...")
        
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Continuing with pre-trained weights only...")
    
    model.model.eval()
    return model


def run_simple_inference(model, dataloader, device, output_dir):
    """Run simple inference on the test set"""
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
            
            try:
                # Preprocess batch
                preprocessed_batch = model.preprocessor(images)
                
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
                    
                    # Map 40 classes to 2 classes (non-bleached coral=0, bleached coral=1)
                    pred_2class = np.zeros_like(pred)
                    
                    # Map bleached coral classes to class 1
                    bleached_classes = [3, 4, 16, 19, 33]  # bleached coral classes
                    for class_id in bleached_classes:
                        pred_2class[pred == class_id] = 1
                    
                    # Map non-bleached coral classes to class 0
                    non_bleached_classes = [17, 20, 22, 23, 25, 28, 31, 32, 34, 36, 37]  # non-bleached coral classes
                    for class_id in non_bleached_classes:
                        pred_2class[pred == class_id] = 0
                    
                    # Everything else remains 0 (treated as non-bleached)
                    
                    # Save predictions
                    pred_path = os.path.join(output_dir, "predictions", f"{img_name}_pred.png")
                    cv2.imwrite(pred_path, (pred_2class * 127).astype(np.uint8))  # Scale for visualization
                    
                    # Create overlay
                    original_img = images[i].permute(1, 2, 0).numpy()
                    original_img = (original_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                    original_img = np.clip(original_img, 0, 255).astype(np.uint8)
                    
                    # Resize to original size
                    original_img = cv2.resize(original_img, (gt.shape[1], gt.shape[0]))
                    
                    # Create colored overlay
                    overlay = original_img.copy()
                    overlay[pred_2class == 1] = [255, 0, 0]  # Red for bleached coral
                    # Background remains original color
                    
                    # Blend with original
                    blended = cv2.addWeighted(original_img, 0.7, overlay, 0.3, 0)
                    
                    # Save overlay
                    overlay_path = os.path.join(output_dir, "overlays", f"{img_name}_overlay.png")
                    cv2.imwrite(overlay_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                    
                    all_predictions.append(pred_2class)
                    all_ground_truths.append(gt)
                    all_image_names.append(img_name)
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx + 1}: {e}")
                continue
    
    return all_predictions, all_ground_truths, all_image_names


def calculate_metrics(predictions, ground_truths):
    """Calculate evaluation metrics"""
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
    parser = argparse.ArgumentParser(description="Run simple inference on trained coral bleaching model")
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
    dataset = SimpleCoralDataset(
        root_dir=args.dataset_dir,
        transform=transforms
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Load trained model
    model = load_model_from_checkpoint(args.checkpoint, device)
    
    # Run inference
    predictions, ground_truths, image_names = run_simple_inference(model, dataloader, device, args.output_dir)
    
    if len(predictions) > 0:
        # Calculate metrics
        metrics = calculate_metrics(predictions, ground_truths)
        
        # Print results
        print("\n" + "="*60)
        print("INFERENCE RESULTS")
        print("="*60)
        print(f"Processed {len(predictions)} images")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"IoU per class:")
        for i, class_name in enumerate(['Non-bleached coral', 'Bleached coral']):
            if i < len(metrics['iou_per_class']):
                print(f"  {class_name}: {metrics['iou_per_class'][i]:.4f}")
            else:
                print(f"  {class_name}: N/A (not present in ground truth)")
        print(f"\nResults saved to: {args.output_dir}")
        print("="*60)
    else:
        print("No predictions were generated. Check the error messages above.")


if __name__ == "__main__":
    main()
