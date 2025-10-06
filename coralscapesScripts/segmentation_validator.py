import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn.functional as F

def calculate_iou(pred_mask, gt_mask, class_id):
    """Calculate Intersection over Union for a specific class"""
    pred_class = (pred_mask == class_id)
    gt_class = (gt_mask == class_id)
    
    intersection = np.logical_and(pred_class, gt_class).sum()
    union = np.logical_or(pred_class, gt_class).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def calculate_dice(pred_mask, gt_mask, class_id):
    """Calculate Dice coefficient for a specific class"""
    pred_class = (pred_mask == class_id)
    gt_class = (gt_mask == class_id)
    
    intersection = np.logical_and(pred_class, gt_class).sum()
    total = pred_class.sum() + gt_class.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2 * intersection) / total

def find_matching_files(pred_path, masks_bleached_dir, masks_non_bleached_dir):
    """Find corresponding ground truth masks for a prediction file"""
    base_name = os.path.splitext(os.path.basename(pred_path))[0]
    
    # Remove '_pred' suffix if present
    if base_name.endswith('_pred'):
        base_name = base_name[:-5]
    
    # Look for bleached mask
    bleached_path = os.path.join(masks_bleached_dir, f"{base_name}_bleached.png")
    if not os.path.exists(bleached_path):
        # Try alternative naming
        bleached_path = os.path.join(masks_bleached_dir, f"{base_name}.png")
    
    # Look for non-bleached mask
    non_bleached_path = os.path.join(masks_non_bleached_dir, f"{base_name}_non_bleached.png")
    if not os.path.exists(non_bleached_path):
        # Try alternative naming
        non_bleached_path = os.path.join(masks_non_bleached_dir, f"{base_name}.png")
    
    return bleached_path, non_bleached_path

def create_combined_gt_mask(bleached_path, non_bleached_path, pred_shape):
    """Create combined ground truth mask from bleached and non-bleached masks"""
    combined_mask = np.zeros(pred_shape, dtype=np.uint8)
    
    if os.path.exists(bleached_path):
        bleached_mask = np.array(Image.open(bleached_path))
        if bleached_mask.ndim == 3:
            bleached_mask = bleached_mask[:, :, 0]  # Take first channel if RGB
        # Resize to match prediction
        if bleached_mask.shape != pred_shape:
            bleached_mask = Image.fromarray(bleached_mask).resize((pred_shape[1], pred_shape[0]), Image.NEAREST)
            bleached_mask = np.array(bleached_mask)
        combined_mask[bleached_mask > 0] = 2  # Bleached = 2
    
    if os.path.exists(non_bleached_path):
        non_bleached_mask = np.array(Image.open(non_bleached_path))
        if non_bleached_mask.ndim == 3:
            non_bleached_mask = non_bleached_mask[:, :, 0]  # Take first channel if RGB
        # Resize to match prediction
        if non_bleached_mask.shape != pred_shape:
            non_bleached_mask = Image.fromarray(non_bleached_mask).resize((pred_shape[1], pred_shape[0]), Image.NEAREST)
            non_bleached_mask = np.array(non_bleached_mask)
        combined_mask[non_bleached_mask > 0] = 1  # Non-bleached = 1
    
    return combined_mask

def create_pred_mask_from_segmentation(seg_mask, coral_classes):
    """Convert segmentation mask to bleached/non-bleached prediction"""
    pred_mask = np.zeros(seg_mask.shape, dtype=np.uint8)
    
    # Map coral classes to bleached/non-bleached
    for coral_type, class_id in coral_classes.items():
        if 'alive' in coral_type:
            pred_mask[seg_mask == class_id] = 1  # Non-bleached
        elif 'bleached' in coral_type:
            pred_mask[seg_mask == class_id] = 2  # Bleached
    
    return pred_mask

def validate_segmentation(pred_path, masks_bleached_dir, masks_non_bleached_dir, output_dir="validation_results"):
    """Validate segmentation predictions against ground truth masks"""
    
    # Define coral classes
    coral_classes = {
        'branching_alive': 22, 'massive_meandering_alive': 17, 'acropora_alive': 25,
        'table_acropora_alive': 28, 'pocillopora_alive': 31, 'stylophora_alive': 34,
        'meandering_alive': 36, 'branching_bleached': 19, 'massive_meandering_bleached': 16,
        'other_coral_bleached': 4, 'meandering_bleached': 33
    }
    
    # Load prediction mask
    pred_seg_mask = np.array(Image.open(pred_path))
    if pred_seg_mask.ndim == 3:
        pred_seg_mask = pred_seg_mask[:, :, 0]  # Take first channel if RGB
    
    # Create prediction mask (1=non-bleached, 2=bleached, 0=background)
    pred_mask = create_pred_mask_from_segmentation(pred_seg_mask, coral_classes)
    
    # Find corresponding ground truth masks
    bleached_path, non_bleached_path = find_matching_files(pred_path, masks_bleached_dir, masks_non_bleached_dir)
    
    print(f"Looking for ground truth masks:")
    print(f"  Bleached: {bleached_path} (exists: {os.path.exists(bleached_path)})")
    print(f"  Non-bleached: {non_bleached_path} (exists: {os.path.exists(non_bleached_path)})")
    
    if not os.path.exists(bleached_path) and not os.path.exists(non_bleached_path):
        print(f"No ground truth masks found for {os.path.basename(pred_path)}")
        return None
    
    # Create combined ground truth mask
    gt_mask = create_combined_gt_mask(bleached_path, non_bleached_path, pred_mask.shape)
    
    # Calculate metrics
    metrics = {}
    
    print(f"Ground truth mask shape: {gt_mask.shape}, unique values: {np.unique(gt_mask)}")
    print(f"Prediction mask shape: {pred_mask.shape}, unique values: {np.unique(pred_mask)}")
    
    # Overall accuracy
    metrics['accuracy'] = accuracy_score(gt_mask.flatten(), pred_mask.flatten())
    print(f"Overall accuracy: {metrics['accuracy']:.3f}")
    
    # Per-class metrics
    for class_name, class_id in [('background', 0), ('non_bleached', 1), ('bleached', 2)]:
        if class_id in gt_mask:
            iou = calculate_iou(pred_mask, gt_mask, class_id)
            dice = calculate_dice(pred_mask, gt_mask, class_id)
            precision = precision_score(gt_mask.flatten(), pred_mask.flatten(), 
                                      labels=[class_id], average=None, zero_division=0)[0]
            recall = recall_score(gt_mask.flatten(), pred_mask.flatten(), 
                                labels=[class_id], average=None, zero_division=0)[0]
            f1 = f1_score(gt_mask.flatten(), pred_mask.flatten(), 
                         labels=[class_id], average=None, zero_division=0)[0]
            
            metrics[f'{class_name}_iou'] = iou
            metrics[f'{class_name}_dice'] = dice
            metrics[f'{class_name}_precision'] = precision
            metrics[f'{class_name}_recall'] = recall
            metrics[f'{class_name}_f1'] = f1
            
            print(f"{class_name}: IoU={iou:.3f}, Dice={dice:.3f}, F1={f1:.3f}")
    
    # Mean IoU for coral classes only
    coral_iou = []
    for class_id in [1, 2]:  # non-bleached and bleached
        if class_id in gt_mask:
            coral_iou.append(calculate_iou(pred_mask, gt_mask, class_id))
    metrics['mean_iou_corals'] = np.mean(coral_iou) if coral_iou else 0.0
    
    # Create visualization
    create_validation_visualization(pred_mask, gt_mask, pred_path, metrics, output_dir)
    
    return metrics

def create_validation_visualization(pred_mask, gt_mask, pred_path, metrics, output_dir):
    """Create visualization comparing prediction and ground truth"""
    
    # Load original image if available
    base_name = os.path.splitext(os.path.basename(pred_path))[0]
    if base_name.endswith('_pred'):
        base_name = base_name[:-5]
    
    original_paths = [
        f"../coralscapes/images/{base_name}.jpg",
        f"../coralscapes/images/{base_name}.JPG"
    ]
    
    original_img = None
    for path in original_paths:
        if os.path.exists(path):
            original_img = np.array(Image.open(path))
            original_img = Image.fromarray(original_img).resize((pred_mask.shape[1], pred_mask.shape[0]))
            original_img = np.array(original_img)
            break
    
    if original_img is None:
        original_img = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    
    # Create color-coded masks
    def create_colored_mask(mask):
        colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored[mask == 1] = [0, 255, 0]  # Green for non-bleached
        colored[mask == 2] = [255, 0, 0]  # Red for bleached
        return colored
    
    pred_colored = create_colored_mask(pred_mask)
    gt_colored = create_colored_mask(gt_mask)
    
    # Create overlay
    pred_overlay = 0.7 * original_img + 0.3 * pred_colored
    gt_overlay = 0.7 * original_img + 0.3 * gt_colored
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Ground truth
    axes[0, 1].imshow(gt_overlay.astype(np.uint8))
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    # Prediction
    axes[0, 2].imshow(pred_overlay.astype(np.uint8))
    axes[0, 2].set_title('DINO Model Prediction')
    axes[0, 2].axis('off')
    
    # Ground truth mask only
    axes[1, 0].imshow(gt_colored)
    axes[1, 0].set_title('GT Mask (Green=Non-bleached, Red=Bleached)')
    axes[1, 0].axis('off')
    
    # Prediction mask only
    axes[1, 1].imshow(pred_colored)
    axes[1, 1].set_title('Pred Mask (Green=Non-bleached, Red=Bleached)')
    axes[1, 1].axis('off')
    
    # Metrics
    axes[1, 2].axis('off')
    metrics_text = f"""SEGMENTATION METRICS:
    
Overall Accuracy: {metrics['accuracy']:.3f}
Mean IoU (Corals): {metrics['mean_iou_corals']:.3f}

Non-bleached Corals:
  IoU: {metrics['non_bleached_iou']:.3f}
  Dice: {metrics['non_bleached_dice']:.3f}
  F1: {metrics['non_bleached_f1']:.3f}

Bleached Corals:
  IoU: {metrics['bleached_iou']:.3f}
  Dice: {metrics['bleached_dice']:.3f}
  F1: {metrics['bleached_f1']:.3f}"""
    
    axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pred_path))[0]
    plt.savefig(os.path.join(output_dir, f"{base_name}_validation.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation visualization saved to: {os.path.join(output_dir, f'{base_name}_validation.png')}")

def validate_all_predictions(pred_dir, masks_bleached_dir, masks_non_bleached_dir, output_dir="validation_results"):
    """Validate all prediction files"""
    
    pred_files = glob.glob(os.path.join(pred_dir, "*_pred.png"))
    all_metrics = []
    
    print(f"Found {len(pred_files)} prediction files to validate")
    
    for pred_file in pred_files:
        print(f"\nValidating: {os.path.basename(pred_file)}")
        metrics = validate_segmentation(pred_file, masks_bleached_dir, masks_non_bleached_dir, output_dir)
        
        if metrics:
            all_metrics.append(metrics)
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Mean IoU (Corals): {metrics['mean_iou_corals']:.3f}")
            print(f"  Non-bleached IoU: {metrics['non_bleached_iou']:.3f}")
            print(f"  Bleached IoU: {metrics['bleached_iou']:.3f}")
    
    if all_metrics:
        # Calculate average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        print(f"\n{'='*50}")
        print("AVERAGE VALIDATION METRICS:")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {avg_metrics['accuracy']:.3f}")
        print(f"Mean IoU (Corals): {avg_metrics['mean_iou_corals']:.3f}")
        print(f"Non-bleached IoU: {avg_metrics['non_bleached_iou']:.3f}")
        print(f"Bleached IoU: {avg_metrics['bleached_iou']:.3f}")
        print(f"Non-bleached F1: {avg_metrics['non_bleached_f1']:.3f}")
        print(f"Bleached F1: {avg_metrics['bleached_f1']:.3f}")
        
        return avg_metrics
    
    return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Validate single file
        pred_file = sys.argv[1]
        masks_bleached_dir = "../coralscapes/masks_bleached"
        masks_non_bleached_dir = "../coralscapes/masks_non_bleached"
        
        validate_segmentation(pred_file, masks_bleached_dir, masks_non_bleached_dir)
    else:
        # Validate all files
        pred_dir = "../inference_results"
        masks_bleached_dir = "../coralscapes/masks_bleached"
        masks_non_bleached_dir = "../coralscapes/masks_non_bleached"
        
        validate_all_predictions(pred_dir, masks_bleached_dir, masks_non_bleached_dir)
