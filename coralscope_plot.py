import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools import mask as mask_utils

# File paths
data_path = 'data/000HCDO8T1M72A9J0SDT.json'
image_path = 'data/000HCDO8T1M72A9J0SDT.jpg'

def decode_rle_mask(rle_data):
    """Decode RLE (Run Length Encoded) mask data"""
    if isinstance(rle_data, str):
        # If it's a string, it's already encoded
        return mask_utils.decode({'size': [1044, 1044], 'counts': rle_data})
    else:
        # If it's a dict with size and counts
        return mask_utils.decode(rle_data)

def visualize_annotations():
    """Visualize coral annotations overlaid on the image"""
    
    # Load the JSON data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Coral Segmentation Annotations Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Coral Image')
    axes[0, 0].axis('off')
    
    # Plot 2: All annotations overlaid
    axes[0, 1].imshow(image)
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    
    for i, annotation in enumerate(data['annotations']):
        # Decode the mask
        mask = decode_rle_mask(annotation['segmentation'])
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 0] = mask * 255  # Red channel
        colored_mask[:, :, 1] = mask * 128  # Green channel
        colored_mask[:, :, 2] = mask * 64   # Blue channel
        
        # Overlay the mask
        axes[0, 1].imshow(colored_mask, alpha=0.6)
        
        # Draw bounding box
        bbox = annotation['bbox']  # [x, y, width, height]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                               linewidth=2, edgecolor=colors[i % len(colors)], 
                               facecolor='none')
        axes[0, 1].add_patch(rect)
        
        # Add annotation info
        axes[0, 1].text(bbox[0], bbox[1]-10, f'Coral {i+1}\nArea: {annotation["area"]}\n(No Label)', 
                        color=colors[i % len(colors)], fontweight='bold', fontsize=8)
    
    axes[0, 1].set_title('All Annotations Overlaid')
    axes[0, 1].axis('off')
    
    # Plot 3: Individual coral masks
    for i, annotation in enumerate(data['annotations'][:3]):  # Show first 3 corals
        row = i // 2 + 1
        col = i % 2
        
        if row < 2:  # Only if we have space
            axes[row, col].imshow(image)
            
            # Decode and display mask
            mask = decode_rle_mask(annotation['segmentation'])
            colored_mask = np.zeros_like(image)
            colored_mask[:, :, 0] = mask * 255
            colored_mask[:, :, 1] = mask * 128
            colored_mask[:, :, 2] = mask * 64
            
            axes[row, col].imshow(colored_mask, alpha=0.7)
            axes[row, col].set_title(f'Coral {i+1} (Area: {annotation["area"]} px²)\nNo Class Label')
            axes[row, col].axis('off')
    
    # Hide unused subplots
    if len(data['annotations']) < 4:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis information
    print("\n" + "="*60)
    print("CORAL SEGMENTATION DATA ANALYSIS")
    print("="*60)
    
    print(f"\n📊 IMAGE INFORMATION:")
    print(f"   • Image dimensions: {data['image']['width']} x {data['image']['height']} pixels")
    print(f"   • Image file: {data['image']['file_name']}")
    
    print(f"\n🐠 CORAL ANNOTATIONS FOUND: {len(data['annotations'])}")
    total_area = 0
    for i, annotation in enumerate(data['annotations']):
        area = annotation['area']
        total_area += area
        bbox = annotation['bbox']
        print(f"   • Coral {i+1}: Area = {area:,} pixels ({area/(1044*1044)*100:.2f}% of image)")
        print(f"     Bounding box: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
    
    print(f"\n📈 STATISTICS:")
    print(f"   • Total coral coverage: {total_area:,} pixels ({total_area/(1044*1044)*100:.2f}% of image)")
    print(f"   • Average coral area: {total_area/len(data['annotations']):,.0f} pixels")
    print(f"   • Largest coral: {max(data['annotations'], key=lambda x: x['area'])['area']:,} pixels")
    print(f"   • Smallest coral: {min(data['annotations'], key=lambda x: x['area'])['area']:,} pixels")
    
    print(f"\n🔬 DATA FORMAT EXPLANATION:")
    print(f"   • This is COCO format segmentation data")
    print(f"   • Each annotation contains:")
    print(f"     - 'segmentation': RLE (Run Length Encoded) mask data")
    print(f"     - 'bbox': Bounding box coordinates [x, y, width, height]")
    print(f"     - 'area': Pixel area of the coral")
    print(f"     - 'id': Unique identifier for each coral")
    print(f"   ⚠️  IMPORTANT: NO CLASS LABELS!")
    print(f"     - No category_id or class_name fields")
    print(f"     - This is BINARY segmentation: coral vs non-coral")
    print(f"     - No distinction between coral types or health status")
    
    print(f"\n🚀 WHAT YOU CAN DO WITH THIS DATA:")
    print(f"   1. 🎯 BINARY SEGMENTATION:")
    print(f"      • Train models to distinguish coral vs non-coral")
    print(f"      • Perfect for: U-Net, DeepLab, FCN, PSPNet")
    print(f"      • Goal: Pixel-level coral vs background classification")
    
    print(f"\n   2. 📦 CORAL INSTANCE DETECTION:")
    print(f"      • Train YOLO for coral detection (class_id = 0 for all corals)")
    print(f"      • Convert to YOLO format: 0 center_x center_y width height")
    print(f"      • Goal: Detect and count coral instances")
    
    print(f"\n   3. 🎭 CORAL INSTANCE SEGMENTATION:")
    print(f"      • Train Mask R-CNN with single class (coral)")
    print(f"      • Perfect for this data format (masks + bboxes)")
    print(f"      • Goal: Separate individual coral colonies")
    
    print(f"\n   4. 📊 CORAL COVERAGE ANALYSIS:")
    print(f"      • Calculate coral coverage percentages")
    print(f"      • Monitor coral abundance over time")
    print(f"      • Analyze coral density and distribution")
    
    print(f"\n   5. 🔄 DATA AUGMENTATION:")
    print(f"      • Apply transforms to both images and masks")
    print(f"      • Rotate, flip, scale while maintaining mask accuracy")
    print(f"      • Increase dataset size for better model training")
    
    print(f"\n   ⚠️  LIMITATIONS:")
    print(f"      • Cannot train coral health classification (healthy vs bleached)")
    print(f"      • Cannot train coral species classification")
    print(f"      • Cannot train coral type classification (hard vs soft)")
    print(f"      • This is purely geometric segmentation data")
    
    print(f"\n💡 RECOMMENDED NEXT STEPS:")
    print(f"   1. Load more images from your dataset")
    print(f"   2. Convert to training format (YOLO, COCO, or custom)")
    print(f"   3. Split data into train/validation/test sets")
    print(f"   4. Choose model architecture based on your goal")
    print(f"   5. Implement data augmentation pipeline")
    
    return data, image

if __name__ == "__main__":
    data, image = visualize_annotations()