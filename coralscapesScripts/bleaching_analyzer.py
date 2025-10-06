import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def analyze_coral_bleaching(segmentation_mask_path, output_dir="bleaching_analysis"):
    """
    Analyze coral bleaching status from segmentation mask
    
    Args:
        segmentation_mask_path: Path to the segmentation mask PNG file
        output_dir: Directory to save analysis results
    """
    
    # Load the segmentation mask
    mask = np.array(Image.open(segmentation_mask_path))
    
    # Define coral classes and their bleaching status
    coral_classes = {
        # Alive corals
        'branching_alive': 22,
        'massive_meandering_alive': 17, 
        'acropora_alive': 25,
        'table_acropora_alive': 28,
        'pocillopora_alive': 31,
        'stylophora_alive': 34,
        'meandering_alive': 36,
        
        # Bleached corals
        'branching_bleached': 19,
        'massive_meandering_bleached': 16,
        'other_coral_bleached': 4,
        'meandering_bleached': 33,
        
        # Dead corals
        'branching_dead': 20,
        'massive_meandering_dead': 23,
        'other_coral_dead': 3,
        'meandering_dead': 37,
        'table_acropora_dead': 32,
    }
    
    # Calculate pixel counts for each category
    total_pixels = mask.size
    coral_pixels = 0
    alive_pixels = 0
    bleached_pixels = 0
    dead_pixels = 0
    
    # Count pixels for each coral type
    coral_counts = {}
    for coral_type, class_id in coral_classes.items():
        count = np.sum(mask == class_id)
        coral_counts[coral_type] = count
        
        if 'alive' in coral_type:
            alive_pixels += count
        elif 'bleached' in coral_type:
            bleached_pixels += count
        elif 'dead' in coral_type:
            dead_pixels += count
            
        coral_pixels += count
    
    # Calculate percentages
    coral_percentage = (coral_pixels / total_pixels) * 100
    alive_percentage = (alive_pixels / coral_pixels) * 100 if coral_pixels > 0 else 0
    bleached_percentage = (bleached_pixels / coral_pixels) * 100 if coral_pixels > 0 else 0
    dead_percentage = (dead_pixels / coral_pixels) * 100 if coral_pixels > 0 else 0
    
    # Create analysis report
    report = f"""
CORAL BLEACHING ANALYSIS REPORT
==============================

Image: {os.path.basename(segmentation_mask_path)}
Total pixels: {total_pixels:,}
Coral pixels: {coral_pixels:,} ({coral_percentage:.2f}% of image)

BLEACHING STATUS:
- Alive corals: {alive_pixels:,} pixels ({alive_percentage:.2f}% of corals)
- Bleached corals: {bleached_pixels:,} pixels ({bleached_percentage:.2f}% of corals)
- Dead corals: {dead_pixels:,} pixels ({dead_percentage:.2f}% of corals)

DETAILED CORAL BREAKDOWN:
"""
    
    for coral_type, count in coral_counts.items():
        if count > 0:
            percentage = (count / coral_pixels) * 100 if coral_pixels > 0 else 0
            report += f"- {coral_type}: {count:,} pixels ({percentage:.2f}% of corals)\n"
    
    # Bleaching assessment
    if bleached_percentage > 30:
        bleaching_status = "HIGH BLEACHING RISK"
        risk_color = "🔴"
    elif bleached_percentage > 10:
        bleaching_status = "MODERATE BLEACHING"
        risk_color = "🟡"
    else:
        bleaching_status = "LOW BLEACHING"
        risk_color = "🟢"
    
    report += f"""
BLEACHING ASSESSMENT: {risk_color} {bleaching_status}

Health Score: {alive_percentage:.1f}/100
Bleaching Score: {bleached_percentage:.1f}/100
"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(segmentation_mask_path))[0]
    
    # Print summary to console
    print(f"\nCORAL BLEACHING SUMMARY:")
    print(f"Image: {os.path.basename(segmentation_mask_path)}")
    print(f"Coral coverage: {coral_percentage:.1f}% of image")
    print(f"Unbleached corals: {alive_percentage:.1f}% of corals")
    print(f"Bleached corals: {bleached_percentage:.1f}% of corals")
    print(f"Assessment: {bleaching_status}")
    
    # Create visualization
    create_bleaching_visualization(mask, coral_classes, output_dir, base_name)
    
    return {
        'coral_percentage': coral_percentage,
        'alive_percentage': alive_percentage,
        'bleached_percentage': bleached_percentage,
        'dead_percentage': dead_percentage,
        'bleaching_status': bleaching_status,
        'coral_counts': coral_counts
    }

def create_bleaching_visualization(mask, coral_classes, output_dir, base_name):
    """Create a visualization showing bleaching status overlaid on original image"""
    
    # Try to find the original image
    original_image_path = None
    possible_paths = [
        f"../coralscapes/images/{base_name.replace('_pred', '')}.jpg",
        f"../coralscapes/images/{base_name.replace('_pred', '')}.JPG",
        f"../inference_results/{base_name.replace('_pred', '')}_overlay.png"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            original_image_path = path
            break
    
    if original_image_path:
        # Load original image
        original_img = Image.open(original_image_path)
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        
        # Resize original image to match mask size
        original_img = original_img.resize((mask.shape[1], mask.shape[0]))
        original_array = np.array(original_img)
    else:
        # If no original image found, create a black background
        original_array = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Create color-coded mask for bleaching status
    bleaching_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Color coding: Green=Unbleached (alive), Red=Bleached, Transparent=Background
    for coral_type, class_id in coral_classes.items():
        if 'alive' in coral_type:
            bleaching_mask[mask == class_id] = [0, 255, 0]  # Green (unbleached)
        elif 'bleached' in coral_type:
            bleaching_mask[mask == class_id] = [255, 0, 0]  # Red (bleached)
        # Skip dead corals - we only want alive vs bleached
    
    # Create overlay by blending original image with bleaching mask
    # Only show colors where there are alive or bleached corals
    coral_mask = np.zeros(mask.shape, dtype=bool)
    for coral_type, class_id in coral_classes.items():
        if 'alive' in coral_type or 'bleached' in coral_type:
            coral_mask |= (mask == class_id)
    
    # Create overlay
    overlay = original_array.copy()
    overlay[coral_mask] = 0.7 * original_array[coral_mask] + 0.3 * bleaching_mask[coral_mask]
    
    # Create the visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    
    # Original image
    ax1.imshow(original_array)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Bleaching overlay
    ax2.imshow(overlay.astype(np.uint8))
    ax2.set_title('Bleaching Analysis Overlay')
    ax2.axis('off')
    
    # Legend
    ax3.axis('off')
    colors = ['green', 'red']
    labels = ['Unbleached Corals', 'Bleached Corals']
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=labels[i]) for i in range(len(colors))]
    ax3.legend(handles=legend_elements, loc='center', fontsize=12)
    ax3.set_title('Legend')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_bleaching_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {os.path.join(output_dir, f'{base_name}_bleaching_analysis.png')}")

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        mask_path = sys.argv[1]
        analyze_coral_bleaching(mask_path)
    else:
        print("Usage: python bleaching_analyzer.py <path_to_segmentation_mask.png>")
