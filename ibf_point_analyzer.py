#!/usr/bin/env python3
"""
IBF Point Label Analyzer - Coral Point Count (CPC) format
CPE4/CPCE software point count data with random points on coral images
Usage: Classification/point-based coverage estimation
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

BASE_DIR = Path("data/point_labels/content/gdrive/MyDrive/Data Challenge 3 - JBG060 AY2526/01_data/benthic_datasets/point_labels/IBF/GA 1")

def parse_cpc_file(cpc_path):
    with open(cpc_path, 'r') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split(',')
    image_name = Path(header[1].strip('"')).name
    img_w, img_h = int(header[2]), int(header[3])
    
    num_points = int(lines[5].strip())
    points = []
    for i in range(6, 6 + num_points):
        x, y = map(int, lines[i].strip().split(','))
        points.append((x, y))
    
    return {
        'image_name': image_name,
        'image_size': (img_w, img_h),
        'points': points,
        'num_points': num_points
    }

def load_ibf_data():
    cpc_files = list(BASE_DIR.glob("*.cpc"))
    data = []
    for cpc in cpc_files[:3]:
        try:
            parsed = parse_cpc_file(cpc)
            img_path = BASE_DIR / parsed['image_name']
            if img_path.exists():
                parsed['image_path'] = img_path
                data.append(parsed)
        except:
            continue
    return data

def plot_ibf_analysis():
    data = load_ibf_data()
    if not data:
        print("No IBF data found")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Plot images with points
    for i, item in enumerate(data):
        img = cv2.imread(str(item['image_path']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[0, i].imshow(img)
        
        # Plot random points
        for x, y in item['points']:
            axes[0, i].plot(x, y, 'ro', markersize=3, alpha=0.7)
        
        axes[0, i].set_title(f"{item['image_name'][:15]}...\n{item['num_points']} points")
        axes[0, i].axis('off')
    
    # Statistics
    point_counts = [item['num_points'] for item in data]
    
    axes[1, 0].bar(range(len(point_counts)), point_counts, color='coral')
    axes[1, 0].set_title('Points per Image')
    axes[1, 0].set_ylabel('Count')
    
    axes[1, 1].hist([p for item in data for p in [pt[0] for pt in item['points']]], 
                   bins=20, alpha=0.7, color='blue', label='X coords')
    axes[1, 1].hist([p for item in data for p in [pt[1] for pt in item['points']]], 
                   bins=20, alpha=0.7, color='red', label='Y coords')
    axes[1, 1].set_title('Point Distribution')
    axes[1, 1].legend()
    
    # Image size distribution
    sizes = [item['image_size'] for item in data]
    widths, heights = zip(*sizes)
    axes[1, 2].scatter(widths, heights, c='green', alpha=0.7)
    axes[1, 2].set_title('Image Dimensions')
    axes[1, 2].set_xlabel('Width')
    axes[1, 2].set_ylabel('Height')
    
    plt.suptitle('IBF Dataset: Random Point Sampling for Coverage Estimation\nUSE CASE: Classification (point-level species ID)')
    plt.tight_layout()
    plt.savefig('ibf_analysis.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"IBF Analysis: {len(data)} images with random point annotations")
    print(f"Average points per image: {np.mean(point_counts):.1f}")
    print("APPLICATION: Point-based classification for benthic coverage estimation")
    print("ML TASK: Multi-class classification (species at point locations)")

if __name__ == "__main__":
    plot_ibf_analysis()
