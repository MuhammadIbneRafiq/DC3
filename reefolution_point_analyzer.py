#!/usr/bin/env python3
"""
Reefolution Point Label Analyzer - CSV annotations with coral species labels
Row/Column coordinates on cropped coral images
Usage: Multi-class coral species classification
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import Counter

BASE_DIR = Path("data/point_labels/content/gdrive/MyDrive/Data Challenge 3 - JBG060 AY2526/01_data/benthic_datasets/point_labels/Reefolution")

def load_reefolution_data():
    annotations = pd.read_csv(BASE_DIR / "annotations.csv")
    labelset = pd.read_csv(BASE_DIR / "labelset.csv")
    
    # Get top 3 images with most annotations
    top_images = annotations['Name'].value_counts().head(3).index
    sample_data = annotations[annotations['Name'].isin(top_images)]
    
    return sample_data, labelset

def plot_reefolution_analysis():
    annotations, labelset = load_reefolution_data()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Plot sample images with points
    unique_images = annotations['Name'].unique()[:3]
    
    for i, img_name in enumerate(unique_images):
        img_path = BASE_DIR / "ALL CROPS COMBINED" / img_name
        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_annotations = annotations[annotations['Name'] == img_name]
            
            axes[0, i].imshow(img)
            
            # Plot points colored by label
            labels = img_annotations['Label'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            for j, label in enumerate(labels):
                label_points = img_annotations[img_annotations['Label'] == label]
                axes[0, i].scatter(label_points['Column'], label_points['Row'], 
                                 c=[colors[j]], label=label, s=15, alpha=0.8)
            
            axes[0, i].set_title(f"{img_name[:15]}...\n{len(img_annotations)} points")
            axes[0, i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
            axes[0, i].axis('off')
    
    # Label distribution
    label_counts = annotations['Label'].value_counts().head(10)
    axes[1, 0].barh(range(len(label_counts)), label_counts.values, color='coral')
    axes[1, 0].set_yticks(range(len(label_counts)))
    axes[1, 0].set_yticklabels(label_counts.index, fontsize=8)
    axes[1, 0].set_title('Top 10 Species Labels')
    axes[1, 0].set_xlabel('Count')
    
    # Points per image distribution
    points_per_image = annotations.groupby('Name').size()
    axes[1, 1].hist(points_per_image, bins=20, color='lightblue', alpha=0.7)
    axes[1, 1].set_title('Points per Image Distribution')
    axes[1, 1].set_xlabel('Number of Points')
    axes[1, 1].set_ylabel('Frequency')
    
    # Spatial distribution of all points
    axes[1, 2].scatter(annotations['Column'], annotations['Row'], alpha=0.5, s=1, c='green')
    axes[1, 2].set_title('Point Spatial Distribution')
    axes[1, 2].set_xlabel('Column (X)')
    axes[1, 2].set_ylabel('Row (Y)')
    axes[1, 2].invert_yaxis()
    
    plt.suptitle('Reefolution Dataset: Species-level Point Annotations\nUSE CASE: Multi-class coral species classification')
    plt.tight_layout()
    plt.savefig('reefolution_analysis.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"Reefolution Analysis: {len(annotations)} total point annotations")
    print(f"Images: {annotations['Name'].nunique()}, Species: {annotations['Label'].nunique()}")
    print(f"Avg points per image: {len(annotations) / annotations['Name'].nunique():.1f}")
    print("APPLICATION: Fine-grained coral species classification")
    print("ML TASK: Multi-class classification with species-level labels")

if __name__ == "__main__":
    plot_reefolution_analysis()
