#!/usr/bin/env python3
"""
SEAVIEW Point Label Analyzer - Global reef survey with functional groups
Large-scale geographic dataset with standardized point sampling
Usage: Classification/coverage estimation across multiple regions
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import Counter

BASE_DIR = Path("data/point_labels/content/gdrive/MyDrive/Data Challenge 3 - JBG060 AY2526/01_data\benthic_datasets/point_labels/SEAVIEW")

def load_seaview_data():
    # Load a representative annotation file
    annotations = pd.read_csv(BASE_DIR / "tabular-data" / "annotations_ATL.csv")
    
    # Sample data for visualization
    sample_quadrats = annotations['quadratid'].unique()[:3]
    sample_data = annotations[annotations['quadratid'].isin(sample_quadrats)]
    
    return sample_data

def plot_seaview_analysis():
    annotations = load_seaview_data()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Plot point distributions for sample quadrats
    unique_quadrats = annotations['quadratid'].unique()[:3]
    
    for i, quadrat_id in enumerate(unique_quadrats):
        quadrat_data = annotations[annotations['quadratid'] == quadrat_id]
        
        # Create scatter plot of points
        func_groups = quadrat_data['func_group'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(func_groups)))
        
        for j, group in enumerate(func_groups):
            group_data = quadrat_data[quadrat_data['func_group'] == group]
            axes[0, i].scatter(group_data['x'], group_data['y'], 
                             c=[colors[j]], label=group, s=20, alpha=0.7)
        
        axes[0, i].set_title(f"Quadrat {quadrat_id}\n{len(quadrat_data)} points")
        axes[0, i].legend(fontsize=6)
        axes[0, i].set_xlim(0, 1000)
        axes[0, i].set_ylim(0, 800)
        axes[0, i].invert_yaxis()
        axes[0, i].set_aspect('equal')
    
    # Functional group distribution
    func_counts = annotations['func_group'].value_counts()
    axes[1, 0].pie(func_counts.values, labels=func_counts.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Functional Group Distribution')
    
    # Label distribution
    label_counts = annotations['label_name'].value_counts().head(8)
    axes[1, 1].barh(range(len(label_counts)), label_counts.values, color='lightcoral')
    axes[1, 1].set_yticks(range(len(label_counts)))
    axes[1, 1].set_yticklabels(label_counts.index, fontsize=8)
    axes[1, 1].set_title('Top Species/Types')
    axes[1, 1].set_xlabel('Count')
    
    # Points per quadrat
    points_per_quadrat = annotations.groupby('quadratid').size()
    axes[1, 2].hist(points_per_quadrat, bins=15, color='lightgreen', alpha=0.7)
    axes[1, 2].set_title('Points per Quadrat')
    axes[1, 2].set_xlabel('Number of Points')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.suptitle('SEAVIEW Dataset: Global Reef Survey with Functional Groups\nUSE CASE: Large-scale reef monitoring & classification')
    plt.tight_layout()
    plt.savefig('seaview_analysis.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # Print regional info
    regions = [d.name for d in (BASE_DIR / "tabular-data").glob("annotations_*.csv")]
    print(f"SEAVIEW Analysis: {len(annotations)} points from Atlantic region sample")
    print(f"Available regions: {len(regions)} (ATL, IND_CHA, IND_MDV, PAC_*)")
    print(f"Functional groups: {annotations['func_group'].nunique()}")
    print(f"Species/types: {annotations['label_name'].nunique()}")
    print("APPLICATION: Global reef coverage estimation & monitoring")
    print("ML TASK: Multi-class classification with hierarchical labels (functional groups + species)")

if __name__ == "__main__":
    plot_seaview_analysis()
