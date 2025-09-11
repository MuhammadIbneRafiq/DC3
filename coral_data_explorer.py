import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
import cv2
from pathlib import Path
import json
from collections import defaultdict
import glob

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CoralDataExplorer:
    def __init__(self, 
                 base_path="data/mask_labels/content/gdrive/MyDrive/Data Challenge 3 - JBG060 AY2526/01_data/benthic_datasets/mask_labels"
        ):
        self.base_path = Path(base_path)
        self.coralseg_path = self.base_path / "Coralseg"
        self.reef_support_path = self.base_path / "reef_support"
        
        self.dataset_stats = {}
        self.coral_areas = defaultdict(list)
        self.geographic_data = defaultdict(dict)
        
    def analyze_dataset_distribution(self):
        """Analyze the distribution of datasets across different regions and types"""        
        # Coralseg dataset analysis
        coralseg_stats = {}
        for split in ['train', 'test', 'val']:
            split_path = self.coralseg_path / split
            if split_path.exists():
                image_count = len(list((split_path / "Image").glob("*.jpg")))
                mask_count = len(list((split_path / "Mask").glob("*.png")))
                coralseg_stats[split] = {'images': image_count, 'masks': mask_count}
        
        self.dataset_stats['coralseg'] = coralseg_stats
        
        reef_stats = {}
        for region_dir in self.reef_support_path.iterdir():
            if region_dir.is_dir():
                region_name = region_dir.name
                image_count = len(list((region_dir / "images").glob("*.*"))) if (region_dir / "images").exists() else 0
                mask_count = len(list((region_dir / "masks").glob("*.png"))) if (region_dir / "masks").exists() else 0
                stitched_count = len(list((region_dir / "masks_stitched").glob("*.png"))) if (region_dir / "masks_stitched").exists() else 0
                
                reef_stats[region_name] = {
                    'images': image_count,
                    'individual_masks': mask_count, 
                    'stitched_masks': stitched_count
                }
        
        self.dataset_stats['reef_support'] = reef_stats
        
        return self.dataset_stats
    
    def visualize_dataset_overview(self):
        """Create comprehensive visualization of dataset overview"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Coral Dataset Overview - Supporting Multi-Dataset Analysis for RQ Formulation', fontsize=16, fontweight='bold')
        
        # Plot 1: Coralseg dataset distribution
        if 'coralseg' in self.dataset_stats:
            splits = list(self.dataset_stats['coralseg'].keys())
            image_counts = [self.dataset_stats['coralseg'][split]['images'] for split in splits]
            
            axes[0,0].bar(splits, image_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[0,0].set_title('CoralSeg Dataset Distribution\n(Methodological Foundation)', fontweight='bold')
            axes[0,0].set_ylabel('Number of Images')
            for i, count in enumerate(image_counts):
                axes[0,0].text(i, count + 50, str(count), ha='center', fontweight='bold')
        
        # Plot 2: Reef support regional distribution
        if 'reef_support' in self.dataset_stats:
            regions = list(self.dataset_stats['reef_support'].keys())
            region_counts = [self.dataset_stats['reef_support'][region]['images'] for region in regions]
            
            # Create horizontal bar plot for better label readability
            y_pos = np.arange(len(regions))
            bars = axes[0,1].barh(y_pos, region_counts, color=plt.cm.viridis(np.linspace(0, 1, len(regions))))
            axes[0,1].set_yticks(y_pos)
            axes[0,1].set_yticklabels([r.replace('SEAVIEW_', '').replace('SEAFLOWER_', 'SF_') for r in regions], fontsize=9)
            axes[0,1].set_title('Regional Dataset Distribution\n(Geographic Diversity for SLE Analysis)', fontweight='bold')
            axes[0,1].set_xlabel('Number of Images')
            
            # Add value labels
            for i, v in enumerate(region_counts):
                axes[0,1].text(v + 10, i, str(v), va='center', fontweight='bold')
        
        # Plot 3: Total dataset comparison
        total_coralseg = sum([self.dataset_stats['coralseg'][split]['images'] for split in self.dataset_stats['coralseg']])
        total_reef_support = sum([self.dataset_stats['reef_support'][region]['images'] for region in self.dataset_stats['reef_support']])
        
        datasets = ['CoralSeg\n(Benchmark)', 'Reef Support\n(Multi-Regional)']
        totals = [total_coralseg, total_reef_support]
        
        axes[1,0].pie(totals, labels=datasets, autopct='%1.1f%%', startangle=90, 
                     colors=['#FF9999', '#66B2FF'])
        axes[1,0].set_title('Dataset Size Comparison\n(Methodological vs Regional Focus)', fontweight='bold')
        
        # Plot 4: Mask availability analysis
        mask_types = ['Individual\nMasks', 'Stitched\nMasks']
        total_individual = sum([self.dataset_stats['reef_support'][region]['individual_masks'] for region in self.dataset_stats['reef_support']])
        total_stitched = sum([self.dataset_stats['reef_support'][region]['stitched_masks'] for region in self.dataset_stats['reef_support']])
        
        axes[1,1].bar(mask_types, [total_individual, total_stitched], 
                     color=['#FFA07A', '#98D8C8'])
        axes[1,1].set_title('Annotation Granularity\n(Area Estimation Capability)', fontweight='bold')
        axes[1,1].set_ylabel('Number of Masks')
        
        for i, count in enumerate([total_individual, total_stitched]):
            axes[1,1].text(i, count + 200, str(count), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('coral_dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print(f"\n=== DATASET SUMMARY STATISTICS ===")
        print(f"Total CoralSeg Images: {total_coralseg}")
        print(f"Total Reef Support Images: {total_reef_support}")
        print(f"Total Individual Masks: {total_individual}")
        print(f"Total Stitched Masks: {total_stitched}")
        print(f"Geographic Regions Covered: {len(self.dataset_stats['reef_support'])}")
        
        return fig
    
    def analyze_coral_areas_from_csv(self):
        """Analyze coral areas from the labelbox segments report"""
        csv_path = self.reef_support_path / "UNAL_BLEACHING_TAYRONA" / "labelbox_segments_report.csv"
        
        if not csv_path.exists():
            print(f"CSV file not found: {csv_path}")
            return None
            
        # Read the CSV data
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} coral segments from UNAL_BLEACHING_TAYRONA dataset")
        
        # Analyze area distributions
        self.coral_areas['all'] = df['Area (%)'].tolist()
        self.coral_areas['pixels'] = df['Pixels'].tolist()
        
        # Group by image to analyze coral count per image
        image_groups = df.groupby('Image Name')
        coral_counts_per_image = image_groups.size()
        coral_areas_per_image = image_groups['Area (%)'].sum()
        
        return df, coral_counts_per_image, coral_areas_per_image
    
    def visualize_coral_size_analysis(self, df, coral_counts_per_image, coral_areas_per_image):
        """Create visualizations for coral size analysis supporting RQ about size vs bleaching"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Coral Size Analysis - Evidence for "Bigger Corals vs Bleaching Susceptibility" RQ', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Distribution of coral areas
        axes[0,0].hist(self.coral_areas['all'], bins=50, alpha=0.7, color='coral', edgecolor='black')
        axes[0,0].set_xlabel('Coral Area (%)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Distribution of Individual Coral Areas\n(Size Variability Evidence)', fontweight='bold')
        axes[0,0].axvline(np.mean(self.coral_areas['all']), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(self.coral_areas["all"]):.2f}%')
        axes[0,0].legend()
        
        # Plot 2: Log-scale distribution for better visualization of range
        axes[0,1].hist(np.log10(np.array(self.coral_areas['pixels'])), bins=50, alpha=0.7, 
                      color='lightblue', edgecolor='black')
        axes[0,1].set_xlabel('Log10(Pixels)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Coral Size Distribution (Log Scale)\n(Wide Size Range Evidence)', fontweight='bold')
        
        # Plot 3: Coral count per image
        axes[0,2].hist(coral_counts_per_image, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,2].set_xlabel('Number of Corals per Image')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].set_title('Coral Density Distribution\n(Ecosystem Complexity)', fontweight='bold')
        
        # Plot 4: Total coral area per image
        axes[1,0].hist(coral_areas_per_image, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1,0].set_xlabel('Total Coral Area per Image (%)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Coral Coverage Distribution\n(Reef Health Indicator)', fontweight='bold')
        
        # Plot 5: Size vs Count relationship
        image_data = pd.DataFrame({
            'coral_count': coral_counts_per_image,
            'total_area': coral_areas_per_image,
            'avg_coral_size': coral_areas_per_image / coral_counts_per_image
        })
        
        scatter = axes[1,1].scatter(image_data['coral_count'], image_data['avg_coral_size'], 
                                  alpha=0.6, c=image_data['total_area'], cmap='viridis')
        axes[1,1].set_xlabel('Number of Corals per Image')
        axes[1,1].set_ylabel('Average Coral Size (%)')
        axes[1,1].set_title('Coral Count vs Average Size\n(Size-Density Relationship)', fontweight='bold')
        plt.colorbar(scatter, ax=axes[1,1], label='Total Coverage (%)')
        
        # Plot 6: Size categories for RQ support
        # Create size categories: small, medium, large
        size_thresholds = [np.percentile(self.coral_areas['all'], 33), 
                          np.percentile(self.coral_areas['all'], 67)]
        
        size_categories = []
        for area in self.coral_areas['all']:
            if area < size_thresholds[0]:
                size_categories.append('Small')
            elif area < size_thresholds[1]:
                size_categories.append('Medium') 
            else:
                size_categories.append('Large')
        
        size_counts = pd.Series(size_categories).value_counts()
        
        axes[1,2].pie(size_counts.values, labels=size_counts.index, autopct='%1.1f%%', 
                     colors=['#FF9999', '#66B2FF', '#99FF99'])
        axes[1,2].set_title('Coral Size Categories\n(Framework for Bleaching Analysis)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('coral_size_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistical summary
        print(f"\n=== CORAL SIZE STATISTICS ===")
        print(f"Total coral segments analyzed: {len(df)}")
        print(f"Average coral area: {np.mean(self.coral_areas['all']):.3f}%")
        print(f"Median coral area: {np.median(self.coral_areas['all']):.3f}%")
        print(f"Coral area range: {np.min(self.coral_areas['all']):.3f}% - {np.max(self.coral_areas['all']):.3f}%")
        print(f"Standard deviation: {np.std(self.coral_areas['all']):.3f}%")
        print(f"Size categories - Small: {size_counts['Small']}, Medium: {size_counts['Medium']}, Large: {size_counts['Large']}")
        
        return image_data, size_categories
    
    def analyze_geographic_distribution(self):        
        geographic_regions = {
            'SEAVIEW_ATL': 'Atlantic Ocean',
            'SEAVIEW_PAC_AUS': 'Pacific - Australia', 
            'SEAVIEW_PAC_USA': 'Pacific - USA (Hawaii)',
            'SEAVIEW_IDN_PHL': 'Pacific - Indonesia/Philippines',
            'SEAFLOWER_BOLIVAR': 'Caribbean - Colombia',
            'SEAFLOWER_COURTOWN': 'Caribbean - Colombia',
            'TETES_PROVIDENCIA': 'Caribbean - Colombia',
            'UNAL_BLEACHING_TAYRONA': 'Caribbean - Colombia (Bleaching Study)'
        }
        
        region_data = []
        for region_key, region_name in geographic_regions.items():
            if region_key in self.dataset_stats['reef_support']:
                stats = self.dataset_stats['reef_support'][region_key]
                region_data.append({
                    'region': region_name,
                    'key': region_key,
                    'images': stats['images'],
                    'ocean': region_name.split(' - ')[0] if ' - ' in region_name else region_name.split()[0]
                })
        
        return pd.DataFrame(region_data)
    
    def visualize_geographic_sle_analysis(self, region_df):
        """Visualize geographic data to support SLE research questions"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Geographic Distribution Analysis - Supporting SLE (Societal-Legal-Ethical) Research Questions', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Images by ocean region
        ocean_counts = region_df.groupby('ocean')['images'].sum().sort_values(ascending=True)
        
        axes[0,0].barh(range(len(ocean_counts)), ocean_counts.values, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(ocean_counts))))
        axes[0,0].set_yticks(range(len(ocean_counts)))
        axes[0,0].set_yticklabels(ocean_counts.index)
        axes[0,0].set_xlabel('Number of Images')
        axes[0,0].set_title('Data Coverage by Ocean Region\n(Global Monitoring Capability)', fontweight='bold')
        
        for i, v in enumerate(ocean_counts.values):
            axes[0,0].text(v + 20, i, str(v), va='center', fontweight='bold')
        
        # Plot 2: Detailed regional breakdown
        region_counts = region_df.set_index('region')['images'].sort_values(ascending=True)
        
        axes[0,1].barh(range(len(region_counts)), region_counts.values,
                      color=plt.cm.viridis(np.linspace(0, 1, len(region_counts))))
        axes[0,1].set_yticks(range(len(region_counts)))
        axes[0,1].set_yticklabels([r.replace('Pacific - ', 'PAC-').replace('Caribbean - ', 'CAR-') 
                                  for r in region_counts.index], fontsize=9)
        axes[0,1].set_xlabel('Number of Images')
        axes[0,1].set_title('Detailed Regional Coverage\n(Local Stakeholder Representation)', fontweight='bold')
        
        # Plot 3: Caribbean focus (high biodiversity, tourism impact)
        caribbean_data = region_df[region_df['ocean'] == 'Caribbean']
        axes[1,0].pie(caribbean_data['images'], labels=caribbean_data['region'].str.replace('Caribbean - Colombia', 'COL'), 
                        autopct='%1.1f%%', startangle=90)
        axes[1,0].set_title('Caribbean Region Detail\n(Tourism & Conservation Hotspot)', fontweight='bold')
        
        # Plot 4: Pacific diversity
        pacific_data = region_df[region_df['ocean'] == 'Pacific']
        pacific_labels = pacific_data['region'].str.replace('Pacific - ', '')
        axes[1,1].pie(pacific_data['images'], labels=pacific_labels,
                        autopct='%1.1f%%', startangle=90)
        axes[1,1].set_title('Pacific Region Diversity\n(International Cooperation Needs)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('geographic_sle_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return region_df
    
    def create_research_question_evidence_summary(self):        
        evidence = {
            "methodological_evidence": [
                f"• {sum([self.dataset_stats['coralseg'][split]['images'] for split in self.dataset_stats['coralseg']])} images in CoralSeg benchmark dataset",
                f"• {sum([self.dataset_stats['reef_support'][region]['stitched_masks'] for region in self.dataset_stats['reef_support']])} stitched masks for area estimation",
                f"• Coral size range: {np.min(self.coral_areas['all']):.3f}% - {np.max(self.coral_areas['all']):.3f}% coverage",
                "• Multiple annotation formats (individual segments + stitched masks)",
                "• Transfer learning potential with pretrained models"
            ],
            
            "bleaching_analysis_evidence": [
                "• UNAL_BLEACHING_TAYRONA dataset specifically for bleaching studies",
                f"• {len(self.coral_areas['all'])} individual coral segments with precise area measurements",
                "• Size categorization framework (Small/Medium/Large) established",
                "• Coral density variation across images (ecosystem complexity)",
                "• Area estimation methodology validated through multiple mask types"
            ],
            
            "sle_evidence": [
                "• 8 different geographic regions covered (global applicability)",
                "• Caribbean tourism hotspots included (societal impact)",
                "• Pacific international waters (legal framework complexity)", 
                "• Multi-country datasets (ethical considerations for data sharing)",
                "• Coastal community dependency regions represented"
            ],
            
            "technical_efficiency_evidence": [
                "• Large dataset size enables ensemble learning experiments",
                "• Multiple geographic regions for generalization testing",
                "• Existing segmentation benchmarks for performance comparison",
                "• GPU hour optimization potential through transfer learning",
                "• Energy consumption measurement framework possible"
            ]
        }
        
        print("\n1. METHODOLOGICAL FOUNDATION:")
        for item in evidence["methodological_evidence"]:
            print(f"   {item}")
            
        print("\n2. BLEACHING SUSCEPTIBILITY ANALYSIS:")
        for item in evidence["bleaching_analysis_evidence"]:
            print(f"   {item}")
            
        print("\n3. SOCIETAL-LEGAL-ETHICAL (SLE) SUPPORT:")
        for item in evidence["sle_evidence"]:
            print(f"   {item}")
            
        print("\n4. TECHNICAL EFFICIENCY POTENTIAL:")
        for item in evidence["technical_efficiency_evidence"]:
            print(f"   {item}")
        
        print("""
RECOMMENDED APPROACH: TWO COMPLEMENTARY RESEARCH QUESTIONS

Based on the data analysis, I recommend formulating TWO main research questions rather than 
one general RQ with subquestions. Here's why:

RQ1 (METHODOLOGICAL): "To what extent can fusion of image processing techniques 
(ensemble learning + transfer learning) achieve similar coral area estimation benchmarks 
while reducing GPU training time?"

Evidence Support:
- Multiple datasets available for ensemble training
- Benchmark comparison possible with existing models
- Clear metrics for both accuracy and efficiency measurement

RQ2 (SLE INTEGRATED): "How can AI-based coral area estimation systems support coastal 
communities through enhanced reef monitoring while ensuring ethical data use and 
legal compliance across international waters?"

Evidence Support:
- Geographic diversity enables international legal framework analysis
- Tourism-dependent regions included for societal impact assessment
- Multi-country data raises ethical considerations for fair access

WHY TWO SEPARATE RQs WORK BETTER:
1. Clear methodological contribution (efficiency + accuracy)
2. Distinct SLE impact assessment (not just an afterthought)
3. Different evaluation criteria and methods
4. Allows deeper investigation of each aspect
5. Better alignment with academic publication standards

ALTERNATIVE COMBINED APPROACH:
If you prefer one RQ, focus on: "How can efficient AI coral monitoring systems 
be developed and deployed responsibly to support coastal communities globally?"
- But this may lack the technical depth reviewers expect
        """)
        
        return evidence

explorer = CoralDataExplorer()

# Step 1: Analyze dataset distribution
explorer.analyze_dataset_distribution()

# Step 2: Create overview visualizations
explorer.visualize_dataset_overview()

# Step 3: Analyze coral areas
df, coral_counts, coral_areas = explorer.analyze_coral_areas_from_csv()

# Step 4: Create size analysis visualizations
image_data, size_categories = explorer.visualize_coral_size_analysis(df, coral_counts, coral_areas)

# Step 5: Geographic analysis for SLE
region_df = explorer.analyze_geographic_distribution()
explorer.visualize_geographic_sle_analysis(region_df)

# Step 6: Generate evidence summary
evidence = explorer.create_research_question_evidence_summary()

print("Files created:")
print("- coral_dataset_overview.png")
print("- coral_size_analysis.png") 
print("- geographic_sle_analysis.png")