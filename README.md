# Group 3 - Coral Bleaching Detection
## JBG060 Capstone Data Challenge 2025-2026 Q1

This repository contains the code for coral bleaching detection using deep learning models. The project implements a comprehensive fine-tuning pipeline for semantic segmentation of coral reef images to identify bleached and non-bleached coral regions.

## 🚀 Quick Start

### Environment Setup

#### For GPU-equipped devices (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd DC3

# Install dependencies
pip install -r requirements.txt

# For coral reef fine-tuning (additional dependencies)
cd coralscapesScripts
pip install -e .
```

#### For CPU-only devices
```bash
# Install basic dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 🧠 Fine-tuning Pipeline

### Overview
The fine-tuning pipeline is designed to train deep learning models for coral bleaching detection using semantic segmentation. It supports multiple model architectures and includes cross-validation for robust evaluation.

### Supported Model Architectures
- **DPT-DINOv2-Giant** (with LoRA fine-tuning)
- **DPT-DINOv2-Base** (with LoRA fine-tuning)
- **SegFormer-MiT-b2/b5** (with LoRA fine-tuning)
- **DeepLabV3+ with ResNet50**
- **U-Net with ResNet50**

### Dataset Requirements
The fine-tuning pipeline expects the following dataset structure:
```
data/
├── images/                    # Original RGB coral reef images (.jpg)
├── masks_bleached/           # Binary masks for bleached coral regions (.png)
└── masks_non_bleached/       # Binary masks for non-bleached coral regions (.png)
```

### Running Fine-tuning

#### Basic Usage
```bash
# Navigate to the coralscapesScripts directory
cd coralscapesScripts

# Run fine-tuning with default settings
python run_fine_tuning.py --config configs/dpt-dinov2-giant_lora.yaml --dataset-dir ../data --epochs 50 --batch-size 2
```

#### Advanced Configuration
```bash
# Custom configuration with specific parameters
python fine_tune_pipeline.py \
    --config configs/dpt-dinov2-giant_lora.yaml \
    --dataset-dir ../data \
    --n-folds 5 \
    --run-name "coral_bleaching_experiment" \
    --batch-size 2 \
    --batch-size-eval 1 \
    --epochs 100 \
    --lr 0.00005 \
    --device cuda:0
```

#### Command Line Arguments
- `--config`: Path to model configuration file
- `--dataset-dir`: Path to dataset directory
- `--n-folds`: Number of cross-validation folds (default: 5)
- `--run-name`: Unique name for the experiment
- `--batch-size`: Training batch size (default: 2)
- `--batch-size-eval`: Evaluation batch size (default: 1)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.00005)
- `--device`: Device to use (cuda, cuda:0, cuda:1, cpu)

### Model Configurations

#### DPT-DINOv2-Giant with LoRA
```yaml
# configs/dpt-dinov2-giant_lora.yaml
model:
  name: "dpt-dinov2-giant"
  
lora:
  r: 128
  lora_alpha: 32
  modules_to_save: ["head"]

training:
  epochs: 1000
  optimizer:
    type: torch.optim.AdamW
    lr: 0.0005
    weight_decay: 0.01
```

#### SegFormer-MiT-b2 with LoRA
```yaml
# configs/segformer-mit-b2_lora.yaml
model:
  name: "segformer-mit-b2"
  
lora:
  r: 64
  lora_alpha: 16
  modules_to_save: ["decode_head"]
```

### Training Process

The fine-tuning pipeline includes:

1. **Cross-Validation**: 5-fold cross-validation by default
2. **Data Augmentation**: Random crops, horizontal flips, normalization
3. **Memory Management**: Automatic memory cleanup and efficient batch processing
4. **Progress Tracking**: Detailed progress bars and timing estimates
5. **Model Checkpointing**: Automatic saving of best models

### Evaluation Metrics

The pipeline evaluates models using:
- **Mean IoU (Intersection over Union)**
- **Pixel Accuracy**
- **Per-class IoU**
- **Precision, Recall, F1-Score**

### Output Structure

After training, the pipeline creates:
```
checkpoints/
├── fold1/
│   ├── coral_bleaching_experiment_fold1_final.pth
│   └── benchmark_results.json
├── fold2/
│   └── ...
└── ...
```

## 📊 Data Visualization

### Coral Reef Dataset Loader
The project includes a PyTorch DataLoader for coral reef images with segmentation masks:

```python
# Example usage from dataloader_example_masks.ipynb
from ReefSegDataset import ReefSegDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = ReefSegDataset(
    images_dir="data/images",
    masks_stitched_dir="data/masks_stitched",
    resize=(512, 512)
)

# Create data loader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

### Color Correction
The project includes color correction utilities for underwater images:
- **Gray-World Algorithm**: `Color Correction/Gray-World.py`
- **Histogram Matching**: `Color Correction/Histogram Matching.py`

## 🔧 Technical Details

### Model Architecture
The fine-tuning pipeline uses state-of-the-art vision transformers and CNNs:
- **DPT (Dense Prediction Transformer)**: For dense prediction tasks
- **DINOv2**: Self-supervised vision transformer backbone
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning technique
- **SegFormer**: Efficient transformer for semantic segmentation

### Class Mapping
The pipeline maps 40 original coral classes to 3 main categories:
- **Class 0**: Background, sand, rubble, non-bleached coral
- **Class 1**: Bleached coral regions
- **Class 2**: Additional background classes
  The following classes are found from the Coralscapes dataset from coralscapes' dataset from huggingface

```python
# Background classes -> 0 (background)
0: 0,   # Undetected background
1: 0,   # seagrass
2: 0,   # trash
5: 0,   # sand
7: 0,   # human
8: 0,   # transect tools
9: 0,   # fish
10: 0,  # algae_covered_substrate
11: 0,  # other animal
12: 0,  # unknown hard subtrate
13: 0,  # black background
14: 0,  # dark
15: 0,  # transect line

# Bleached coral classes -> 1 (bleached)
4: 1,   # other coral bleached
16: 1,  # massive_meandering_bleached -> bleached
19: 1,  # branching_bleached -> bleached
33: 1,  # meandering_bleached -> bleached

# Non-bleached coral classes -> 2 (non-bleached)
3: 2,   # other coral dead
6: 2,   # other coral alive
17: 2,  # massive_meandering_alive -> non-bleached
18: 2,  # rubble (coral rubble) -> non-bleached
20: 2,  # branching_dead -> non-bleached (dead but not bleached)
21: 2,  # millepora -> non-bleached
22: 2,  # branching_alive -> non-bleached
23: 2,  # massive_meandering_dead -> non-bleached (dead but not bleached)
24: 2,  # clam -> non-bleached
25: 2,  # acropora_alive -> non-bleached
26: 2,  # sea_cucumber -> non-bleached
27: 2,  # turbinaria -> non-bleached
28: 2,  # table_acropora_alive -> non-bleached
29: 2,  # sponge -> non-bleached
30: 2,  # anemone -> non-bleached
31: 2,  # pocillopora_alive -> non-bleached
32: 2,  # table_acropora_dead -> non-bleached (dead but not bleached)
34: 2,  # stylophora_alive -> non-bleached
35: 2,  # sea_urchin -> non-bleached
36: 2,  # meandering_alive -> non-bleached
37: 2,  # meandering_dead -> non-bleached (dead but not bleached)
38: 2,  # crown_of_thorn -> non-bleached
39: 2,  # dead_clam -> non-bleached
```


### Memory Optimization
- Automatic memory cleanup between epochs
- Efficient batch processing with custom collate functions
- GPU memory monitoring and optimization
- Fallback to CPU processing when GPU memory is insufficient

## 📈 Performance

The fine-tuning pipeline achieves competitive results on coral bleaching detection:
- **Mean IoU**: ~68% on validation set
- **Pixel Accuracy**: ~90% on background classes
- **Bleached Coral Detection**: ~53% IoU for bleached coral regions

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size: `--batch-size 1`
   - Use CPU: `--device cpu`
   - Enable memory cleanup in configuration

2. **Dataset Not Found**
   - Ensure correct dataset structure
   - Check file paths in configuration
   - Verify image and mask file naming conventions

3. **CUDA Issues**
   - Check GPU availability: `torch.cuda.is_available()`
   - Verify CUDA version compatibility
   - Use CPU fallback: `--device cpu`

## 📚 Additional Resources
- **Group 03 dc3 original drive**: [Google drive link](https://drive.google.com/drive/folders/15pPCEVRFyHb3JQkFSnP8L6a0_nfm8ijp?usp=sharing)
- **Our experiment result checkpoints**: [Model weights Experiments](https://drive.google.com/drive/folders/1vh7podhA54w_I_SvLaeoT15at0R-qawB?usp=sharing)
- **Coralscapes Dataset**: [Hugging Face](https://huggingface.co/datasets/EPFL-ECEO/coralscapes)
- **Model Checkpoints**: [Hugging Face Models](https://huggingface.co/EPFL-ECEO)

## Authors
* Juliette Hattingh-Haasbroek (1779192)
* Doah Lee (2034395)
* Roan van Merwijk (1856022)
* Elvir Nikq (1931075)
* Muhammad Rafiq (1924214)
* Melissa Selamet (1921495)
