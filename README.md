# Group 3 - Coral Bleaching Detection
## JBG060 Capstone Data Challenge 2025-2026 Q1

This repository contains the code for coral bleaching detection using deep learning models. The project implements a comprehensive fine-tuning pipeline for semantic segmentation of coral reef images to identify bleached and non-bleached coral reefs.

## Quick Start
### Python Version
```aiignore
python 3.11
```

### Environment Setup

Install the required packages.
```
pip install -r requirements.txt
```

### Dataset Requirements
Download the [`UNAL_BLEACHING_TAYRONA`](https://drive.google.com/drive/folders/1mOuhlo0y-b65eo8QzlyUYLQMpwmvJYXF?usp=sharing) 
dataset provided by `reef_support` and upload the dataset into the directory `DC3/data/unclustered_none`.

The fine-tuning pipeline expects the following dataset structure:
```
unclustered_none/
├── images/                   # Original RGB coral reef images (.jpg)
├── masks_bleached/           # Binary masks for bleached coral reefs (.png)
└── masks_non_bleached/       # Binary masks for non-bleached coral reefs (.png)
```

## Fine-tuning Pipeline

### Overview
The fine-tuning pipeline is designed to train deep learning models for coral bleaching detection using semantic segmentation. It supports multiple model architectures and includes cross-validation for robust evaluation.

### Model Architectures
We are using the **SegFormer-MiT-B2 (LoRA)** model. Its configurations can be
found in `configs/segformer-mit-b2_lora.yaml`

### Running Fine-tuning
In the file `config.py`, specify the dataset you want to run the model on.
In the `main.py` file, the images will be clustered based on their mean color and color filters will 
be applied to each cluster. Afterward, the model runs on the specified dataset `root` in `config.py`.


#### Command Line Arguments
- `--config`: Path to model configuration file
- `--dataset-dir`: Path to dataset directory
- `--n-folds`: Number of cross-validation folds (default: 5)
- `--run-name`: Unique name for the experiment
- `--batch-size`: Training batch size (default: 1)
- `--batch-size-eval`: Evaluation batch size (default: 1)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.00005)
- `--device`: Device to use (cuda, cuda:0, cuda:1, cpu)

[//]: # (### Model Configurations)

[//]: # (#### DPT-DINOv2-Giant with LoRA)

[//]: # (```yaml)

[//]: # (# configs/dpt-dinov2-giant_lora.yaml)

[//]: # (model:)

[//]: # (  name: "dpt-dinov2-giant")

[//]: # (  )
[//]: # (lora:)

[//]: # (  r: 128)

[//]: # (  lora_alpha: 32)

[//]: # (  modules_to_save: ["head"])

[//]: # ()
[//]: # (training:)

[//]: # (  epochs: 1000)

[//]: # (  optimizer:)

[//]: # (    type: torch.optim.AdamW)

[//]: # (    lr: 0.0005)

[//]: # (    weight_decay: 0.01)

[//]: # (```)

[//]: # (#### SegFormer-MiT-B2 with LoRA)

[//]: # (```yaml)

[//]: # (# configs/segformer-mit-b2_lora.yaml)

[//]: # (model:)

[//]: # (  name: "segformer-mit-b2")

[//]: # (  )
[//]: # (lora:)

[//]: # (  r: 64)

[//]: # (  lora_alpha: 16)

[//]: # (  modules_to_save: ["decode_head"])

[//]: # (```)

### Training Process

The fine-tuning pipeline includes:

1. **Cross-Validation**: 5-fold cross-validation by default
2. **Data Augmentation**: Random crops, horizontal flips, normalization
3. **Memory Management**: Automatic memory cleanup and efficient batch processing
4. **Progress Tracking**: Detailed progress bars and timing estimates
5. **Model Checkpointing**: Automatic saving of best models based on the Coral Rank score

[//]: # (The pipeline evaluates models using:)

[//]: # (- **Mean Intersection over Union &#40;IoU&#41;**)

[//]: # (- **Pixel Accuracy**)

[//]: # (- **Per-class IoU**)

[//]: # (- **Coral Rank**)

[//]: # (- $`\text{mIoU} = \sum_{c \in \mathcal{C}} w_c \cdot \text{IoU}_c = 0.1 \cdot \text{IoU}_{\text{background}} + 0.45 \cdot \text{IoU}_{\text{bleached}} + 0.45 \cdot \text{IoU}_{\text{non-bleached}}`$)

### Output Structure

After training, the pipeline creates:
```
checkpoints_{color_name or baseline}_{cluster_number or none}/
├── best_overall_model.pth
├── kfold_metrics.json
├── metrics_fold_1.json
├── metrics_fold_2.json
├── metrics_fold_3.json
├── metrics_fold_4.json
├── metrics_fold_5.json
├── model_fold_1_best_score.pth
├── model_fold_2_best_score.pth
├── model_fold_3_best_score.pth
├── model_fold_4_best_score.pth
└── model_fold_5_best_score.pth
```
### Evaluation Metrics
The Coral Rank (CR) score is calculated as a weighted combination of accuracy and mean Intersection over Union (mIoU):

$`\text{CR} = 0.2 \cdot \mathcal{A} + 0.8 \cdot \text{mIoU}`$,

where $\mathcal{A}$ represents the classification accuracy, and the mIoU is computed as:

$`\text{mIoU} = \sum_{c \in \mathcal{C}} w_c \cdot \text{IoU}_c = 0.1 \cdot \text{IoU}_{\text{background}} + 0.45 \cdot \text{IoU}_{\text{bleached}} + 0.45 \cdot \text{IoU}_{\text{non-bleached}}`$.

### Inference
After running the model, evaluation metrics can be run by running `inference.py`.

[//]: # (After training, the pipeline creates:)

[//]: # (```)

[//]: # (checkpoints_{cluster_name or basline}_{color or none}/)

[//]: # (├── fold1/)

[//]: # (│   ├── coral_bleaching_experiment_fold1_final.pth)

[//]: # (│   └── benchmark_results.json)

[//]: # (├── fold2/)

[//]: # (│   └── ...)

[//]: # (└── ...)

[//]: # (```)

[//]: # (## 📊 Data Visualization)

[//]: # ()
[//]: # (### Coral Reef Dataset Loader)

[//]: # (The project includes a PyTorch DataLoader for coral reef images with segmentation masks:)

[//]: # ()
[//]: # (```python)

[//]: # (# Example usage from dataloader_example_masks.ipynb)

[//]: # (from ReefSegDataset import ReefSegDataset)

[//]: # (from torch.utils.data import DataLoader)

[//]: # ()
[//]: # (# Create dataset)

[//]: # (dataset = ReefSegDataset&#40;)

[//]: # (    images_dir="data/images",)

[//]: # (    masks_stitched_dir="data/masks_stitched",)

[//]: # (    resize=&#40;512, 512&#41;)

[//]: # (&#41;)

[//]: # ()
[//]: # (# Create data loader)

[//]: # (dataloader = DataLoader&#40;dataset, batch_size=4, shuffle=True&#41;)

[//]: # (```)

[//]: # ()
[//]: # (### Color Correction)

[//]: # (The project includes color correction utilities for underwater images:)

[//]: # (- **Gray-World Algorithm**: `Color Correction/Gray-World.py`)

[//]: # (- **Histogram Matching**: `Color Correction/Histogram Matching.py`)

## Technical Details

[//]: # (### Model Architecture)

[//]: # (The fine-tuning pipeline uses state-of-the-art vision transformers and CNNs:)

[//]: # (- **DPT &#40;Dense Prediction Transformer&#41;**: For dense prediction tasks)

[//]: # (- **DINOv2**: Self-supervised vision transformer backbone)

[//]: # (- **LoRA &#40;Low-Rank Adaptation&#41;**: Efficient fine-tuning technique)

[//]: # (- **SegFormer**: Efficient transformer for semantic segmentation)

### Class Mapping
The pipeline maps 40 original coral classes to 3 main categories:
- **Class 0**: Background (sand, rubble, fish)
- **Class 1**: Bleached coral reefs
- **Class 2**: Non-bleached coral reefs

The following classes are found from the Coralscapes dataset:
```python
class_mapping = {
    # Background classes
    0: 0, 1: 0, 2: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
    # Bleached coral classes
    3: 1, 4: 1, 16: 1, 19: 1, 33: 1,
    # Non-bleached coral classes
    17: 2, 18: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2,
    30: 2, 31: 2, 32: 2, 34: 2, 35: 2, 36: 2, 37: 2, 38: 2, 39: 2
}
```

### Memory Optimization
- Automatic memory cleanup between epochs
- Efficient batch processing with custom collate functions
- GPU memory monitoring and optimization
- Fallback to CPU processing when GPU memory is insufficient

[//]: # (## 📈 Performance)

[//]: # ()
[//]: # (The fine-tuning pipeline achieves competitive results on coral bleaching detection:)

[//]: # (- **Mean IoU**: ~68% on validation set)

[//]: # (- **Pixel Accuracy**: ~90% on background classes)

[//]: # (- **Bleached Coral Detection**: ~53% IoU for bleached coral regions)

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size: `--batch-size 1` or `--batch-size-eval 1`
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

## Additional Resources

[//]: # (- **Group 03 dc3 original drive**: [Google drive link]&#40;https://drive.google.com/drive/folders/15pPCEVRFyHb3JQkFSnP8L6a0_nfm8ijp?usp=sharing&#41;)
- **Our color filtered cluster data**: [DC3 color filter data](https://drive.google.com/drive/folders/1yafN3OAzJ5BbFOOeogXgWdIEAqdzqh-T?usp=sharing)
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
