# Group 3 - Coral Bleaching Detection
## JBG060 Capstone Data Challenge 2025-2026 Q1

This repository contains the code for coral bleaching detection using deep learning models. The project implements a comprehensive fine-tuning pipeline for semantic segmentation of coral reef images to identify bleached and non-bleached coral reefs.

Repository: [https://github.com/MuhammadIbneRafiq/DC3](https://github.com/MuhammadIbneRafiq/DC3)

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

### Running Preprocessing, Fine-tuning, Inference, and Grad-CAM
In the file `config.py`, specify the dataset you want to run the model on.
Ensure that the file paths are valid and lead to the needed file/folder.

In the `main.ipynb` file, the images will be clustered based on their mean color and color filters will 
be applied to each cluster. The model runs on the specified dataset `root` in `config.py`. 
Afterward, inference will be run where evaluation metrics will be computed on the
test set. Lastly, a sample image from each cluster will be taken and a Grad-CAM
heatmap will be applied to explain the model's output.

### Training Process
The fine-tuning pipeline includes:
1. **Cross-Validation**: 5-fold cross-validation by default
2. **Data Augmentation**: Random crops, horizontal flips, normalization
3. **Memory Management**: Automatic memory cleanup and efficient batch processing
4. **Progress Tracking**: Detailed progress bars and timing estimates
5. **Model Checkpointing**: Automatic saving of best models based on the Coral Rank score

### Output Structure
After training, the pipeline creates a checkpoints folder for the specified cluster and color:
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

## Technical Details
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

## Additional Resources
- **Our color filtered cluster data and models' checkpoints and best weights**: [DC3 data](https://drive.google.com/drive/folders/1yafN3OAzJ5BbFOOeogXgWdIEAqdzqh-T?usp=sharing)
- **Coralscapes Dataset**: [Hugging Face](https://huggingface.co/datasets/EPFL-ECEO/coralscapes)
- **Model Checkpoints**: [Hugging Face Models](https://huggingface.co/EPFL-ECEO)

## Authors
* Juliette Hattingh-Haasbroek (1779192)
* Doah Lee (2034395)
* Roan van Merwijk (1856022)
* Elvir Nikq (1931075)
* Muhammad Rafiq (1924214)
* Melissa Selamet (1921495)
