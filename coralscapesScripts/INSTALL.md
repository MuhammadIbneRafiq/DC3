# Installation Guide for Coral Bleaching Detection

This guide provides instructions for setting up the environment needed to run the coral bleaching detection fine-tuning pipeline.

## Option 1: Using pip (recommended for most users)

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv coral_env
   # On Windows
   coral_env\Scripts\activate
   # On Linux/Mac
   source coral_env/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Option 2: Using conda

1. Create a conda environment:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate coral_bleaching
   ```

## Running the Fine-Tuning Pipeline

Once you have set up the environment, you can run the fine-tuning pipeline:

```bash
# Basic usage
python run_fine_tuning.py

# With custom parameters
python run_fine_tuning.py --n-folds=3 --epochs=50 --batch-size=2
```

### Command Line Arguments

- `--config`: Path to config file (default: configs/coral_bleaching_dpt_dinov2.yaml)
- `--dataset-dir`: Path to dataset directory (default: ../coralscapes)
- `--n-folds`: Number of cross-validation folds (default: 5)
- `--batch-size`: Training batch size (default: 2)
- `--epochs`: Number of training epochs (default: 50)

## Dataset Structure

The dataset should be organized as follows:

```
coralscapes/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── masks_bleached/
│   ├── image1_bleached.png
│   ├── image2_bleached.png
│   └── ...
└── masks_non_bleached/
    ├── image1_non_bleached.png
    ├── image2_non_bleached.png
    └── ...
```

## Troubleshooting

If you encounter CUDA out of memory errors:
- Reduce the batch size (`--batch-size=1`)
- Reduce the input image size in the config file

For other issues, please check the error messages and ensure all dependencies are correctly installed.
