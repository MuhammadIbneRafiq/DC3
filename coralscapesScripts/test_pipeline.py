import os
import argparse
import random
import glob
from datetime import datetime

parser = argparse.ArgumentParser(description="Quick test of coral bleaching fine-tuning pipeline")
parser.add_argument("--config", type=str, default="configs/coral_bleaching_dpt_dinov2.yaml", 
                    help="Path to config file")
parser.add_argument("--dataset-dir", type=str, default="../coralscapes", 
                    help="Path to dataset directory")
parser.add_argument("--test-images", type=int, default=10, 
                    help="Number of images to use for testing")
parser.add_argument("--train-ratio", type=float, default=0.6, 
                    help="Ratio of images for training (default: 0.6 = 18/30)")
parser.add_argument("--val-ratio", type=float, default=0.233, 
                    help="Ratio of images for validation (default: 0.233 = 7/30)")
parser.add_argument("--test-ratio", type=float, default=0.167, 
                    help="Ratio of images for testing (default: 0.167 = 5/30)")
parser.add_argument("--epochs", type=int, default=3, 
                    help="Number of epochs for testing")
parser.add_argument("--batch-size", type=int, default=2, 
                    help="Training batch size")
parser.add_argument("--device", type=str, default="gpu1", 
                    choices=["cpu", "gpu1"], 
                    help="Device to use for training")

args = parser.parse_args()

random.seed(42)

# Get all available images
images_dir = os.path.join(args.dataset_dir, "images")
all_images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

selected_images = random.sample(all_images, args.test_images)

print(f"🎯 Selected {len(selected_images)} images for testing")

# Split images into train/val/test
n_train = int(len(selected_images) * args.train_ratio)
n_val = int(len(selected_images) * args.val_ratio)
n_test = len(selected_images) - n_train - n_val

random.shuffle(selected_images)
train_images = selected_images[:n_train]
val_images = selected_images[n_train:n_train + n_val]
test_images = selected_images[n_train + n_val:]
# Create temporary directory structure for testing
test_dir = "./test_data"
os.makedirs(test_dir, exist_ok=True)
os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "masks_bleached"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "masks_non_bleached"), exist_ok=True)

import shutil

print(f"📁 Creating test dataset in {test_dir}...")
for img_path in selected_images:
    img_name = os.path.basename(img_path)
    img_name_no_ext = os.path.splitext(img_name)[0]
    
    # Copy image
    shutil.copy2(img_path, os.path.join(test_dir, "images", img_name))
    
    # Copy corresponding masks if they exist
    bleached_mask_src = os.path.join(args.dataset_dir, "masks_bleached", f"{img_name_no_ext}_bleached.png")
    non_bleached_mask_src = os.path.join(args.dataset_dir, "masks_non_bleached", f"{img_name_no_ext}_non_bleached.png")
    
    bleached_mask_dst = os.path.join(test_dir, "masks_bleached", f"{img_name_no_ext}_bleached.png")
    non_bleached_mask_dst = os.path.join(test_dir, "masks_non_bleached", f"{img_name_no_ext}_non_bleached.png")
    
    if os.path.exists(bleached_mask_src):
        shutil.copy2(bleached_mask_src, bleached_mask_dst)
    if os.path.exists(non_bleached_mask_src):
        shutil.copy2(non_bleached_mask_src, non_bleached_mask_dst)


# Create a unique run name with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"test_run_{timestamp}"

# Convert device format for fine_tune_pipeline.py
if args.device == "gpu1":
    device_arg = "cuda:1"
elif args.device == "gpu0":
    device_arg = "cuda:0"
else:
    device_arg = args.device

# Build the command for fine-tuning pipeline
cmd = [
    "python", "fine_tune_pipeline.py",
    f"--config={args.config}",
    f"--dataset-dir={test_dir}",
    f"--n-folds=2",  # Use only 2 folds for quick testing
    f"--run-name={run_name}",
    f"--batch-size={args.batch_size}",
    f"--epochs={args.epochs}",
    f"--device={device_arg}"
]

print(f"\n🚀 Running test pipeline with:")
print(f"   Config: {args.config}")
print(f"   Dataset: {test_dir}")
print(f"   Epochs: {args.epochs}")
print(f"   Batch size: {args.batch_size}")
print(f"   Device: {device_arg}")
print(f"   Cross-validation folds: 2")

print(f"\n⏱️  Starting test run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

import subprocess
result = subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
print(f"📁 Results saved in: ./checkpoints/")