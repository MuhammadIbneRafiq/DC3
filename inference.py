import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
import albumentations as A
import cv2
import pandas as pd
import math
import random
from config import __RANDOM_STATE__, root, MODEL_NAME, VIS_DIR
from inference_functions import (CoralBleachingDataset, load_lora_segformer_model, run_inference_and_metrics,
                                 calculate_custom_metrics, visualize_single_sample_canvas)


torch.manual_seed(__RANDOM_STATE__)
torch.cuda.manual_seed(__RANDOM_STATE__)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def inference(N_CLASSES, device):
    print(f"Final canvas visualizations will be saved to: {VIS_DIR}")
    inference_runs = [('Cyan 0', 'data_cyan_0', 'checkpoints_cyan_0'),
                      ('Cyan 1', 'data_cyan_1', 'checkpoints_cyan_1'),
                      ('Cyan 2', 'data_cyan_2', 'checkpoints_cyan_2'),
                      ('Magenta 0', 'data_magenta_0', 'checkpoints_magenta_0'),
                      ('Magenta 1', 'data_magenta_1', 'checkpoints_magenta_1'),
                      ('Magenta 2', 'data_magenta_2', 'checkpoints_magenta_2'),
                      ('Yellow 0', 'data_yellow_0', 'checkpoints_yellow_0'),
                      ('Yellow 1', 'data_yellow_1', 'checkpoints_yellow_1'),
                      ('Yellow 2', 'data_yellow_2', 'checkpoints_yellow_2'),
                      ('Cluster 0 (Base)', 'cluster_0', 'checkpoints_baseline_0'),
                      ('Cluster 1 (Base)', 'cluster_1', 'checkpoints_baseline_1'),
                      ('Cluster 2 (Base)', 'cluster_2', 'checkpoints_baseline_2'),
                      ('Baseline)', 'unclustered_none', 'checkpoints_baseline_none')]
    all_results = []
    inference_transform = A.Compose([A.Resize(height=(512, 512)[0], width=(512, 512)[1], interpolation=cv2.INTER_NEAREST)])

    print(f"Starting 13-run inference loop using device: {device}\n")
    for run_name, dataset_folder, checkpoint_folder in tqdm(inference_runs, desc="Overall Inference Runs"):
        print(f"\nRunning: {run_name} (Data: {dataset_folder})")
        if not os.path.isdir(root) or not os.path.exists(checkpoint_folder):
            print(f"Skipping {run_name}: Data or Checkpoint path not found.")
            continue

        try:
            full_dataset = CoralBleachingDataset(root=root, transform=inference_transform)
            total_size = len(full_dataset)
            if total_size == 0:
                print(f"Skipping {run_name}: Dataset size is 0.")
                continue

            subset_start_index = math.floor(total_size * 0.85)
            subset_indices = list(range(subset_start_index, total_size))
            inference_subset = Subset(full_dataset, subset_indices)

            print(f"Using the last 15% of images: {len(inference_subset)}/{total_size} images.")
            inference_dataloader = DataLoader(inference_subset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
            model = load_lora_segformer_model(checkpoint_folder, MODEL_NAME, N_CLASSES, device)

            if model is None:
                continue

            metrics = run_inference_and_metrics(model, inference_dataloader, device, N_CLASSES)
            mIoU_weighted, coral_rank_score = calculate_custom_metrics(metrics)
            all_results.append({"Run Name": run_name, "Accuracy": metrics['Accuracy'], "mIoU": metrics['mIoU'],
                                "IoU (Background)": metrics['IoU - Background'], "IoU (Bleached)": metrics['IoU - Bleached'],
                                "IoU (Non-Bleached)": metrics['IoU - Non-Bleached'], "Rank Score": coral_rank_score,
                                "Test Images": len(inference_subset)})

            if len(inference_subset) > 0:
                random_index_in_subset = random.choice(subset_indices)
                visualize_single_sample_canvas(model, full_dataset, random_index_in_subset, run_name, VIS_DIR)

        except FileNotFoundError as e:
            print(f"Skipping {run_name}: File not found error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during {run_name}: {e}")

    print(f"\n\nFinal Canvas Visualizations saved to: {VIS_DIR}\n{'-' * 70}")
    if all_results:
        results_df = pd.DataFrame(all_results)
        float_cols = results_df.columns.drop(['Run Name', 'Test Images'])
        results_df[float_cols] = results_df[float_cols].apply(lambda x: pd.to_numeric(x, errors='coerce').round(4))
        results_df['Cluster'] = results_df['Run Name'].str.extract('(\d+)').astype(int)
        final_table = results_df.sort_values(by=["Cluster", "Rank Score"], ascending=[True, False])
        final_table = final_table[["Run Name", "Test Images", "Accuracy", "mIoU", "IoU (Background)", "IoU (Bleached)",
                                   "IoU (Non-Bleached)", "Rank Score"]]
        print(final_table.to_markdown(index=False))
    else:
        print("No results were generated. Please check all paths and data contents.")
