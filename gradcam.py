import os
import math
import cv2
import torch
import torch.nn.functional as F
import albumentations as A
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from config import input_path, output_path, VIS_DIR, MODEL_NAME
from image_preprocessing import ImagePreprocessing
from inference_functions import CoralBleachingDataset, load_lora_segformer_model


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
N_CLASSES = ImagePreprocessing(input_path, output_path).optimal_k
cluster_runs = {0: [('Cyan 0', 'data_cyan_0', 'checkpoints_cyan_0'),
                    ('Magenta 0', 'data_magenta_0', 'checkpoints_magenta_0'),
                    ('Yellow 0', 'data_yellow_0', 'checkpoints_yellow_0'),
                    ('Cluster 0 (Base)', 'cluster_0', 'checkpoints_baseline_0')],
                1: [('Cyan 1', 'data_cyan_1', 'checkpoints_cyan_1'),
                    ('Magenta 1', 'data_magenta_1', 'checkpoints_magenta_1'),
                    ('Yellow 1', 'data_yellow_1', 'checkpoints_yellow_1'),
                    ('Cluster 1 (Base)', 'cluster_1', 'checkpoints_baseline_1')],
                2: [('Cyan 2', 'data_cyan_2', 'checkpoints_cyan_2'),
                    ('Magenta 2', 'data_magenta_2', 'checkpoints_magenta_2'),
                    ('Yellow 2', 'data_yellow_2', 'checkpoints_yellow_2'),
                    ('Cluster 2 (Base)', 'cluster_2', 'checkpoints_baseline_2')]}


class SegformerWrapper(torch.nn.Module):
    """
    Create wrapper for Segformer - MiT-B2 model.
    Ensures that forward pass returns logit tensor.
    """
    def __init__(self, model):
        super(SegformerWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        """
        Forward pass.
        """
        out = self.model(x)
        if isinstance(out, dict) and 'logits' in out:
            return out['logits']
        return out


inference_transform = A.Compose([A.Resize(height=(512, 512)[0], width=(512, 512)[1], interpolation=cv2.INTER_NEAREST)])

for cluster_id, runs in cluster_runs.items():
    base_dataset_folder = runs[-1][1]
    dataset_root = os.path.join(input_path, base_dataset_folder)
    dataset = CoralBleachingDataset(dataset_root, transform=inference_transform)
    if len(dataset) == 0:
        continue

    total_size = len(dataset)
    subset_start_index = math.floor(total_size * 0.85)
    subset_indices = list(range(subset_start_index, total_size))
    img_tensor, mask_tensor, img_name = dataset[subset_indices[-1]]
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    print(f"\nCluster {cluster_id}: Using image {img_name} for Grad-CAM across all filters.")
    for run_name, dataset_folder, checkpoint_path in runs:
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {run_name}: checkpoint not found.")
            continue

        model = load_lora_segformer_model(checkpoint_path, MODEL_NAME, N_CLASSES, DEVICE)
        if model is None:
            continue

        wrapped_model = SegformerWrapper(model)
        base_model = wrapped_model.model
        target_layer = base_model.decode_head.linear_fuse
        cam = GradCAM(model=wrapped_model, target_layers=[target_layer])

        with torch.no_grad():
            dummy_out = wrapped_model(img_tensor)
            mask_resized = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                                         size=dummy_out.shape[2:], mode='nearest').squeeze(0).squeeze(0)
            mask_np = mask_resized.cpu().numpy()
            targets = [SemanticSegmentationTarget(category=1, mask=mask_np)]

        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
        orig_image = (img_tensor[0].cpu().numpy().transpose(1, 2, 0)).astype(np.float32)
        orig_image = orig_image / orig_image.max()
        cam_overlay = show_cam_on_image(orig_image, grayscale_cam, use_rgb=True)
        overlay_pil = Image.fromarray(cam_overlay)
        clean_name = run_name.replace(" ", "_").replace("(", "").replace(")", "")
        save_path = os.path.join(VIS_DIR, f"{clean_name}_gradcam_{img_name}.png")
        overlay_pil.save(save_path)

        print(f"Saved Grad-CAM for {run_name} -> {save_path}")
