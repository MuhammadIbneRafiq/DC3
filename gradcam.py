import os
import math
import cv2
import torch
import torch.nn.functional as F
import albumentations as A
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from config import input_path, VIS_DIR, MODEL_NAME
from inference import load_lora_segformer_model


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


class CoralBleachingDatasetGradCam(Dataset):
    """
    Note: this object is made by Coralscapes and modified by us to map to 3 classes instead of 40.
    The original object can be found at: https://github.com/eceo-epfl/coralscapesScripts/blob/main/coralscapesScripts/datasets/dataset.py

    Dataset for coral bleaching segmentation.
    Expects data structure:
    - images/: RGB images
    - masks_bleached/: Masks for bleached corals
    - masks_non_bleached/: Masks for non-bleached corals
    """
    def __init__(self, root: str, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform

        self.images_dir = os.path.join(self.root, 'images')
        self.masks_bleached_dir = os.path.join(self.root, 'masks_bleached')
        self.masks_non_bleached_dir = os.path.join(self.root, 'masks_non_bleached')

        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")

        all_images = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.jpg', '.png'))]
        self.images = []

        for img_name in all_images:
            base_name, ext = os.path.splitext(img_name)

            bleached_mask_name = f"{base_name}_bleached.png"
            non_bleached_mask_name = f"{base_name}_non_bleached.png"

            bleached_mask_path = os.path.join(self.masks_bleached_dir, bleached_mask_name)
            non_bleached_mask_path = os.path.join(self.masks_non_bleached_dir, non_bleached_mask_name)

            if os.path.exists(bleached_mask_path) and os.path.exists(non_bleached_mask_path):
                 self.images.append(img_name)

        print(f"Found {len(self.images)} image-mask pairs for inference.")
        if not self.images:
            print(f"CRITICAL: Found 0 pairs in {self.root}. Check naming and file extensions.")

    def __getitem__(self, index: int):
        """
        Retrieve and transform an image and its corresponding segmentation maps.
        Args:
            index (int): Index of the image and target to retrieve.
        Returns:
            tuple: A tuple containing:
            - image (numpy.ndarray): The transformed image.
            - target (numpy.ndarray): The transformed segmentation map.
        """
        img_name = self.images[index]
        base_name, ext = os.path.splitext(img_name)

        image_path = os.path.join(self.root, 'images', img_name)
        bleached_mask_path = os.path.join(self.root, 'masks_bleached', f"{base_name}_bleached.png")
        non_bleached_mask_path = os.path.join(self.root, 'masks_non_bleached', f"{base_name}_non_bleached.png")

        image = np.array(Image.open(image_path).convert('RGB'))
        bleached_mask = np.array(Image.open(bleached_mask_path).convert('L'))
        non_bleached_mask = np.array(Image.open(non_bleached_mask_path).convert('L'))

        target = np.zeros_like(bleached_mask, dtype=np.uint8)
        target[non_bleached_mask > 128] = 2
        target[bleached_mask > 128] = 1

        if self.transform:
            transformed = self.transform(image=image, mask=target)
            image = transformed["image"]
            target = transformed["mask"]

        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        target = torch.tensor(target, dtype=torch.long)

        return image, target, img_name

    def __len__(self):
        return len(self.images)


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

def grad_cam(N_CLASSES, DEVICE):
    for cluster_id, runs in cluster_runs.items():
        base_dataset_folder = runs[-1][1]
        dataset_root = os.path.join(input_path, base_dataset_folder)
        dataset = CoralBleachingDatasetGradCam(dataset_root, transform=inference_transform)
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
