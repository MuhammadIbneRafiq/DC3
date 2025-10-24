import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from peft import PeftModel, LoraConfig


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CoralBleachingDataset(Dataset):
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


def load_lora_segformer_model(checkpoint_path: str, model_name: str, n_classes: int, device) -> PeftModel | None:
    """
    Load SegFormer-MiT-B2 (LoRA) model with weights from specific cluster.
    :param checkpoint_path: path to cluster's best weights
    :param model_name: model to load, SegFormer-MiT-B2 (LoRA)
    :param n_classes: number of classes of masks
    :param device: load to CUDA or CPU
    :return: model
    """
    config = SegformerConfig.from_pretrained(model_name, num_labels=n_classes)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name, config=config, ignore_mismatched_sizes=True
    ).to(device)

    lora_config = LoraConfig(r=128, lora_alpha=16, target_modules=["query", "value"],
                             lora_dropout=0.1, bias="none", modules_to_save=["decode_head"])
    model = PeftModel(model, lora_config)

    try:
        checkpoint = torch.load(f"{checkpoint_path}/best_overall_model.pth", map_location=device)
        state_dict_to_load = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict_to_load, strict=False)
    except Exception as e:
        print(f"FATAL ERROR during model loading for {checkpoint_path}: {e}")
        return None

    model.to(device)
    model.eval()

    return model


def run_inference_and_metrics(model: PeftModel, dataloader, device, n_classes: int) -> dict[str, float]:
    """
    Compute the variables of the mIoU and Coral Rank scores.
    :param model: model to use, SegFormer-MiT-B2 (LoRA)
    :param dataloader: torch dataloader
    :param device: load on CUDA or CPU
    :param n_classes: number of classes of masks
    :return: metrics
    """
    jaccard = MulticlassJaccardIndex(num_classes=n_classes, average='macro', ignore_index=None).to(device)
    accuracy = MulticlassAccuracy(num_classes=n_classes, average='macro', ignore_index=None).to(device)
    jaccard_per_class = MulticlassJaccardIndex(num_classes=n_classes, average='none', ignore_index=None).to(device)

    model.eval()
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            resized_logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            preds = torch.argmax(resized_logits, dim=1)

            jaccard.update(preds, labels)
            accuracy.update(preds, labels)
            jaccard_per_class.update(preds, labels)

    results = {"Accuracy": accuracy.compute().item(),
               "mIoU": jaccard.compute().item(),
               "IoU - Background": jaccard_per_class.compute()[0].item(),
               "IoU - Bleached": jaccard_per_class.compute()[1].item(),
               "IoU - Non-Bleached": jaccard_per_class.compute()[2].item()}

    return results


def calculate_custom_metrics(metrics: dict[str, float]) -> tuple[float, float]:
    """
    Calculate mIoU and Coral Rank scores.
    :param metrics: variables to compute scores
    :return: mIoU and Coral Rank scores.
    """
    mIoU_weighted = (0.1 * metrics['IoU - Background']) + (0.45 * metrics['IoU - Bleached']) + (0.45 * metrics['IoU - Non-Bleached'])
    coral_rank_score = (0.2 * metrics['Accuracy']) + (0.8 * mIoU_weighted)
    return mIoU_weighted, coral_rank_score


def visualize_single_sample_canvas(model: PeftModel, full_dataset, index: int, run_name: str, vis_dir: str):
    """
    Plot 4 plots side-by-side.
    1. input image
    2. ground truth mask of input image
    3. predicted mask of input image using model weights
    4. predicted mask over the input image
    :param model: model to use, SegFormer-MiT-B2 (LoRA)
    :param full_dataset: dataset of images to sample image from
    :param index: index of image
    :param run_name: run name to save image
    :param vis_dir: directory to save image
    """
    COLOR_MAP_RGB = np.array([
        [0, 0, 0],
        [255, 153, 153],
        [229, 255, 204]
    ], dtype=np.uint8)
    COLOR_MAP_RGBA_OVERLAY = np.array([
        [0, 0, 0, 0],
        [255, 153, 153, 180],
        [229, 255, 204, 180]
    ], dtype=np.uint8)

    IMAGE_W, IMAGE_H = (512, 512)
    MARGIN = 60
    PANEL_GAP = 40
    LEGEND_HEIGHT = 80

    try:
        font_path_title = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font_path_legend = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font_size_title = 45
        font_size_legend = 35
        font_title = ImageFont.truetype(font_path_title, font_size_title)
        font_legend = ImageFont.truetype(font_path_legend, font_size_legend)
    except IOError:
        font_size_title = 35
        font_title = ImageFont.load_default()
        font_legend = ImageFont.load_default()
        print(f"  Warning: mwah, fond not found. Using default font with size {font_size_title} for tltles ")

    CANVAS_W = (IMAGE_W * 4) + (PANEL_GAP * 3) + (MARGIN * 2)
    CANVAS_H = IMAGE_H + (MARGIN * 2) + LEGEND_HEIGHT
    CANVAS_COLOR = (255, 255, 255)

    image_tensor, label_tensor, img_name = full_dataset[index]
    inputs = image_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        resized_logits = F.interpolate(
            outputs.logits, size=label_tensor.shape[-2:], mode='bilinear', align_corners=False
        )
        preds = torch.argmax(resized_logits, dim=1)

    image_np = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    label_indices = label_tensor.cpu().numpy()
    gt_mask_color_np = COLOR_MAP_RGB[label_indices]

    pred_indices = preds[0].cpu().numpy()
    pred_mask_color_np = COLOR_MAP_RGB[pred_indices]

    overlay_image_pil = Image.fromarray(image_np).convert("RGBA")
    overlay_mask_rgba = COLOR_MAP_RGBA_OVERLAY[pred_indices]
    overlay_mask_pil = Image.fromarray(overlay_mask_rgba)
    final_overlay_pil = Image.alpha_composite(overlay_image_pil, overlay_mask_pil)
    final_overlay_np = np.array(final_overlay_pil.convert("RGB"))

    panels = [
        Image.fromarray(image_np),
        Image.fromarray(gt_mask_color_np),
        Image.fromarray(pred_mask_color_np),
        Image.fromarray(final_overlay_np)
    ]

    canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), color=CANVAS_COLOR)
    draw = ImageDraw.Draw(canvas)

    titles = ["Input Image", "Ground Truth Mask", "Prediction Mask", "Prediction Overlay"]
    text_color = (0, 0, 0)

    current_x = MARGIN

    for idx, panel in enumerate(panels):
        title = titles[idx]
        text_w = draw.textlength(title, font=font_title)
        title_x = current_x + (IMAGE_W - text_w) / 2

        try:
            _, _, _, title_bbox_h = font_title.getbbox(title)
            title_y = (MARGIN - title_bbox_h) / 2
        except:
            title_y = (MARGIN - font_size_title) / 2

        draw.text((title_x, title_y), title, font=font_title, fill=text_color)
        image_y = MARGIN
        canvas.paste(panel, (current_x, image_y))
        current_x += IMAGE_W + PANEL_GAP

    legend_items = [
        ("Background (Class 0)", COLOR_MAP_RGB[0]),
        ("Bleached Coral (Class 1)", COLOR_MAP_RGB[1]),
        ("Non-Bleached Coral (Class 2)", COLOR_MAP_RGB[2])
    ]

    legend_title = "LEGEND:"
    color_box_size = 30
    spacing = 50

    total_legend_width = draw.textlength(legend_title, font=font_legend) + 30
    for text, _ in legend_items:
        total_legend_width += color_box_size + 10
        total_legend_width += draw.textlength(text, font=font_legend) + spacing
    total_legend_width -= spacing

    legend_start_x = (CANVAS_W - total_legend_width) / 2
    text_bbox = font_legend.getbbox("A")
    text_height = text_bbox[3] - text_bbox[1]
    legend_y = CANVAS_H - LEGEND_HEIGHT + (LEGEND_HEIGHT - text_height) / 2
    current_x = legend_start_x
    draw.text((current_x, legend_y), legend_title, font=font_legend, fill=text_color)
    current_x += draw.textlength(legend_title, font=font_legend) + 20

    for text, color_rgb in legend_items:
        rect_y_start = legend_y - 2
        draw.rectangle([current_x, rect_y_start, current_x + color_box_size, rect_y_start + color_box_size], fill=tuple(color_rgb))
        current_x += color_box_size + 10
        draw.text((current_x, legend_y), text, font=font_legend, fill=text_color)
        current_x += draw.textlength(text, font=font_legend) + spacing

    clean_run_name = run_name.replace(" ", "_").replace("(", "").replace(")", "")
    output_filename = f"{clean_run_name}_sample_canvas_{os.path.splitext(img_name)[0]}.png"
    canvas.save(os.path.join(vis_dir, output_filename))

    print(f"  -> Saved final canvas visualization for: {run_name} ({img_name})")
