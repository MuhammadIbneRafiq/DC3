#!/usr/bin/env python3
"""
Coral Robustness Augmentation Demo

Applies robustness-oriented augmentations to a couple of coral images to simulate
environmental degradations (turbidity/haze, pollution particles, color cast,
blur+noise, low-light, compression, motion blur, speckle noise, vignette,
color jitter, cutout) and produces:
 - A gallery of original vs augmented images
 - Simple image-level metrics plots (contrast, sharpness, colorfulness, haze score)
 - Printed literature references motivating each augmentation

Inputs: Attempts to load 2 images from CoralSeg test set by default. Adjust
IMAGE_DIR if needed.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


IMAGE_DIR = (
    Path("data/mask_labels/content/gdrive/MyDrive/Data Challenge 3 - JBG060 AY2526/01_data/benthic_datasets/mask_labels/Coralseg/test/Image")
)


def load_sample_images(image_dir: Path, max_images: int = 2) -> List[np.ndarray]:
    image_paths = sorted(list(image_dir.glob("*.jpg")))
    images: List[np.ndarray] = []
    for p in image_paths[:max_images]:
        img = cv2.imread(str(p))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images


def to_float01(img: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def apply_turbidity(img: np.ndarray, transmission: float = 0.75, airlight: Tuple[float, float, float] = (0.8, 0.9, 0.95)) -> np.ndarray:
    # Simple atmospheric scattering model: I = J * t + A * (1 - t)
    x = to_float01(img)
    t = np.full_like(x, transmission)
    A = np.array(airlight, dtype=np.float32).reshape(1, 1, 3)
    y = x * t + A * (1.0 - t)
    return to_uint8(y)


def apply_pollution_particles(img: np.ndarray, count: int = 500, radius_range: Tuple[int, int] = (1, 3), opacity: float = 0.35) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()
    for _ in range(count):
        r = np.random.randint(radius_range[0], radius_range[1] + 1)
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        color = (np.random.randint(0, 30),) * 3  # dark specks
        cv2.circle(overlay, (x, y), r, color, -1, lineType=cv2.LINE_AA)
    out = cv2.addWeighted(overlay, opacity, img, 1.0 - opacity, 0)
    return out


def apply_color_cast(img: np.ndarray, gains: Tuple[float, float, float] = (0.9, 1.05, 1.15)) -> np.ndarray:
    x = to_float01(img)
    g = np.array(gains, dtype=np.float32).reshape(1, 1, 3)
    y = np.clip(x * g, 0.0, 1.0)
    return to_uint8(y)


def apply_blur_noise(img: np.ndarray, blur_ksize: int = 5, noise_std: float = 0.02) -> np.ndarray:
    x = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    n = np.random.normal(0.0, noise_std, img.shape).astype(np.float32)
    y = to_float01(x) + n
    return to_uint8(np.clip(y, 0.0, 1.0))


def apply_low_light(img: np.ndarray, gamma: float = 1.8, gain: float = 0.7, noise_std: float = 0.01) -> np.ndarray:
    x = to_float01(img)
    y = np.power(np.clip(x * gain, 0.0, 1.0), gamma)
    n = np.random.normal(0.0, noise_std, img.shape).astype(np.float32)
    y = np.clip(y + n, 0.0, 1.0)
    return to_uint8(y)


def apply_jpeg_compression(img: np.ndarray, quality: int = 25) -> np.ndarray:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, enc = cv2.imencode(".jpg", bgr, encode_param)
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def apply_motion_blur(img: np.ndarray, ksize: int = 9, angle_deg: float = 0.0) -> np.ndarray:
    k = max(3, int(ksize) | 1)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred


def apply_speckle_noise(img: np.ndarray, std: float = 0.05) -> np.ndarray:
    x = to_float01(img)
    noise = np.random.normal(0.0, std, img.shape).astype(np.float32)
    y = x + x * noise
    return to_uint8(np.clip(y, 0.0, 1.0))


def apply_vignette(img: np.ndarray, strength: float = 0.45) -> np.ndarray:
    h, w = img.shape[:2]
    kx = cv2.getGaussianKernel(w, w * strength)
    ky = cv2.getGaussianKernel(h, h * strength)
    mask = (ky @ kx.T)
    mask = mask / mask.max()
    # Keep center near 1, darken edges
    mask = 0.6 + 0.4 * mask
    out = img.astype(np.float32)
    for c in range(3):
        out[..., c] *= mask
    return to_uint8(np.clip(out / 255.0, 0.0, 1.0))


def apply_color_jitter(img: np.ndarray, brightness: float = 1.1, saturation: float = 1.15, contrast: float = 1.1) -> np.ndarray:
    # Adjust saturation and brightness in HSV, then contrast in RGB
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * saturation, 0, 255)
    v = np.clip(v * brightness, 0, 255)
    hsv = cv2.merge([h, s, v]).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    rgb = np.clip((rgb - 0.5) * contrast + 0.5, 0.0, 1.0)
    return to_uint8(rgb)


def apply_cutout(img: np.ndarray, num_holes: int = 5, max_hole_fraction: float = 0.18) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    hole_h = int(h * max_hole_fraction)
    hole_w = int(w * max_hole_fraction)
    hole_h = max(8, hole_h)
    hole_w = max(8, hole_w)
    mean_color = tuple(int(c) for c in np.mean(out.reshape(-1, 3), axis=0))
    for _ in range(num_holes):
        hh = np.random.randint(hole_h // 2, hole_h)
        ww = np.random.randint(hole_w // 2, hole_w)
        y = np.random.randint(0, max(1, h - hh))
        x = np.random.randint(0, max(1, w - ww))
        out[y:y + hh, x:x + ww, :] = mean_color
    return out


def metric_colorfulness(img: np.ndarray) -> float:
    r, g, b = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)


def metric_contrast(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(np.std(gray))


def metric_sharpness_laplacian(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def metric_dark_channel_mean(img: np.ndarray, patch: int = 15) -> float:
    # Dark Channel Prior proxy (higher mean ⇒ more haze/backscatter)
    min_rgb = np.min(img.astype(np.float32) / 255.0, axis=2)
    k = patch
    kernel = np.ones((k, k), np.uint8)
    dark = cv2.erode(min_rgb, kernel, iterations=1)
    return float(np.mean(dark))


def build_augmentations(img: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "original": img,
        "turbidity": apply_turbidity(img, transmission=0.72, airlight=(0.82, 0.90, 0.96)),
        "pollution": apply_pollution_particles(img, count=600, radius_range=(1, 2), opacity=0.4),
        "color_cast": apply_color_cast(img, gains=(0.88, 1.03, 1.18)),
        "blur_noise": apply_blur_noise(img, blur_ksize=5, noise_std=0.015),
        "motion_blur": apply_motion_blur(img, ksize=9, angle_deg=20.0),
        "speckle": apply_speckle_noise(img, std=0.05),
        "vignette": apply_vignette(img, strength=0.45),
        "color_jitter": apply_color_jitter(img, brightness=1.08, saturation=1.12, contrast=1.08),
        "cutout": apply_cutout(img, num_holes=4, max_hole_fraction=0.16),
        "low_light": apply_low_light(img, gamma=1.9, gain=0.65, noise_std=0.012),
        "jpeg": apply_jpeg_compression(img, quality=20),
    }


def compute_metrics(images: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for name, im in images.items():
        metrics[name] = {
            "contrast": metric_contrast(im),
            "sharpness": metric_sharpness_laplacian(im),
            "colorfulness": metric_colorfulness(im),
            "haze_score": metric_dark_channel_mean(im),
        }
    return metrics


def plot_gallery(sample_sets: List[Dict[str, np.ndarray]], out_path: str = "aug_gallery.png") -> None:
    aug_order = [
        "original",
        "turbidity",
        "pollution",
        "color_cast",
        "blur_noise",
        "motion_blur",
        "speckle",
        "vignette",
        "color_jitter",
        "cutout",
        "low_light",
        "jpeg",
    ]
    rows = len(sample_sets)
    cols = len(aug_order)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    for r, aug_dict in enumerate(sample_sets):
        for c, key in enumerate(aug_order):
            axes[r, c].imshow(aug_dict[key])
            axes[r, c].set_title(key, fontsize=9)
            axes[r, c].axis("off")
    fig.suptitle("Coral robustness augmentations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics_list: List[Dict[str, Dict[str, float]]], out_path: str = "aug_metrics.png") -> None:
    aug_order = [
        "original",
        "turbidity",
        "pollution",
        "color_cast",
        "blur_noise",
        "motion_blur",
        "speckle",
        "vignette",
        "color_jitter",
        "cutout",
        "low_light",
        "jpeg",
    ]
    metric_names = ["contrast", "sharpness", "colorfulness", "haze_score"]
    titles = {
        "contrast": "Contrast (std gray)",
        "sharpness": "Sharpness (var Laplacian)",
        "colorfulness": "Colorfulness (Hasler-Süsstrunk)",
        "haze_score": "Haze score (dark-channel mean)",
    }
    n = len(metrics_list)
    fig, axes = plt.subplots(1, len(metric_names), figsize=(4.2 * len(metric_names), 3.4))
    if len(metric_names) == 1:
        axes = [axes]
    for i, mname in enumerate(metric_names):
        vals = []
        labels = []
        for aug in aug_order:
            labels.append(aug)
            aug_vals = [metrics_list[k][aug][mname] for k in range(n)]
            vals.append(np.mean(aug_vals))
        axes[i].bar(np.arange(len(labels)), vals, color=plt.cm.Set3(np.linspace(0, 1, len(labels))))
        axes[i].set_title(titles[mname])
        axes[i].set_xticks(np.arange(len(labels)))
        axes[i].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        axes[i].grid(True, axis="y", alpha=0.2)
    fig.suptitle("Augmentation impact on simple image metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)




imgs = load_sample_images(IMAGE_DIR)
sample_sets: List[Dict[str, np.ndarray]] = []
metrics_list: List[Dict[str, Dict[str, float]]] = []

for img in imgs:
    aug = build_augmentations(img)
    sample_sets.append(aug)
    metrics_list.append(compute_metrics(aug))

plot_gallery(sample_sets, out_path="aug_gallery.png")
plot_metrics(metrics_list, out_path="aug_metrics.png")
