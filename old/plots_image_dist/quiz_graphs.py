import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def count_image_files(root: Path, extensions: Tuple[str, ...]) -> int:
	"""Recursively count image files under root matching extensions."""
	if not root.exists():
		return 0
	count = 0
	for dirpath, _dirnames, filenames in os.walk(root):
		for filename in filenames:
			lower = filename.lower()
			if any(lower.endswith(ext) for ext in extensions):
				count += 1
	return count


def count_images_per_immediate_subfolder(root: Path, extensions: Tuple[str, ...]) -> Dict[str, int]:
	"""Count images grouped by immediate subfolder under root (recursive within each subfolder)."""
	counts: Dict[str, int] = {}
	if not root.exists():
		return counts
	for entry in sorted(root.iterdir()):
		if entry.is_dir():
			counts[entry.name] = count_image_files(entry, extensions)
	return counts


def kfold_partition_counts(total: int, k: int) -> List[int]:
	"""Distribute total samples into k folds as evenly as possible."""
	if k <= 0:
		raise ValueError("k must be positive")
	base = total // k
	remainder = total % k
	return [base + (1 if i < remainder else 0) for i in range(k)]


def plot_counts_bar(counts: Dict[str, int], output_path: Path) -> None:
	labels = list(counts.keys())
	values = [counts[label] for label in labels]
	fig, ax = plt.subplots(figsize=(7.5, 4))
	bars = ax.bar(labels, values, color=["#1f77b4", "#ff7f0e"])  # blue, orange
	ax.set_title("Image Counts by Dataset")
	ax.set_ylabel("Number of images")
	for bar, value in zip(bars, values):
		height = bar.get_height()
		ax.text(bar.get_x() + bar.get_width() / 2, height, str(value), ha="center", va="bottom")
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def plot_kfold_bars(dataset_to_counts: Dict[str, List[int]], output_path: Path) -> None:
	"""Plot per-fold counts for each dataset (grouped bars)."""
	datasets = list(dataset_to_counts.keys())
	k = len(next(iter(dataset_to_counts.values()), []))
	if k == 0:
		raise ValueError("k-fold counts must be non-empty")

	fig, ax = plt.subplots(figsize=(10, 4))

	# Grouped bars: for each fold, plot bars for each dataset side-by-side
	indices = list(range(k))
	width = 0.35 if len(datasets) == 2 else 0.8 / max(1, len(datasets))
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

	for idx, dataset in enumerate(datasets):
		offset = (idx - (len(datasets) - 1) / 2) * width
		positions = [i + offset for i in indices]
		bars = ax.bar(positions, dataset_to_counts[dataset], width=width, label=dataset, color=colors[idx % len(colors)])
		for bar, value in zip(bars, dataset_to_counts[dataset]):
			height = bar.get_height()
			ax.text(bar.get_x() + bar.get_width() / 2, height, str(value), ha="center", va="bottom", fontsize=8)

	ax.set_xticks(indices)
	ax.set_xticklabels([f"Fold {i+1}" for i in indices])
	ax.set_ylabel("Images per fold")
	ax.set_title("5-Fold Cross-Validation Split Sizes")
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def plot_per_folder_counts(per_folder: Dict[str, Dict[str, int]], output_path: Path) -> None:
	"""Plot horizontal bars for counts per immediate subfolder across datasets.

	per_folder: {dataset_label: {subfolder_name: count}}
	"""
	# Flatten with prefixed labels like "others/<sub>" and "mask_labels/<sub>"
	labels: List[str] = []
	values: List[int] = []
	for dataset_label, folder_counts in per_folder.items():
		for sub, cnt in folder_counts.items():
			labels.append(f"{dataset_label}/{sub}")
			values.append(cnt)

	# Sort descending for readability
	pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
	labels, values = zip(*pairs) if pairs else ([], [])

	n = len(labels)
	fig_height = max(3.0, min(0.3 * n + 1.5, 16))
	fig, ax = plt.subplots(figsize=(10, fig_height))
	ax.barh(range(n), values, color="#4e79a7")
	ax.set_yticks(range(n))
	ax.set_yticklabels(labels, fontsize=8)
	ax.invert_yaxis()
	ax.set_xlabel("Number of images")
	ax.set_title("Image Counts per Immediate Subfolder (others/…, mask_labels/…)")

	for i, v in enumerate(values):
		ax.text(v + max(values) * 0.005 if values else 0.5, i, str(v), va="center", fontsize=8)

	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def plot_single_root_subfolders(title: str, folder_counts: Dict[str, int], output_path: Path) -> None:
	"""Horizontal bar chart for one dataset's immediate subfolders."""
	if not folder_counts:
		fig, ax = plt.subplots(figsize=(8, 3))
		ax.text(0.5, 0.5, "No subfolders found", ha="center", va="center")
		ax.axis("off")
		fig.savefig(output_path, dpi=150)
		plt.close(fig)
		return

	pairs = sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)
	labels, values = zip(*pairs)
	n = len(labels)
	fig_height = max(3.0, min(0.3 * n + 1.5, 16))
	fig, ax = plt.subplots(figsize=(10, fig_height))
	ax.barh(range(n), values, color="#59a14f")
	ax.set_yticks(range(n))
	ax.set_yticklabels(labels, fontsize=8)
	ax.invert_yaxis()
	ax.set_xlabel("Number of images")
	ax.set_title(title)
	for i, v in enumerate(values):
		ax.text(v + max(values) * 0.005 if values else 0.5, i, str(v), va="center", fontsize=8)
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def write_counts_csv(path: Path, folder_counts: Dict[str, int]) -> None:
	"""Write counts to a simple CSV file with columns: folder,count."""
	lines = ["folder,count\n"]
	for folder, cnt in sorted(folder_counts.items(), key=lambda x: x[1], reverse=True):
		lines.append(f"{folder},{cnt}\n")
	path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
	project_root = Path(__file__).resolve().parent
	others_dir = project_root / "data" / "others" / "content"
	mask_labels_dir = project_root / "data" / "mask_labels" / "content"

	extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

	others_count = count_image_files(others_dir, extensions)
	mask_labels_count = count_image_files(mask_labels_dir, extensions)

	# Clarify labels with dataset sources
	counts = {
		"others (data/others/content)": others_count,
		"mask_labels (data/mask_labels/content)": mask_labels_count,
	}

	# Plot 1: total counts with clearer labels
	counts_png = project_root / "quiz_counts.png"
	plot_counts_bar(counts, counts_png)

	# Plot 2: 5-fold CV sizes per dataset (use short keys in legend)
	k = 5
	cv_counts = {
		"others": kfold_partition_counts(others_count, k),
		"mask_labels": kfold_partition_counts(mask_labels_count, k),
	}
	cv_png = project_root / "quiz_crossval.png"
	plot_kfold_bars(cv_counts, cv_png)

	# Plot 3: combined per-folder counts (both datasets together)
	others_folders = count_images_per_immediate_subfolder(others_dir, extensions)
	mask_labels_folders = count_images_per_immediate_subfolder(mask_labels_dir, extensions)
	per_folder_png = project_root / "quiz_counts_per_folder.png"
	plot_per_folder_counts({"others": others_folders, "mask_labels": mask_labels_folders}, per_folder_png)

	# Plot 4: separate per-folder plots
	others_only_png = project_root / "quiz_counts_per_folder_others.png"
	mask_only_png = project_root / "quiz_counts_per_folder_mask_labels.png"
	plot_single_root_subfolders("Others: Counts per Immediate Subfolder (data/others/content)", others_folders, others_only_png)
	plot_single_root_subfolders("Mask Labels: Counts per Immediate Subfolder (data/mask_labels/content)", mask_labels_folders, mask_only_png)

	# Plot 5: specific 9 folders under coral_bleaching/others
	others_specific_dir = others_dir / "gdrive" / "MyDrive" / "Data Challenge 3 - JBG060 AY2526" / "01_data" / "coral_bleaching" / "others"
	others_9_counts = count_images_per_immediate_subfolder(others_specific_dir, extensions)
	others_9_png = project_root / "quiz_counts_others_9_folders.png"
	plot_single_root_subfolders("Others: 9 Coral Bleaching Folders (images per folder)", others_9_counts, others_9_png)
	others_9_csv = project_root / "quiz_counts_others_9_folders.csv"
	write_counts_csv(others_9_csv, others_9_counts)

	print("Counts (totals):", {"others": others_count, "mask_labels": mask_labels_count})
	print(f"Saved plot: {counts_png}")
	print(f"Saved plot: {cv_png}")
	print(f"Saved plot: {per_folder_png}")
	print(f"Saved plot: {others_only_png}")
	print(f"Saved plot: {mask_only_png}")
	print(f"Saved plot: {others_9_png}")
	print(f"Saved csv:  {others_9_csv}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
