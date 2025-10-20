import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import os
from collections import namedtuple
import numpy as np
from PIL import Image
from tqdm.autonotebook import tqdm
import torch.nn.functional as F
import json
import copy
from config import __RANDOM_STATE__


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
    def __init__(self, root, split, transform=None, transform_target=True, random_state=__RANDOM_STATE__):
        """
        Initialize the dataset.
        Args:
            transform (callable): A function/transform that takes in an image and returns a transformed version.
            transform_target (bool): Whether to also transform the segmentation mask.
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.transform_target = transform_target
        self.split = split

        self.CoralClass = namedtuple('CoralClass', ['name', 'id', 'train_id', 'category', 'category_id', 'ignore_in_eval', 'color'])

        self.classes = []
        self.classes.append(self.CoralClass('background', 0, 0, 'background', 0, True, (0, 0, 0)))      # Black - background
        self.classes.append(self.CoralClass('bleached', 1, 1, 'bleached', 1, False, (255, 255, 255)))    # White - bleached coral
        self.classes.append(self.CoralClass('non_bleached', 2, 2, 'non_bleached', 2, False, (128, 128, 128)))  # Gray - non-bleached coral

        self.N_classes = 3  # we have 3 classes: background, bleached, non-bleached

        self.id2label = {0: "background", 1: "bleached", 2: "non_bleached"}
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.train_id_to_color = np.array([c.color for c in self.classes])

        self.class_mapping = {
            # Background classes -> 0
            0: 0, 1: 0, 2: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
            # Bleached coral classes -> 1
            3: 1, 4: 1, 16: 1, 19: 1, 33: 1,
            # Non-bleached coral classes -> 2
            17: 2, 18: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2,
            30: 2, 31: 2, 32: 2, 34: 2, 35: 2, 36: 2, 37: 2, 38: 2, 39: 2
        }

        # Create reverse mapping for model output interpretation
        self.reverse_class_mapping = {}
        for original_class, our_class in self.class_mapping.items():
            if our_class not in self.reverse_class_mapping:
                self.reverse_class_mapping[our_class] = []
            self.reverse_class_mapping[our_class].append(original_class)

        # Create train-val-test splits
        self.images_dir = os.path.join(self.root, 'images')
        all_images = [f for f in os.listdir(self.images_dir)
                     if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

        np.random.seed(random_state)
        np.random.shuffle(all_images)

        if split == 'train':
            self.images = all_images[:int(0.7*len(all_images))]
        elif split == 'val':
            self.images = all_images[int(0.7*len(all_images)):int(0.85*len(all_images))]
        elif split == 'test':
            self.images = all_images[int(0.85*len(all_images)):]

        print(f"Loaded {len(self.images)} images for {split} split")

    def __getitem__(self, index):
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
        image_path = os.path.join(self.root, 'images', img_name)

        base_name = os.path.splitext(img_name)[0]  # get the base name without extension to get masks
        image = np.array(Image.open(image_path).convert('RGB'))  # load image

        # Load bleached and non-bleached masks
        bleached_mask_path = os.path.join(self.root, 'masks_bleached', f"{base_name}_bleached.png")
        non_bleached_mask_path = os.path.join(self.root, 'masks_non_bleached', f"{base_name}_non_bleached.png")

        try:
            bleached_mask = np.array(Image.open(bleached_mask_path).convert('L'))
            non_bleached_mask = np.array(Image.open(non_bleached_mask_path).convert('L'))
        except FileNotFoundError as e:
            print(f"Error loading mask for {img_name}: {e}")
            raise

        # Create a combined mask where 0 = background, 1 = bleached, 2 = non-bleached
        target = np.zeros_like(bleached_mask, dtype=np.uint8)
        target[bleached_mask > 128] = 1  # white pixels in bleached mask -> class 1
        target[non_bleached_mask > 128] = 2  # white pixels in non-bleached mask -> class 2
        # background remains 0 (black pixels)

        # Resize image
        target_size = (512, 512)
        image_pil = Image.fromarray(image)
        image_resized = image_pil.resize(target_size)
        image = np.array(image_resized)

        # Resize mask
        target_pil = Image.fromarray(target)
        target_resized = target_pil.resize(target_size, Image.NEAREST)  # Use NEAREST for masks
        target = np.array(target_resized)

        if self.transform:
            if self.transform_target:
                transformed = self.transform(image=image, mask=target)
                image = transformed["image"].transpose(2, 0, 1)
                target = transformed["mask"]
            else:
                transformed = self.transform(image=image)
                image = transformed["image"].transpose(2, 0, 1)

        return image, target

    def __len__(self):
        return len(self.images)


def calculate_iou(pred, target, num_classes=3):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float().item()
        union = (pred_inds | target_inds).sum().float().item()
        iou = intersection / union if union > 0 else 0
        ious.append(iou)

    return ious, np.mean(ious)


def train_and_validate_kfold(train_loader, val_loader, benchmark_run, criterion, n_splits, start_epoch, end_epoch,
                             save_dir, save_best_only) -> dict[str, float | dict]:
    """
    Perform k-fold cross-validation training and validation.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        benchmark_run: Object containing model, optimizer, device, etc.
        criterion: Loss function
        n_splits: Number of folds for k-fold cross-validation
        logger: Optional logger for metrics
        start_epoch: Starting epoch number
        end_epoch: Ending epoch number
        save_dir: Directory to save model checkpoints
        save_best_only: If True, only save model when validation performance improves
    """
    if end_epoch is None:
        end_epoch = benchmark_run.training_hyperparameters.epochs

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Store metrics for each fold
    fold_metrics = []

    # Get the dataset from the train loader
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    # Combine datasets for k-fold splitting
    # If they're the same dataset with different samplers, just use one
    if train_dataset == val_dataset:
        full_dataset = train_dataset
    else:
        # This assumes both datasets have the same structure and can be concatenated
        try:
            full_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        except:
            print("Warning: Could not concatenate datasets. Using training dataset for k-fold.")
            full_dataset = train_dataset

    # Initialize k-fold cross validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Get batch size and num_workers from original loaders
    batch_size = train_loader.batch_size
    num_workers = train_loader.num_workers

    # Track the best overall model across all folds based on the custom score
    best_overall_score = -1.0
    best_overall_fold = -1
    best_overall_epoch = -1
    best_overall_metrics = None
    best_overall_model_state_dict = None

    # Loop through each fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_dataset)))):
        print(f"\n{'='*20} FOLD {fold+1}/{n_splits} {'='*20}")

        # Create data loaders for this fold
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        fold_train_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=train_subsampler,
            num_workers=num_workers
        )

        fold_val_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=val_subsampler,
            num_workers=num_workers
        )

        # Reset model weights for each fold if possible
        if hasattr(benchmark_run.model, 'reset_parameters'):
            benchmark_run.model.reset_parameters()
        else:
            print("Warning: Model does not have reset_parameters method. Using current weights.")

        # Reset optimizer for the new model parameters
        if hasattr(benchmark_run, 'optimizer_class') and hasattr(benchmark_run, 'optimizer_params'):
            benchmark_run.optimizer = benchmark_run.optimizer_class(
                benchmark_run.model.parameters(),
                **benchmark_run.optimizer_params
            )

        # Initialize metrics for this fold
        fold_benchmark_metrics = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_miou": [],
            "val_iou_background": [],
            "val_iou_bleached": [],
            "val_iou_non_bleached": [],
            "val_custom_score": []
        }

        # Track best performance for this fold
        best_fold_score = -1.0

        # Train for specified number of epochs
        for epoch in range(start_epoch, end_epoch):
            print(f"EPOCH {epoch+1}/{end_epoch}:")

            # Train
            benchmark_run.model.train()
            train_loss = 0.0
            num_batches = 0

            for data in tqdm(fold_train_loader, desc="Training"):
                # Move data to device
                inputs, labels = data
                inputs = inputs.to(benchmark_run.device)
                labels = labels.to(benchmark_run.device)

                # Zero the parameter gradients
                benchmark_run.optimizer.zero_grad()

                # Forward pass
                outputs = benchmark_run.model(inputs)

                # Resize model output (logits) to match target size
                resized_logits = F.interpolate(
                    outputs.logits,
                    size=labels.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

                # Calculate loss
                loss = criterion(resized_logits, labels.long())

                # Backward pass and optimize
                loss.backward()
                benchmark_run.optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            if num_batches > 0:
                train_loss /= num_batches

            fold_benchmark_metrics["train_loss"].append(train_loss)

            if hasattr(benchmark_run, "scheduler"):
                benchmark_run.scheduler.step()

            print(f'LOSS train {train_loss:.4f}')

            # Validation
            benchmark_run.model.eval()
            val_loss = 0.0
            num_batches_val = 0
            val_correct = 0
            val_total = 0
            val_ious = np.zeros(3)  # For 3 classes

            with torch.no_grad():
                for vdata in tqdm(fold_val_loader, desc="Validating"):
                    inputs, labels = vdata
                    inputs = inputs.to(benchmark_run.device)
                    labels = labels.to(benchmark_run.device)

                    # Forward pass
                    outputs = benchmark_run.model(inputs)

                    # Resize model output (logits) to match target size
                    resized_logits = F.interpolate(
                        outputs.logits,
                        size=labels.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )

                    # Calculate loss
                    loss = criterion(resized_logits, labels.long())

                    # Calculate accuracy
                    preds = torch.argmax(resized_logits, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.numel()

                    # Calculate IoU
                    batch_ious, _ = calculate_iou(preds, labels)
                    val_ious += np.array(batch_ious)

                    val_loss += loss.item()
                    num_batches_val += 1

            if num_batches_val > 0:
                val_loss /= num_batches_val
                val_ious /= num_batches_val

            val_accuracy = val_correct / val_total if val_total > 0 else 0
            val_miou = np.mean(val_ious)

            # Calculate custom ranking score
            # rank = alpha * accuracy + beta * mIoU
            # mIoU = 0.1 * iou_background + 0.45 * iou_bleached + 0.45 * iou_non_bleached
            alpha = 0.2
            beta = 0.8
            miou_weighted = (0.1 * val_ious[0] + 0.45 * val_ious[1] + 0.45 * val_ious[2])
            custom_score = alpha * val_accuracy + beta * miou_weighted

            # Store metrics
            fold_benchmark_metrics["val_loss"].append(val_loss)
            fold_benchmark_metrics["val_accuracy"].append(val_accuracy)
            fold_benchmark_metrics["val_miou"].append(val_miou)
            fold_benchmark_metrics["val_iou_background"].append(val_ious[0])
            fold_benchmark_metrics["val_iou_bleached"].append(val_ious[1])
            fold_benchmark_metrics["val_iou_non_bleached"].append(val_ious[2])
            fold_benchmark_metrics["val_custom_score"].append(custom_score)

            print(f'LOSS valid {val_loss:.4f}')
            print(f'Accuracy valid {val_accuracy:.4f}')
            print(f'mIoU valid {val_miou:.4f}')
            print(f'IoU per class: Background={val_ious[0]:.4f}, Bleached={val_ious[1]:.4f}, Non-bleached={val_ious[2]:.4f}')
            print(f'Custom Score valid {custom_score:.4f}')

            # Save best model for this fold based on custom score
            if save_best_only:
                if custom_score > best_fold_score:
                    best_fold_score = custom_score
                    checkpoint_path = os.path.join(save_dir, f"model_fold_{fold+1}_best_score.pth")
                    torch.save(benchmark_run.model.state_dict(), checkpoint_path)
                    print(f"Saved improved model checkpoint to {checkpoint_path} (Custom Score: {custom_score:.4f})")

            # Check if this is the best overall model
            if custom_score > best_overall_score:
                best_overall_score = custom_score
                best_overall_fold = fold + 1
                best_overall_epoch = epoch + 1
                best_overall_metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_miou": val_miou,
                    "val_iou_background": val_ious[0],
                    "val_iou_bleached": val_ious[1],
                    "val_iou_non_bleached": val_ious[2],
                    "val_custom_score": custom_score
                }
                best_overall_model_state_dict = copy.deepcopy(benchmark_run.model.state_dict())


        # Save metrics for this fold
        fold_metrics.append(fold_benchmark_metrics)

        # Save fold metrics
        metrics_path = os.path.join(save_dir, f"metrics_fold_{fold+1}.json")
        with open(metrics_path, "w") as f:
            json.dump(fold_benchmark_metrics, f)

    # Calculate and save average metrics across all folds
    avg_metrics = {
        "avg_val_miou_final": np.mean([metrics["val_miou"][-1] for metrics in fold_metrics]),
        "avg_val_accuracy_final": np.mean([metrics["val_accuracy"][-1] for metrics in fold_metrics]),
        "avg_val_iou_background_final": np.mean([metrics["val_iou_background"][-1] for metrics in fold_metrics]),
        "avg_val_iou_bleached_final": np.mean([metrics["val_iou_bleached"][-1] for metrics in fold_metrics]),
        "avg_val_iou_non_bleached_final": np.mean([metrics["val_iou_non_bleached"][-1] for metrics in fold_metrics]),
        "avg_val_custom_score_final": np.mean([metrics["val_custom_score"][-1] for metrics in fold_metrics]),
        "best_val_miou_across_folds": np.max([np.max(metrics["val_miou"]) for metrics in fold_metrics]),
        "best_val_custom_score_across_folds": np.max([np.max(metrics["val_custom_score"]) for metrics in fold_metrics]),
        "fold_metrics": fold_metrics,
        "best_overall_model": {
            "fold": best_overall_fold,
            "epoch": best_overall_epoch,
            "metrics": best_overall_metrics
        }
    }

    # Save the best overall model checkpoint
    if best_overall_model_state_dict is not None:
        best_overall_checkpoint_path = os.path.join(save_dir, "best_overall_model.pth")
        torch.save(best_overall_model_state_dict, best_overall_checkpoint_path)
        print(f"\nSaved best overall model checkpoint to {best_overall_checkpoint_path} (Custom Score: {best_overall_score:.4f})")


    # Save overall metrics
    overall_metrics_path = os.path.join(save_dir, "kfold_metrics.json")
    with open(overall_metrics_path, "w") as f:
        json.dump(avg_metrics, f)

    print(f"\nAverage Final mIoU across folds: {avg_metrics['avg_val_miou_final']:.4f}")
    print(f"Best mIoU across all folds: {avg_metrics['best_val_miou_across_folds']:.4f}")
    print(f"Average Final Custom Score across folds: {avg_metrics['avg_val_custom_score_final']:.4f}")
    print(f"Best Custom Score across all folds: {avg_metrics['best_val_custom_score_across_folds']:.4f}")
    print(f"\nBest overall model found in Fold {best_overall_fold}, Epoch {best_overall_epoch} with Custom Score: {best_overall_score:.4f}")
    print("="*50)

    return avg_metrics
