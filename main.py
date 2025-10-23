import albumentations as A
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from coralscapesScripts.utils import calculate_weights
from coralscapesScripts.model import Benchmark_Run
from coralscapesScripts.io import setup_config, get_parser, update_config_with_args
from image_preprocessing import ImagePreprocessing
from train_functions import CoralBleachingDataset, train_and_validate_kfold, color_cluster
from config import __RANDOM_STATE__, __EPOCHS__, __n_SPLITS__, root, input_path, output_path


torch.manual_seed(__RANDOM_STATE__)
torch.cuda.manual_seed(__RANDOM_STATE__)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

processed_images = ImagePreprocessing(input_path, output_path)
processed_images.cluster_images()  # cluster images
processed_images.apply_filters()  # apply CMY filters to each cluster
color, cluster = color_cluster(root)  # to name the checkpoints folder

device_count = torch.cuda.device_count()  # connect to GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
for i in range(device_count):
    print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")


def main():
    args = get_parser().parse_args(f"--run-name=fine_tune --batch-size=1 --batch-size-eval=1 --epochs={__EPOCHS__}".split(" "))
    cfg = setup_config(config_path='configs/segformer-mit-b2_lora.yaml', config_base_path='configs/base.yaml')
    cfg = update_config_with_args(cfg, args)

    transforms = {}
    for split in cfg.augmentation:
        transform_list = [A.Resize(height=512, width=512)]  # SegFormer model uses this input size

        for transform_name, transform_params in cfg.augmentation[split].items():
            if "Crop" in transform_name and "height" in transform_params and "width" in transform_params:  # skipping any crop operations that would be larger than our resize
                if transform_params["height"] > 512 or transform_params["width"] > 512:
                    print(f"Warning: Adjusting '{transform_name}' to match 512x512 size in split '{split}'")
                    transform_params["height"] = min(transform_params["height"], 512)
                    transform_params["width"] = min(transform_params["width"], 512)

            transform_list.append(getattr(A, transform_name)(**transform_params))
        transforms[split] = A.Compose(transform_list)

    cfg.data.root = root
    transform_target = cfg.training.eval.transform_target if cfg.training.eval and cfg.training.eval.transform_target else True

    train_dataset = CoralBleachingDataset(root=cfg.data.root, split='train', transform=transforms["train"])
    val_dataset = CoralBleachingDataset(root=cfg.data.root, split='val', transform=transforms["val"], transform_target=transform_target)
    test_dataset = CoralBleachingDataset(root=cfg.data.root, split='test', transform=transforms["test"], transform_target=transform_target)

    print(f"Classes: {train_dataset.id2label}\nNumber of classes: {train_dataset.N_classes}\n"
          f"Train dataset has {len(train_dataset)} samples\nValidation dataset has {len(val_dataset)} samples\n"
          f"Test dataset has {len(test_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size_eval, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size_eval, shuffle=False, num_workers=0)

    benchmark_run = Benchmark_Run(run_name=cfg.run_name, model_name=cfg.model.name, N_classes=processed_images.optimal_k,
                                  device=device, model_kwargs=cfg.model.kwargs, model_checkpoint=cfg.model.checkpoint,
                                  lora_kwargs=cfg.lora, training_hyperparameters=cfg.training)

    weight = calculate_weights(train_dataset).to(device) if cfg.data.weight else None
    criterion = nn.CrossEntropyLoss(weight=weight)  # loss function

    # fine-tune model
    train_and_validate_kfold(train_loader=train_loader, val_loader=val_loader, benchmark_run=benchmark_run,
                             criterion=criterion, n_splits=__n_SPLITS__, start_epoch=0, end_epoch=__EPOCHS__,
                             save_dir=f"./checkpoints_{color}_{cluster}", save_best_only=True, random_state=__RANDOM_STATE__)

if __name__ == "__main__":
    main()
