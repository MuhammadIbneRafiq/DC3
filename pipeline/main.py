import albumentations as A
import torch.nn as nn
from torch.utils.data import DataLoader
from coralscapesScripts.utils import calculate_weights
from coralscapesScripts.model import Benchmark_Run
from coralscapesScripts.io import setup_config, get_parser, update_config_with_args
from train_functions import *
from config import __RANDOM_STATE__, __EPOCHS__, __n_SPLITS__


# TODO: specify paths
root = "data_yellow_0"
root_copy = root
color = "yellow"  # these are to name the checkpoints folder
cluster = "0"

torch.manual_seed(__RANDOM_STATE__)
torch.cuda.manual_seed(__RANDOM_STATE__)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device_count = torch.cuda.device_count()  # connect to GPU
for i in range(device_count):
    print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

cfg = setup_config(config_path='configs/segformer-mit-b2_lora.yaml', config_base_path='configs/base.yaml')

args_input = f"--run-name=fine_tune_notebook --batch-size=1 --batch-size-eval=1 --epochs={__EPOCHS__}"
args_input = args_input.split(" ")

parser = get_parser()
args = parser.parse_args(args_input)

cfg = update_config_with_args(cfg, args)
cfg_logger = copy.deepcopy(cfg)

# for root, dirs, files in os.walk('/content/drive/MyDrive/Data Challenge 3 Group 3/01_data/coral_bleaching/reef_support/UNAL_BLEACHING_TAYRONA'):
for root, dirs, files in os.walk(root):
    for name in dirs:
        print(os.path.join(root, name), 'idoes it exist?')



transforms = {}
for split in cfg.augmentation:
    transform_list = [A.Resize(height=512, width=512)] # start with resize to ensure consistent size

    for transform_name, transform_params in cfg.augmentation[split].items():
        if "Crop" in transform_name and "height" in transform_params and "width" in transform_params:# skipping any crop operations that would be larger than our resize
            if transform_params["height"] > 512 or transform_params["width"] > 512:
                print(f"Warning: Adjusting '{transform_name}' to match 512x512 size in split '{split}'")
                transform_params["height"] = min(transform_params["height"], 512)
                transform_params["width"] = min(transform_params["width"], 512)

        transform_list.append(getattr(A, transform_name)(**transform_params))

    transforms[split] = A.Compose(transform_list)

cfg.data.root = root_copy

train_dataset = CoralBleachingDataset(root=cfg.data.root, split='train', transform=transforms["train"])

transform_target = cfg.training.eval.transform_target if cfg.training.eval and cfg.training.eval.transform_target else True
val_dataset = CoralBleachingDataset(root=cfg.data.root, split='val', transform=transforms["val"], transform_target=transform_target)
test_dataset = CoralBleachingDataset(root=cfg.data.root, split='test', transform=transforms["test"], transform_target=transform_target)

print(f"Classes: {train_dataset.id2label}")
print(f"Number of classes: {train_dataset.N_classes}")
print(f"Train dataset has {len(train_dataset)} samples")
print(f"Validation dataset has {len(val_dataset)} samples")
print(f"Test dataset has {len(test_dataset)} samples")

train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size_eval, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size_eval, shuffle=False, num_workers=0)

print(len(train_loader), len(val_loader), len(test_loader))

weight = calculate_weights(train_dataset).to(device) if(cfg.data.weight) else None

benchmark_run = Benchmark_Run(run_name=cfg.run_name,
                             model_name=cfg.model.name,
                             N_classes=3,
                             device=device,
                             model_kwargs=cfg.model.kwargs,
                             model_checkpoint=cfg.model.checkpoint,
                             lora_kwargs=cfg.lora,
                             training_hyperparameters=cfg.training)

benchmark_run.print_trainable_parameters()

data_batch = next(iter(train_loader))
labels_batch = data_batch[1] # Access labels using index 1

print("Shape of labels batch:", labels_batch.shape)
print("Unique values in labels batch:", torch.unique(labels_batch))

criterion = nn.CrossEntropyLoss(weight=weight) # Use the calculated weight
save_directory = "./custom_checkpoints"
save_frequency_epochs = 1 # Set to an integer to save checkpoints

# fine-tune model
obtain_metrics = train_and_validate_kfold(
    train_loader=train_loader,
    val_loader=val_loader,
    benchmark_run=benchmark_run,
    criterion=criterion,
    n_splits=__n_SPLITS__,
    start_epoch=0,
    end_epoch=__EPOCHS__,
    save_dir=f"./checkpoints_{color}_{cluster}",
    save_best_only=True
)
