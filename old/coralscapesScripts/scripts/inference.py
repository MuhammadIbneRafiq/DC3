import sys
import os
# Add the parent directory to Python path to find coralscapesScripts module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch   # pyright: ignore[reportMissingImports]
import albumentations as A  # pyright: ignore[reportMissingImports]
import torch.nn.functional as F  # pyright: ignore[reportMissingImports]
from coralscapesScripts_old.segmentation.model import Benchmark_Run
from coralscapesScripts_old.io import setup_config, get_parser, update_config_with_args

from coralscapesScripts_old.datasets.preprocess import preprocess_inference
from coralscapesScripts_old.segmentation.model import predict
import glob
import os
import numpy as np  # pyright: ignore[reportMissingImports]
from PIL import Image  # pyright: ignore[reportMissingImports]

def resize_image(image, target_size=1024):
    h_img, w_img = image.size
    if h_img < w_img:
        new_h, new_w = target_size, int(w_img * (target_size / h_img))
    else:
        new_h, new_w  = int(h_img * (target_size / w_img)), target_size
    
    resized_img = image.resize((new_w, new_h))  # PIL expects (width, height)
    return resized_img

def segment_image(image, preprocessor, model, crop_size = (1024, 1024), num_classes = 40, transform=None):    
    h_crop, w_crop = crop_size
    resized_img = resize_image(image, target_size=min(crop_size))
    img_array = np.array(resized_img)
    # img_array should be (height, width, channels)
    # transpose to (channels, height, width)
    img_transposed = img_array.transpose(2, 0, 1)
    img = torch.Tensor(img_transposed).unsqueeze(0)
    batch_size, _, h_img, w_img = img.size()

    h_grids = int(np.round(3/2*h_img/h_crop)) if h_img > h_crop else 1
    w_grids = int(np.round(3/2*w_img/w_crop)) if w_img > w_crop else 1
    
    h_stride = int((h_img - h_crop + h_grids -1)/(h_grids -1)) if h_grids > 1 else h_crop
    w_stride = int((w_img - w_crop + w_grids -1)/(w_grids -1)) if w_grids > 1 else w_crop
    
    preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
    
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            # Ensure we don't go out of bounds
            if y2 - y1 < h_crop:
                y1 = max(0, y2 - h_crop)
            if x2 - x1 < w_crop:
                x1 = max(0, x2 - w_crop)
            
            crop_img = img[:, :, y1:y2, x1:x2]
            print(f"Crop shape: {crop_img.shape}")

            if transform:
                crop_img = torch.Tensor(transform(image = crop_img.squeeze(0).numpy())["image"]).unsqueeze(0)  

            with torch.no_grad():
                if(preprocessor):
                    # Convert tensor back to PIL Image for preprocessor
                    crop_squeezed = crop_img[0]  # Use indexing instead of squeeze
                    print(f"Crop squeezed shape: {crop_squeezed.shape}")
                    crop_permuted = crop_squeezed.permute(1, 2, 0)  # Should be (1024, 1024, 3)
                    print(f"Crop permuted shape: {crop_permuted.shape}")
                    crop_img_np = crop_permuted.numpy().astype(np.uint8)
                    print(f"Crop numpy shape: {crop_img_np.shape}")
                    crop_img_pil = Image.fromarray(crop_img_np)
                    inputs = preprocessor(crop_img_pil, return_tensors = "pt")
                    inputs["pixel_values"] = inputs["pixel_values"].to(device)
                    outputs = model(**inputs)
                else:
                    inputs = crop_img.to(device)
                    outputs = model(pixel_values=inputs)

            if(hasattr(outputs, "logits")): 
                outputs = outputs.logits

            resized_logits = F.interpolate(
                outputs[0].unsqueeze(dim=0), size=crop_img.shape[-2:], mode="bilinear", align_corners=False
            )

            preds += F.pad(resized_logits,
                            (int(x1), int(preds.shape[3] - x2), int(y1),
                            int(preds.shape[2] - y2))).cpu()
            count_mat[:, :, y1:y2, x1:x2] += 1
    
    assert (count_mat == 0).sum() == 0
    preds = preds / count_mat
    
    preds = preds.argmax(dim=1)
    
    preds = F.interpolate(preds.unsqueeze(0).type(torch.uint8), size=image.size[::-1], mode='nearest')
    label_pred = preds.squeeze().cpu().numpy()
    
    return label_pred


# device_count = torch.cuda.device_count()
# for i in range(device_count):
#     print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

id2label = {"0": "unlabeled", "1": "seagrass", "2": "trash", "3": "other coral dead", "4": "other coral bleached", "5": "sand", "6": "other coral alive", "7": "human", "8": "transect tools", "9": "fish", "10": "algae covered substrate", "11": "other animal", "12": "unknown hard substrate", "13": "background", "14": "dark", "15": "transect line", "16": "massive/meandering bleached", "17": "massive/meandering alive", "18": "rubble", "19": "branching bleached", "20": "branching dead", "21": "millepora", "22": "branching alive", "23": "massive/meandering dead", "24": "clam", "25": "acropora alive", "26": "sea cucumber", "27": "turbinaria", "28": "table acropora alive", "29": "sponge", "30": "anemone", "31": "pocillopora alive", "32": "table acropora dead", "33": "meandering bleached", "34": "stylophora alive", "35": "sea urchin", "36": "meandering alive", "37": "meandering dead", "38": "crown of thorn", "39": "dead clam"}
label2color = {"unlabeled": [255, 255, 255], "human": [255, 0, 0], "background": [29, 162, 216], "fish": [255, 255, 0], "sand": [194, 178, 128], "rubble": [161, 153, 128], "unknown hard substrate": [125, 125, 125], "algae covered substrate": [125, 163, 125], "dark": [31, 31, 31], "branching bleached": [252, 231, 240], "branching dead": [123, 50, 86], "branching alive": [226, 91, 157], "stylophora alive": [255, 111, 194], "pocillopora alive": [255, 146, 150], "acropora alive": [236, 128, 255], "table acropora alive": [189, 119, 255], "table acropora dead": [85, 53, 116], "millepora": [244, 150, 115], "turbinaria": [228, 255, 119], "other coral bleached": [250, 224, 225], "other coral dead": [114, 60, 61], "other coral alive": [224, 118, 119], "massive/meandering alive": [236, 150, 21], "massive/meandering dead": [134, 86, 18], "massive/meandering bleached": [255, 248, 228], "meandering alive": [230, 193, 0], "meandering dead": [119, 100, 14], "meandering bleached": [251, 243, 216], "transect line": [0, 255, 0], "transect tools": [8, 205, 12], "sea urchin": [0, 142, 255], "sea cucumber": [0, 231, 255], "anemone": [0, 255, 189], "sponge": [240, 80, 80], "clam": [189, 255, 234], "other animal": [0, 255, 255], "trash": [255, 0, 134], "seagrass": [125, 222, 125], "crown of thorn": [179, 245, 234], "dead clam": [89, 155, 134]}
id2color = {int(k): label2color[v] for k, v in id2label.items()}

parser = get_parser()
args = parser.parse_args()

cfg_base_path = 'configs/base.yaml'
cfg = setup_config(args.config, cfg_base_path)
cfg = update_config_with_args(cfg, args)


transform = A.Compose([
    getattr(A, transform_name)(**transform_params) for transform_name, 
    transform_params in cfg.augmentation["test"].items()
])

benchmark_run = Benchmark_Run(run_name = cfg.run_name, model_name = cfg.model.name, 
                                    N_classes = len(id2label), device= device, 
                                    model_kwargs = cfg.model.kwargs,
                                    model_checkpoint = cfg.model.checkpoint,
                                    lora_kwargs = cfg.lora,
                                    training_hyperparameters = cfg.training)
benchmark_run.model.to(device)
benchmark_run.model.eval()

input_dir = args.inputs
output_dir = args.outputs

# Get the list of image paths
image_paths = glob.glob(f'{input_dir}/*.png') + glob.glob(f'{input_dir}/*.jpg') + glob.glob(f'{input_dir}/*.jpeg')

for image_path in image_paths:
    print(f"Processing image: {image_path}")
    image = Image.open(image_path)
    
    # Simple approach: resize to 1024x1024 and process directly
    image_resized = image.resize((1024, 1024))
    print(f"Resized image size: {image_resized.size}")
    
    # Process with preprocessor
    if benchmark_run.preprocessor:
        inputs = benchmark_run.preprocessor(image_resized, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(device)
        
        with torch.no_grad():
            outputs = benchmark_run.model(**inputs)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
                preds = logits.argmax(dim=1)
                
                # Convert back to numpy
                label_pred = preds.squeeze().cpu().numpy()
                
                # Resize back to original image size
                label_pred_resized = F.interpolate(
                    torch.tensor(label_pred).unsqueeze(0).unsqueeze(0).float(), 
                    size=image.size[::-1], 
                    mode='nearest'
                ).squeeze().numpy().astype(np.uint8)
                
                label_pred_colors = np.array(
                    [[id2color[pixel] for pixel in row] for row in label_pred_resized]
                )
                mask_image = Image.fromarray(label_pred_resized.astype(np.uint8))
                mask_image_colors = Image.fromarray(label_pred_colors.astype(np.uint8), 'RGB')
                overlay = Image.blend(
                    image.convert("RGBA"), mask_image_colors.convert("RGBA"), alpha=0.6)

                # Create output filename
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                mask_image.save(f"{output_dir}/{base_name}_pred.png")
                overlay.save(f"{output_dir}/{base_name}_overlay.png")
                print(f"Saved results for {base_name}")
            else:
                print("No logits in model output")
    else:
        print("No preprocessor available")

print(f"Processed {len(image_paths)} images successfully!")