import os
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score

similar_folder = 'similar_images'
dissimilar_folder = 'dissimilar_images'
output_folder = 'inference_results'
masks_bleached_folder = r'data\reef_support\content\gdrive\MyDrive\Data Challenge 3 - JBG060 AY2526\01_data\coral_bleaching\reef_support\UNAL_BLEACHING_TAYRONA\masks_bleached'
masks_non_bleached_folder = r'data\reef_support\content\gdrive\MyDrive\Data Challenge 3 - JBG060 AY2526\01_data\coral_bleaching\reef_support\UNAL_BLEACHING_TAYRONA\masks_non_bleached'

os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, 'similar'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'dissimilar'), exist_ok=True)

model_name = 'EPFL-ECEO/segformer-b5-finetuned-coralscapes-1024-1024'
processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('are u gpu or what?', device)
model.to(device)

def resize_image(image, target_size=1024):
    h_img, w_img = image.size
    if h_img < w_img:
        new_h, new_w = target_size, int(w_img * (target_size / h_img))
    else:
        new_h, new_w = int(h_img * (target_size / w_img)), target_size
    return image.resize((new_w, new_h))

def segment_image(image, crop_size=(1024, 1024), num_classes=40):
    h_crop, w_crop = crop_size
    img = torch.Tensor(np.array(resize_image(image, target_size=1024)).transpose(2, 0, 1)).unsqueeze(0)
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
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            
            with torch.no_grad():
                inputs = processor(crop_img, return_tensors="pt")
                inputs["pixel_values"] = inputs["pixel_values"].to(device)
                outputs = model(**inputs)
                
                resized_logits = F.interpolate(
                    outputs.logits[0].unsqueeze(dim=0), size=crop_img.shape[-2:], mode="bilinear", align_corners=False
                )
                preds += F.pad(resized_logits,
                                (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2))).cpu()
                count_mat[:, :, y1:y2, x1:x2] += 1
    
    preds = preds / count_mat
    preds = preds.argmax(dim=1)
    preds = F.interpolate(preds.unsqueeze(0).type(torch.uint8), size=image.size[::-1], mode='nearest')
    return preds.squeeze().cpu().numpy()

def get_ground_truth_mask(img_name):
    base_name = img_name.replace('.jpg', '').replace('.JPG', '')
    bleached_mask_path = os.path.join(masks_bleached_folder, f"{base_name}_bleached.png")
    non_bleached_mask_path = os.path.join(masks_non_bleached_folder, f"{base_name}_non_bleached.png")
    
    if os.path.exists(bleached_mask_path):
        bleached_mask = np.array(Image.open(bleached_mask_path).convert('L'))
        non_bleached_mask = np.array(Image.open(non_bleached_mask_path).convert('L'))
        combined_mask = np.zeros_like(bleached_mask)
        combined_mask[bleached_mask > 0] = 1
        combined_mask[non_bleached_mask > 0] = 0
        return combined_mask
    return None

def coralscapes_to_binary(segmentation):
    binary_mask = np.zeros_like(segmentation)
    coral_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    for class_id in coral_classes:
        binary_mask[segmentation == class_id] = 1
    return binary_mask

def process_folder(folder_path, output_subfolder):
    print(f"Processing {folder_path}...")
    all_precisions = []
    all_recalls = []
    all_accuracies = []
    
    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, img_name)
            image = Image.open(img_path).convert('RGB')
            
            segmentation = segment_image(image)
            binary_pred = coralscapes_to_binary(segmentation)
            
            ground_truth = get_ground_truth_mask(img_name)
            if ground_truth is not None:
                ground_truth_resized = Image.fromarray(ground_truth).resize(image.size, Image.NEAREST)
                ground_truth_resized = np.array(ground_truth_resized)
                
                precision = precision_score(ground_truth_resized.flatten(), binary_pred.flatten(), average='binary', zero_division=0)
                recall = recall_score(ground_truth_resized.flatten(), binary_pred.flatten(), average='binary', zero_division=0)
                accuracy = accuracy_score(ground_truth_resized.flatten(), binary_pred.flatten())
                
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_accuracies.append(accuracy)
                
                print(f"{img_name}: P={precision:.3f}, R={recall:.3f}, A={accuracy:.3f}")
            
            output_path = os.path.join(output_folder, output_subfolder, f"{img_name}_segmentation.png")
            Image.fromarray(segmentation.astype('uint8')).save(output_path)
    
    if all_precisions:
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_accuracy = np.mean(all_accuracies)
        print(f"\n{output_subfolder.upper()} SET AVERAGES:")
        print(f"Precision: {avg_precision:.3f}")
        print(f"Recall: {avg_recall:.3f}")
        print(f"Accuracy: {avg_accuracy:.3f}")
        return avg_precision, avg_recall, avg_accuracy
    return 0, 0, 0

print("Starting inference and evaluation...")
similar_metrics = process_folder(similar_folder, 'similar')
dissimilar_metrics = process_folder(dissimilar_folder, 'dissimilar')

print("\n" + "="*50)
print("FINAL RESULTS:")
print(f"Similar images - P: {similar_metrics[0]:.3f}, R: {similar_metrics[1]:.3f}, A: {similar_metrics[2]:.3f}")
print(f"Dissimilar images - P: {dissimilar_metrics[0]:.3f}, R: {dissimilar_metrics[1]:.3f}, A: {dissimilar_metrics[2]:.3f}")
print("Inference and evaluation completed!")
