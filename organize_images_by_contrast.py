import cv2
import numpy as np
import os
import shutil
from sklearn.metrics.pairwise import cosine_similarity

reefsupport_og_image = r'data\reef_support\content\gdrive\MyDrive\Data Challenge 3 - JBG060 AY2526\01_data\coral_bleaching\reef_support\UNAL_BLEACHING_TAYRONA\images'
base_image = r'data\reef_support\content\gdrive\MyDrive\Data Challenge 3 - JBG060 AY2526\01_data\coral_bleaching\reef_support\UNAL_BLEACHING_TAYRONA\images\C1_BC_ESb_T2_29nov24_CGomez_corr.jpg'

similar_folder = 'similar_images'
dissimilar_folder = 'dissimilar_images'

os.makedirs(similar_folder, exist_ok=True)
os.makedirs(dissimilar_folder, exist_ok=True)

def extract_color_histogram(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])
    histogram = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
    return histogram / np.sum(histogram)

def classify_and_organize_images(base_image_path, image_folder, num_similar=10, num_dissimilar=10):
    base_hist = extract_color_histogram(base_image_path)
    
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    similarities = []
    
    for img_file in image_files:
        print('Processing image:', img_file, 'out of', len(image_files))
        img_path = os.path.join(image_folder, img_file)
        img_hist = extract_color_histogram(img_path)
        similarity = cosine_similarity([base_hist], [img_hist])[0][0]
        similarities.append((img_file, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    similar_images = [img[0] for img in similarities[:num_similar]]
    dissimilar_images = [img[0] for img in similarities[-num_dissimilar:]]
    
    for img in similar_images:
        src_path = os.path.join(image_folder, img)
        dst_path = os.path.join(similar_folder, img)
        shutil.copy2(src_path, dst_path)
        print(f'Copied {img} to similar folder')
    
    for img in dissimilar_images:
        src_path = os.path.join(image_folder, img)
        dst_path = os.path.join(dissimilar_folder, img)
        shutil.copy2(src_path, dst_path)
        print(f'Copied {img} to dissimilar folder')
    
    return similar_images, dissimilar_images

similar, dissimilar = classify_and_organize_images(base_image, reefsupport_og_image)

print(f"\nOrganized {len(similar)} similar images into '{similar_folder}' folder")
print(f"Organized {len(dissimilar)} dissimilar images into '{dissimilar_folder}' folder")
