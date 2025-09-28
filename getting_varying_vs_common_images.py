import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

reefsupport_og_image = r'data\reef_support\content\gdrive\MyDrive\Data Challenge 3 - JBG060 AY2526\01_data\coral_bleaching\reef_support\UNAL_BLEACHING_TAYRONA\images'
base_image = r'data\reef_support\content\gdrive\MyDrive\Data Challenge 3 - JBG060 AY2526\01_data\coral_bleaching\reef_support\UNAL_BLEACHING_TAYRONA\images\C1_BC_ESb_T2_29nov24_CGomez_corr.jpg'

def extract_color_histogram(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])
    histogram = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
    return histogram / np.sum(histogram)

def classify_images_by_contrast(base_image_path, image_folder, num_similar=10, num_dissimilar=10):
    base_hist = extract_color_histogram(base_image_path)
    
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    similarities = []
    
    for img_file in image_files:
        print('processing image blha blha ', img_file, 'out of ', len(image_files))
        img_path = os.path.join(image_folder, img_file)
        img_hist = extract_color_histogram(img_path)
        similarity = cosine_similarity([base_hist], [img_hist])[0][0]
        similarities.append((img_file, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    similar_images = [img[0] for img in similarities[:num_similar]]
    dissimilar_images = [img[0] for img in similarities[-num_dissimilar:]]
    
    return similar_images, dissimilar_images

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def create_image_grid(image_list, title, folder_path):
    fig = make_subplots(
        rows=2, cols=5,
        subplot_titles=[f"{i+1}. {img}" for i, img in enumerate(image_list)],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    for i, img_name in enumerate(image_list):
        row = (i // 5) + 1
        col = (i % 5) + 1
        
        img_path = os.path.join(folder_path, img_name)
        img_data = encode_image_to_base64(img_path)
        
        fig.add_trace(
            go.Image(source=f"data:image/jpeg;base64,{img_data}"),
            row=row, col=col
        )
    
    fig.update_layout(
        title=title,
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig

similar, dissimilar = classify_images_by_contrast(base_image, reefsupport_og_image)

fig_similar = create_image_grid(similar, "10 Most Similar Images", reefsupport_og_image)
fig_dissimilar = create_image_grid(dissimilar, "10 Most Dissimilar Images", reefsupport_og_image)

fig_similar.show()
fig_dissimilar.show()