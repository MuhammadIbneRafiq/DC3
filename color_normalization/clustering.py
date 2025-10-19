import cv2
import numpy as np
import os
import sys
from sklearn.cluster import KMeans
from kneed import KneeLocator
import shutil


def cluster_images(data_root: str, output_dir: str):
    """
    Cluster images based on their dominant color using KMeans.
    :param data_root: directory containing image and mask folders
    :param output_dir: directory to save clustered folders
    """
    images_dir = os.path.join(data_root, "images")
    masks_bleached_dir = os.path.join(data_root, "masks_bleached")
    masks_non_bleached_dir = os.path.join(data_root, "masks_non_bleached")
    features, image_paths = [], []
    print(f"\nLooking for images in: {os.path.abspath(images_dir)}\nExtracting features from images...")

    for idx, file in enumerate(os.listdir(images_dir)):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(images_dir, file)
            image = cv2.imread(path)
            if image is None:
                continue
            image_paths.append(path)

            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            mean_lab = lab.reshape(-1, 3).mean(axis=0)
            features.append(mean_lab)

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1} images...")

    features = np.array(features)
    sse = []
    K_range = range(1, 16)

    for k in K_range:
        print(f"  Testing k={k}...", end=" ")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        sse.append(kmeans.inertia_)

    kl = KneeLocator(K_range, sse, curve="convex", direction="decreasing")
    optimal_k = kl.elbow

    print(f"\nPerforming final clustering with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    print("\nCluster distribution:")
    for i in range(optimal_k):
        print(f"  Cluster {i}: {np.sum(labels == i)} images")

    medoid_images = []
    for i in range(optimal_k):
        cluster_indices = np.where(labels == i)[0]
        cluster_features = features[cluster_indices]
        centroid = kmeans.cluster_centers_[i]

        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        medoid_index = cluster_indices[np.argmin(distances)]
        medoid_images.append(image_paths[medoid_index])

    print("\nMedoid images (representative image for each cluster):")
    for idx, path in enumerate(medoid_images):
        print(f"  Cluster {idx}: {os.path.basename(path)}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\nCreating cluster folder structure in: {os.path.abspath(output_dir)}")
    for i in range(optimal_k):
        cluster_folder = os.path.join(output_dir, f"cluster_{i}")
        os.makedirs(os.path.join(cluster_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(cluster_folder, "masks_bleached"), exist_ok=True)
        os.makedirs(os.path.join(cluster_folder, "masks_non_bleached"), exist_ok=True)
        print(f"  Created cluster_{i} folders")

    print("\nCopying images and masks to cluster folders...")
    copied_count, masks_copied, masks_not_found = 0, 0, 0

    for img_path, label in zip(image_paths, labels):
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]  # get base name without extension to find corresponding masks
        cluster_folder = os.path.join(output_dir, f"cluster_{label}")

        # copy images and masks
        img_dest = os.path.join(cluster_folder, "images", filename)
        if not os.path.exists(img_dest):
            shutil.copy(img_path, img_dest)

        mask_bleached_path = os.path.join(masks_bleached_dir, f"{base_name}_bleached.png")
        mask_bleached_dest = os.path.join(cluster_folder, "masks_bleached", f"{base_name}_bleached.png")
        if os.path.exists(mask_bleached_path):
            shutil.copy(mask_bleached_path, mask_bleached_dest)
            masks_copied += 1
        else:
            masks_not_found += 1
            if masks_not_found == 1:
                print(f"  [DEBUG] First missing mask: {mask_bleached_path}")

        mask_non_bleached_path = os.path.join(masks_non_bleached_dir, f"{base_name}_non_bleached.png")
        mask_non_bleached_dest = os.path.join(cluster_folder, "masks_non_bleached", f"{base_name}_non_bleached.png")
        if os.path.exists(mask_non_bleached_path):
            shutil.copy(mask_non_bleached_path, mask_non_bleached_dest)
            masks_copied += 1
        else:
            masks_not_found += 1

        copied_count += 1
        if copied_count % 50 == 0:
            print(f"  Copied {copied_count}/{len(image_paths)} images...")
            sys.stdout.flush()

    print(f"Copied all {copied_count} images")
    print(f"Copied {masks_copied} mask files")
    print("CLUSTERING COMPLETE!")
    print(f"Output location: {os.path.abspath(output_dir)}")
    print(f"Created {optimal_k} cluster folders with images and masks")
    sys.stdout.flush()
