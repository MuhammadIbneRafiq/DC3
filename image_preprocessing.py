import PIL
from PIL import Image
import os
import shutil
import cv2
import numpy as np
import sys
from sklearn.cluster import KMeans
from kneed import KneeLocator
from config import __RANDOM_STATE__


class ImagePreprocessing:
    """
    Cluster images based on their dominant color using the KMeans clustering algorithm.
    Apply CMY color filters to each image.
    """
    def __init__(self, input_path: str, output_path: str, random_state: int = __RANDOM_STATE__):
        self.input_path = input_path
        self.output_path = output_path
        self.random_state = random_state
        self.optimal_k: int = 3  # will change value once self.cluster_images() finds a different optimal k

    def cluster_images(self):
        """
        Cluster images based on their dominant color using KMeans clustering.
        """
        images_dir = os.path.join(self.input_path, "images")
        masks_bleached_dir = os.path.join(self.input_path, "masks_bleached")
        masks_non_bleached_dir = os.path.join(self.input_path, "masks_non_bleached")
        features, image_paths, sse, medoid_images = [], [], [], []
        print(f"\nLooking for images in: {os.path.abspath(images_dir)}\nExtracting features from images...")

        for idx, file in enumerate(os.listdir(images_dir)):
            if file.lower().endswith(('.jpg', '.png')):
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
        K_range = range(1, 16)

        for k in K_range:
            print(f"  Testing k={k}...", end=" ")
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(features)
            sse.append(kmeans.inertia_)

        kl = KneeLocator(K_range, sse, curve="convex", direction="decreasing")
        self.optimal_k = kl.elbow

        print(f"\nPerforming final clustering with k={self.optimal_k}...")
        kmeans = KMeans(n_clusters=self.optimal_k, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(features)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        print("\nMedoid images (representative image for each cluster):")
        for idx, path in enumerate(medoid_images):
            print(f"  Cluster {idx}: {os.path.basename(path)}")

        print("\nCluster distribution:")
        for i in range(self.optimal_k):
            print(f"  Cluster {i}: {np.sum(labels == i)} images")
            cluster_indices = np.where(labels == i)[0]
            cluster_features = features[cluster_indices]
            centroid = kmeans.cluster_centers_[i]

            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            medoid_index = cluster_indices[np.argmin(distances)]
            medoid_images.append(image_paths[medoid_index])

            cluster_folder = os.path.join(self.output_path, f"cluster_{i}")
            os.makedirs(os.path.join(cluster_folder, "images"), exist_ok=True)
            os.makedirs(os.path.join(cluster_folder, "masks_bleached"), exist_ok=True)
            os.makedirs(os.path.join(cluster_folder, "masks_non_bleached"), exist_ok=True)
            print(f"  Created cluster_{i} folders")

        print("\nCopying images and masks to cluster folders...")
        copied_count, masks_copied, masks_not_found = 0, 0, 0

        for img_path, label in zip(image_paths, labels):
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]  # find corresponding masks
            cluster_folder = os.path.join(self.output_path, f"cluster_{label}")

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

        print(f"Clustering complete!\nOutput location: {os.path.abspath(self.output_path)}")
        sys.stdout.flush()

    @staticmethod
    def apply_color_filter(image: PIL.Image.Image, color: tuple[int, int, int], intensity: float) -> PIL.Image.Image:
        """
        Apply semi-transparent color overlay.
        :param image: image to apply color filter to
        :param color: color to apply on image
        :param intensity: intensity of color filter
        :return: image with color filter applied.
        """
        overlay = Image.new("RGBA", image.size, color + (int(255 * intensity),))
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    def apply_filters(self):
        """
        Create separate folders for each color filtered cluster with copied masks.
        Apply all color filters to all clusters.
        """
        intensity = 0.1
        color_filters = {"cyan": (0, 255, 255), "yellow": (255, 255, 0), "magenta": (255, 0, 255)}

        for cluster in range(0, self.optimal_k):
            for color_name, rgb in color_filters.items():
                output_folder = f"{self.output_path}_{color_name}_{cluster}"
                os.makedirs(f"../{output_folder}/images", exist_ok=True)

                for image in os.listdir(f"{self.input_path}/cluster_{cluster}"):
                    if image.lower().endswith(".jpg"):
                        img_path = os.path.join(f"{self.input_path}/cluster_{cluster}", image)
                        img = Image.open(img_path)
                        filtered_img = self.apply_color_filter(img, rgb, intensity)

                        output_dir = os.path.join(f"{output_folder}/images", image)
                        filtered_img.save(output_dir)

                print(f"{color_name.capitalize()} filter applied to {cluster}")

                shutil.copytree(f"{self.input_path}/masks_bleached", f"{output_folder}/masks_bleached",
                                dirs_exist_ok=False)
                shutil.copytree(f"{self.input_path}/masks_non_bleached", f"{output_folder}/masks_non_bleached",
                                dirs_exist_ok=False)
