import PIL
from PIL import Image
import os
import shutil


def apply_filters(input_path: str, output_path: str):
    """
    Create separate folders for each color filtered cluster with copied masks.
    :param input_folder: folder containing the clustered image folders
    :param output_base: folder to save each colored filtered cluster and masks
    """
    intensity = 0.1
    color_filters = {"cyan": (0, 255, 255), "yellow": (255, 255, 0), "magenta": (255, 0, 255)}

    def apply_color_filter(image, color, intensity) -> PIL.Image.Image:
        """
        Apply semi-transparent color overlay.
        :param image: image to apply color filter to
        :param color: color to apply on image
        :param intensity: intensity of color filter
        :return: image with color filter applied.
        """
        overlay = Image.new("RGBA", image.size, color + (int(255 * intensity),))
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    for cluster in range(0, 1):
        for color_name, rgb in color_filters.items():
            output_folder = f"{output_path}_{color_name}_{cluster}"
            os.makedirs(f"{output_folder}/images", exist_ok=True)

            for image in os.listdir(f"{input_path}/cluster_{cluster}"):
                if image.lower().endswith(".jpg"):
                    img_path = os.path.join(f"{input_path}/cluster_{cluster}", image)
                    img = Image.open(img_path)

                    filtered_img = apply_color_filter(img, rgb, intensity)

                    output_path = os.path.join(f"{output_folder}/images", image)
                    filtered_img.save(output_path)

                    print(f"{color_name.capitalize()} filter applied to {image}")

            shutil.copytree(f"{input_path}/masks_bleached", f"{output_folder}/masks_bleached", dirs_exist_ok=False)
            shutil.copytree(f"{input_path}/masks_non_bleached", f"{output_folder}/masks_non_bleached", dirs_exist_ok=False)
