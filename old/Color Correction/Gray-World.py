import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

def gray_world_white_balance(image):
    result = image.astype(np.float32)
    avg_rgb = np.mean(result, axis=(0, 1))
    gray_val = np.mean(avg_rgb)
    scale = gray_val / avg_rgb
    result *= scale
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

image = io.imread("20220912_AnB_CB8 (7).JPG")
corrected = gray_world_white_balance(image)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(corrected)
axes[1].set_title("Gray-World Corrected")
axes[1].axis("off")

plt.tight_layout()
plt.show()
