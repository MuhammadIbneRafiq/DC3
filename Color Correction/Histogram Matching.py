import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from skimage import io, exposure

image = io.imread("20220912_AnB_CB8 (8).JPG")

reference = io.imread("20220912_AnB_CB8 (80).JPG")

matched = exposure.match_histograms(image, reference, channel_axis=-1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(reference)
axes[1].set_title("Reference")
axes[1].axis("off")

axes[2].imshow(matched)
axes[2].set_title("Histogram Matched")
axes[2].axis("off")

plt.tight_layout()
plt.show()
