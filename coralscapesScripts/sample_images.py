import random
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
from datetime import datetime


def plot_images(ncols: int = 8, nrows: int = 3, figsize: tuple = (10, 10), seed: int = 42) -> None:
    """
    Plot a custom number of images per cluster.
    """
    random.seed(seed)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle(t="Examples of images per cluster", fontweight="bold", fontsize=12)
    path = "data/"
    row_nr = 0

    for folder in os.listdir(path):
        if 'cluster' in folder:
            image_lists = os.listdir(f"{path}/{folder}/images")

            for i in range(ncols):
                index = random.randint(a=0, b=len(image_lists) - 1)
                image = image_lists[index]
                img = os.path.join(path, folder, 'images', image)
                img = plt.imread(img)
                ax[row_nr, i].imshow(img)

            row_nr += 1

    now = datetime.now()
    plt.savefig(f"artifacts/samples_per_cluster_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}_{now.second:02}.png")
    plt.show()


plot_images(ncols=8, nrows=3, figsize=(14, 5))
