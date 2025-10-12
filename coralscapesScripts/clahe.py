import cv2
from matplotlib import pyplot as plt

image_path = 'data/cluster_2/images/C1_PB_PM_T3_19nov24_CGomez_Corr.JPG'

def apply_clahe(image_path: str = image_path) -> None:
    """
    Apply CLAHE and plot before and after image.
    """
    before2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
    before = cv2.cvtColor(before2, cv2.COLOR_RGB2Lab)
    clahe = cv2.createCLAHE(clipLimit=10,tileGridSize=(8,8))

    #0 to 'L' channel, 1 to 'a' channel, and 2 to 'b' channel
    before[:,:,0] = clahe.apply(before[:,:,0])
    img = cv2.cvtColor(before, cv2.COLOR_Lab2RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(before2)
    plt.title("Before")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb)
    plt.title("After")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


apply_clahe(image_path=image_path)