import os
import matplotlib.pyplot as plt
from PIL import Image
dataset_path = "dataset/Normal"
image_files = os.listdir(dataset_path)

plt.figure(figsize=(10, 8))

for i in range(9):
    image_path = os.path.join(
        dataset_path,
        image_files[i]
    )
    image = Image.open(image_path)

    plt.subplot(3, 3, i + 1)

    plt.imshow(image, cmap="gray")

    plt.axis("off")

plt.show()