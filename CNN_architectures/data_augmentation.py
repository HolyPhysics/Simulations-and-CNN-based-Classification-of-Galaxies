"""
Author: Chidiebere N. Okafor

Purpose: Applies a select data augmentation processes an image
"""



import torch
import numpy as np
import torchvision.transforms.v2 as transformer
from PIL import Image
import matplotlib.pyplot as plt
import os

# print(transformer)

image_path = "/Users/holyphysics/Desktop/Galaxy_Classification/ZCA_whitened_random_images/zca_0000.png"

image_container = None

if os.path.exists(image_path):
    image_container = Image.open(image_path)
    print(image_container)
else:
    print()

# Note, the PIL file is good for torchvision which can either accept normal image, PIL image or tensors of an image
# Keep it as a PIL image or float tensor for easy plotting

image_augmentation = transformer.Compose(
                [
                    transformer.RandomResizedCrop(size=(224,224), antialias=True),
                    transformer.RandomHorizontalFlip(p=0.5),
                    transformer.RandomRotation(degrees=30)
                ]
)


figure, axis = plt.subplots(1,2, figsize=(12.5,7.5))

axis[0].imshow(image_container, cmap="viridis")
axis[0].set_title("Original Image")
axis[0].axis("off")
#

augmented_image = image_augmentation(image_container)

axis[1].imshow(augmented_image, cmap="viridis")
axis[1].set_title("Augmented Image")
axis[1].axis("off")


plt.tight_layout()
plt.show()




