"""
Author: Chidiebere N. Okafor

"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import v2
import torchvision.transforms.v2 as transformer
from PIL import Image


class GalaxyCNNDataSet(Dataset):

    def __init__(self, image_directory, label_directory, transform=None):
        """
        image_directory: str
            Direct (or Relative path) to the galaxy images for CNN training
        label_diectory: str
            Direct (or Relative path) to the labels for corresponding galaxy images for CNN training
        """

        self.image_directory = image_directory
        self.label_directory = label_directory
        self.transform = transform

        self.image_label_to_number_map = {
                "Spiral" : 0,
                "Elliptical" : 1,
                "Irregular" : 2
        }

        self.sorted_image_paths = sorted( [image for image in os.listdir( self.image_directory ) if image.lower().endswith( (".png", ".jpeg", ".jpg") ) ] )
        # It should be os.listdir() not os.path.listdir()

        # Capture the image data using "List COmprehension Technique."

    def __len__(self):
        print(f" Your custom Dataset object's __len__ method works!")

        return len( self.sorted_image_paths )

        

    def __getitem__(self, selector_id):
        """
        selector_id : int
            An internal call id to select/grab images from the list of sorted images
        """

        # First things first, we select the image path
        image_path = self.sorted_image_paths[selector_id]

        image_path_root, image_path_extension = os.path.splitext(image_path)

        # Next, we construct the path to the image label
        image_label_path  = os.path.join(self.label_directory, f"{image_path_root}.txt")

        if not os.path.exists(image_label_path):
            raise FileNotFoundError(f" The file {image_label_path} is nonexistent!")
        
        with open(image_label_path, "r") as f:
            image_text_label = f.read().strip()
        
        if image_text_label not in self.image_label_to_number_map:
            raise ValueError(f" Invalid label '{image_text_label}' found in the file {image_label_path}")

        image_numeric_label = self.image_label_to_number_map[image_text_label]

        image_itself = Image.open(image_path)

        if self.transform:
            image_itself = self.transform(image_itself)

        print(f" Your custom Dataset object's __getitem__ method works!")

        return image_itself, torch.tensor(image_numeric_label, dtype=torch.long)




if __name__ == "__main__":

    image_augmentation = transformer.Compose(
                [
                    transformer.RandomResizedCrop(size=(224,224), antialias=True),
                    transformer.RandomHorizontalFlip(p=0.5),
                    transformer.RandomRotation(degrees=30)
                ]
    )

    image_directory = "/Users/holyphysics/Desktop/Galaxy_Classification/rgb_training_data/Images"
    label_directory = "/Users/holyphysics/Desktop/Galaxy_Classification/rgb_training_data/Labels"

    training_dataset = GalaxyCNNDataSet(image_directory, label_directory, transform=image_augmentation)