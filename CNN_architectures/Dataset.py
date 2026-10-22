"""
Author: Chidiebere N. Okafor

"""
import random
import matplotlib.pyplot as plt
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
                "Irregular" : 2,
                "Uncertain" : -1,
        }

        self.sorted_image_file_names = sorted( [image for image in os.listdir( self.image_directory ) if image.lower().endswith( (".png", ".jpeg", ".jpg") ) ] ) ## returns only file names! Not the full path!
        # self.sorted_image_paths = []

        # It should be os.listdir() not os.path.listdir()

        # Capture the image data using "List COmprehension Technique."

    def __len__(self):
        # print(f" Your custom Dataset object's __len__ method works!")

        return len( self.sorted_image_file_names )

        

    def __getitem__(self, selector_id):
        """
        selector_id : int
            An internal call id to select/grab images from the list of sorted images
        """

        # First things first, we select the image path
        image_path = os.path.join(self.image_directory, self.sorted_image_file_names[selector_id]) # This forms the complete image path.

        image_path_root, image_path_extension = os.path.splitext(self.sorted_image_file_names[selector_id])

        # Next, we construct the path to the image label
        image_label_path  = os.path.join(self.label_directory, f"{image_path_root}.txt")

        # print(image_path_root)
        # print(image_path_extension)

        if not os.path.exists(image_label_path):
            raise FileNotFoundError(f" The file {image_label_path} is nonexistent!")
        
        with open(image_label_path, "r") as f:
            image_text_label = f.read().strip()
        
        if image_text_label not in self.image_label_to_number_map:
            raise ValueError(f" Invalid label '{image_text_label}' found in the file {image_label_path}")

        image_numeric_label = self.image_label_to_number_map[image_text_label]

        image_itself = Image.open(image_path).convert("RGB") ## I'm just seeing that ResNet50 require three channels SO surprises with RGBA, are excluded!

        if self.transform:
            image_itself = self.transform(image_itself)

        # print(f" Your custom Dataset object's __getitem__ method works!")

        return image_itself, torch.tensor(image_numeric_label, dtype=torch.long)




if __name__ == "__main__":

    image_augmentation = transformer.Compose(
                [
                    # transformer.RandomResizedCrop(size=(224,224), antialias=True),
                    transformer.RandomResizedCrop(224, scale=(0.5, 1.0), antialias=True),

                    transformer.RandomHorizontalFlip(p=0.5),
                    transformer.RandomRotation(degrees=30)
                ]
    )

    print(image_augmentation, flush=True) # I want instant alert!

    image_directory = "/Volumes/Research/emcgrath/Chidiebere_Okafor_N/Summer_Research_2026/magnitude_cut_rgb_training_data/Images"
    label_directory = "/Volumes/Research/emcgrath/Chidiebere_Okafor_N/Summer_Research_2026/magnitude_cut_rgb_training_data/Labels"

    training_dataset = GalaxyCNNDataSet(image_directory, label_directory, transform=image_augmentation)


    def display_augmentations(dataset, index_start = 0, show_all_augmentation = False, augmentations_per_image = 3):
        ''' How do I build the logic to desplay the labels correctly as my current labels remain quite misleading'''
        # image_path = os.path.join( dataset.image_directory, dataset.sorted_image_file_names[index_start] )
        # original_image = Image.open(image_path).convert("RGB")

        # initialize a random.seed(2) here when consistency of an image is desired!

        if (show_all_augmentation == True):
            index_stop = ( dataset.__len__() ) // 2  # Lets go with the floor division
        elif ( dataset.__len__() ) // 2 > 10:
            index_stop = random.randint(1, 10 )
        else: 
            index_stop = random.randint(1, ( dataset.__len__() ) // 2 )




        fig, axes = plt.subplots(1, index_stop, figsize=(3*(index_stop), 3  ) )

        # axes[0].imshow(original_image) ## I made the intial error or trying to visualize this with plt.plot()
        # axes[0].set_title("Original image")

        for k in range(index_stop):
            augment, label = dataset[k] ## These are the new augmentations
            # augment = augment.permute(1,2,0).numpy() # Since the tensor augment has shape (C,H,W) but matplotlib requires shape (H,W,C)
            axes[k].imshow(augment)
            axes[k].set_title(f"Image {k} Augmented")

        for ax in axes:
            ax.axis("off")

        plt.suptitle(f"Image Augmentation Showing { index_stop } out of { dataset.__len__()  } images")
        plt.tight_layout()
        plt.show()



    def display_single_image_varienats(dataset, index = 0, number_of_variants = 5):
        fig, axes = plt.subplots( 1, number_of_variants, figsize=(15, 5))

        for k in range(number_of_variants):
            augment, label = dataset[index] ## These are the new augmentations
            # augment = augment.permute(1,2,0).numpy() # Since the tensor augment has shape (C,H,W) but matplotlib requires shape (H,W,C)
            axes[k].imshow(augment)
            axes[k].axis("off")
            axes[k].set_title(f"Variant {k + 1}")
        
        plt.suptitle(f"Augmentation variants for Image Index : {index}")
        plt.tight_layout()
        plt.show()




    display_augmentations(training_dataset)

    display_single_image_varienats(training_dataset, index = 56)

    # image_itself, image_tensor = training_dataset.__getitem__(0)

    # plt.plot( [image_itself, image_itself])
    # print( training_dataset.__getitem__(0) )
