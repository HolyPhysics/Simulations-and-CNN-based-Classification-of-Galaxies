"""
Author: Chidiebere N. Okafor
Purpsoe: Grabs a randomized lists of 100 galaxies from each class and stores them up for classification.
"""

"""
TASK for later:
    - Complete the docstring for the gal_image_extraction function
    - I don't think I need to do any fancy type annotations
"""


import os
import numpy as np
from PIL import Image
import random

def gal_image_extraction(input_image_dir, output_file_path, number_of_required_per_class_extracts) -> None:
    """
    Parameters
    ----------


    Return
    ------

    """


    os.makedirs(output_file_path, exist_ok=True)
    output_dir = output_file_path

    rgb_data_path = None

    if os.path.exists(input_image_dir):
        rgb_data_path = input_image_dir
        print("The path is correctly written!")
    else:
        print(f"{input_image_dir} is nonexistent")

    counter = {"elliptical" : 0, "spiral" : 0, "irregular" : 0, "uncertain" : 0}

    image_path = os.path.join(rgb_data_path, "Images")
    label_path = os.path.join(rgb_data_path, "Labels")


    # image_list = list( os.listdir(image_path) ) 
    # print(image_list[0])
    # image_dict = { index : labels.replace(".txt", ".png") for index, labels in enumerate( list(os.listdir(label_path)) )}
    ## Matches 
    # print(image_dict)

    collected_labels = []

    for index, labels in enumerate( os.listdir(label_path) ) :
        # print(index, labels)

        # image_list = list( os.listdir(image_path) ) 
        # print(image_list)

        image_label = os.path.join(label_path, labels)
        # print(image_label)
        with open(image_label, "r") as f:
            content = f.read()

            if content.lower() == "elliptical" and counter["elliptical"] < number_of_required_per_class_extracts:

                corresponding_image_path = labels.replace(".txt", ".png")

                collected_labels.append(corresponding_image_path)
                counter["elliptical"] += 1
            
            if content.lower() == "spiral" and counter["spiral"] < number_of_required_per_class_extracts:

                corresponding_image_path = labels.replace(".txt", ".png")

                collected_labels.append(corresponding_image_path)
                counter["spiral"] += 1

            if content.lower() == "irregular" and counter["irregular"] < number_of_required_per_class_extracts:

                corresponding_image_path = labels.replace(".txt", ".png")

                collected_labels.append(corresponding_image_path)
                counter["irregular"] += 1

    
    # Shuffle the images in the data
    indices_from_collected_labels = list(range( len(collected_labels) ))
    # print(len(collected_labels))
    # print(f" Before: { indices_from_collected_labels }")
    random.shuffle(indices_from_collected_labels)
    # print(f" After: { indices_from_collected_labels }")

    # print(collected_labels)

    shuffled_collected_labels = [collected_labels[index] for index in indices_from_collected_labels]

    # print(labels for labels in collected_labels)

    for image_names in shuffled_collected_labels:
        image_file_path = os.path.join(image_path, image_names)
        
        # if os.path.exists(image_file_path):
        #     print("Keep going")
        # else:
        #     print("Wrong!")
            # return;

        image = Image.open(image_file_path).convert("RGB")
        output_path = os.path.join(output_dir, image_names)
        image.save(output_path)

    # image_list = os.listdir( image_path )
    # print(image_list)

    # for labels in collected_labels:
    #     print(labels)
        # /Users/holyphysics/Downloads/rgb_training_data/Images



    


if __name__ == "__main__":

    input_image_dir = "/Users/holyphysics/Downloads/rgb_training_data"
    ouput_dir = "images_to_be_classified"
    number_of_required_per_class_extracts = 100

    gal_image_extraction(input_image_dir, ouput_dir, number_of_required_per_class_extracts)

