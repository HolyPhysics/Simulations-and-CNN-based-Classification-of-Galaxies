"""
Author: Chidiebere N. Okafor

Some handy codes here!
"""

import os

def quickprintAllImagePath(image_path: str) -> None:

    image_container = []

    if os.path.exists(image_path):
        image_container = sorted( [image for image in os.listdir(image_path) if image.lower().endswith((".png", ".jpeg", ".jpg")) ] ) ## captures everything using list comprehension

        for image in image_container:
            print(image)
    else:
        print(f" The path {image_path} is nonexistent! ", flush=True) # With  flush=True, I'm able to have this immediately printed to the screen at this junction in the codes linear execution


def quicklyReadImageLabels(label_path: str, image_name: str) -> None:
    
    label_folder_path = None

    if os.path.exists(label_path):
        label_folder_path = label_path
    elif label_folder_path == None:
        print("You have a Nonetype! Ensure that you're connected to the server!", flush=True)
    else:
        print(f" The supposed label folder {label_path} is nonexistent!", flush=True)
    
    image_label_name = image_name.replace(".png",".txt").strip()
    image_label_path = os.path.join(label_folder_path, image_label_name)

    if os.path.exists(image_label_path):
        with open(image_label_path, "r") as f:
            image_label = f.read()
        
        print(f" {image_name.strip()} has the label: {image_label}")
    else:
        print(f" The path {image_label_path} is absurdly nonexistent! What could be wrong?", flush=True)

    

if __name__ == "__main__":
    # image_path = "/Volumes/Research/emcgrath/Lydia Parent/Irregular New"

    # quickprintAllImagePath(image_path)

    label_path = "/Volumes/Research/emcgrath/Chidiebere_Okafor_N/Summer_Research_2026/magnitude_cut_rgb_training_data/Labels"
    image_name = "     galaxy_GDS_18396.png   "

    quicklyReadImageLabels(label_path, image_name)


