""" 
Author: Chidiebere N. Okafor

Purpose: This file takes in the training data, performs the so-called ZCA-whitening on the
images ( possible image extensions considered include: ".png", ".jpg", ".jpeg") and returns the list of the ZCA-whitened images.

ZCA whitenning decorrelates and scales images pixels in the image channels and rotates the final image as best as possible to the original shape
"""

"""
Task/Notes:
    - This code fails for single images(Any workaround for this edge case?)
    - Finish up the docstring for the ZCA_whitening() function
    - Save the covariance matrices as well and the 
    - Mean to a file to be applied on the testing data set.
    - Write other files to perform various other data augmentation tasks like:
        - Random Rotation
        - Horizontal Flipping
        - Random Cropping
        - Color Augmentation
        - Scale jittering
    - Write a different file to add interacting merger/non interacting classification for each galaxy
"""


import numpy as np
import os
# import PIL
from PIL import Image, ImageOps # This is a versatile python library for image processing
from typing import Tuple, List
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# print(dir(Image))
# print(f"\n {True if "open" in dir(Image) else 0}")

# for things in dir(PIL):
#     print(things)

# print(True if Image in PIL else 0)


def normalize_image_robust(image_array, percentile=1):
    """
    Min-Max normalization with outlier handling
    Uses percentiles instead of min/max to avoid extreme outliers
    """
    # Use percentiles to ignore extreme outliers
    lower = np.percentile(image_array, percentile)
    upper = np.percentile(image_array, 100 - percentile)
    
    # Clip to percentile range
    clipped = np.clip(image_array, lower, upper)
    
    # Normalize to 0-255
    if upper - lower > 0:
        normalized = (clipped - lower) / (upper - lower) * 255
        return normalized.astype(np.uint8)
    else:
        return np.zeros_like(image_array, dtype=np.uint8)


def ZCA_whitening(image_path, output_path, regularization_constant = 1e-4) -> None: # Change this to the appropriate type annotation later
    """
    some notes here

    Parameters: 
    ----------
    image_path:
        descriptions
    ouput_path:
        desc
    regularization_constant:
        desc
    


    Return:
    -------



    Notes:
    -----
    """

    # First create the ouput directory if it doesn't already exist
    os.makedirs(output_path, exist_ok=True) 

    # Next, we ensure that the the image path does exist and make a suiting assignment
    image_folder = None

    if os.path.exists(image_path):
        image_folder = image_path
        print(' Image folder exists')
    else:
        print("\n The image path is nonexistent!")
    
    # cropped_images = [] # Uncomment the code below when working with images of different shapes
    # target_size = (150,150) # Uncomment the code below when working with images of different shapes

    ## Uncomment the code below when working with images of different shapes
    # for filename in os.listdir(image_folder): # this part of the code is credited to: https://www.pythoninformer.com/python-libraries/pillow/imageops-resizing/
    #     if filename.lower().endswith((".png",".jpg",".jpeg")):
    #         img_path = os.path.join(image_folder, filename)
            
    #         with Image.open(img_path) as img:
    #             # ImageOps.fit automatically scales and center-crops to fit the exact size of the images
    #             final_img = ImageOps.fit(img, target_size, centering=(0.5, 0.5))
    #             cropped_images.append(final_img)
    #             # final_img.save(os.path.join(output_folder, f"fit_{filename}"))

    # grab all files ending in ".png" from the provided file path as follows:
    image_container = [image for image in os.listdir(image_folder) if image.lower().endswith((".png",".jpg",".jpeg")) ]
    # print(image_container)
    # print(image_container)
    collected_images = []

    # Then we load each image and get them ready for the ZCA
    for image_name in image_container: # change to for image_name in cropped_images when dealing with images of different shapes
        image_path = os.path.join(image_folder, image_name)   #comment this out when dealing with images of different shapes
        # print(f"\n {image_path}")
        image = Image.open(image_path).convert("RGB") #comment this out when dealing with images of different shapes
        '''
        I can convert to only RGB using Image.open(img_path).convert('RGB') if the 
        Alpha channel is not needed or using an if channel = 4 images = images[:,:,:, :3]
        #comment the above out when dealing with images of different shapes
        '''
        # print(f"\n {image}")
        image_array = np.array(image, dtype=np.float32) # This is a 3-dimensional array of a single image
        # Change the image in above line to image_name when working with images of different dimesnions and after commenting out all the required codes for such case
        # print(f"\n {image_array}")
        collected_images.append(image_array)
        # print(f"\n {collected_images}")

    original_image_datatype = np.uint8 # assuming the uniformity of the datatypes across all images
    print(original_image_datatype)

    # Next, convert the entire images to a numpy array. 
    # This is because the ZCA is applied to all the training data at once

    images = np.array(collected_images, dtype=np.float32) # This is a 4-dimensional array of all the collected images
    # print(images.shape)
    number_of_images, height, width, color_channels = images.shape # if color_channels = 3(just RGB) if it equals 4 then RGBA(where the A = Alpha)

    # Now, flatten each image into a vector of pixels such that the flattened image has dimension D = height * width * color_channels
    # That is Shape: (number_of_images, height * width * color_channels ), each iimage becomes one row
    flattened_image = images.reshape(number_of_images, -1) # This does it but we can equally just hard code (number_of_images, height * width * color_channels )
    # print(flattened_image.shape)
    # print(flattened_image)

    flattened_image_mean = np.mean(flattened_image, axis=0) # axis = 0 operates vertically for each column and axis = 1 operates horizontally for each rows
    # this computes the mean of the pixels value for each color channel as outlined in the ZCA steps
    # print(flattened_image_mean)

    # Centering the data becomes easy
    centered_flattened_image = flattened_image - flattened_image_mean
    # print(centered_flattened_image)

    # This leads us up to computing the covariance matrix
    # For small dataset, use:
    # covariance_matrix = np.cov(centered_flattened_image, rowvar=False) 
    # For larger datasets, the more efficient thing to do is to use:
    # covariance_matrix = (centered_flattened_image.T @ centered_flattened_image)/ (number_of_images - 1)

    # Compute eigenvectors and eigenvalues
    # eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # print(eigenvalues)

    # Shoot, this is much much computationally expensiove and took over 7 minutes without completion
    # print(474.457/60)
    # Use Singular Value Decomposition
    svd_start_time = time.time()

    U, S, vt = np.linalg.svd(centered_flattened_image, full_matrices=False)

    print(f" SVD completed in { time.time() - svd_start_time } sec")
    # print(S)
    # print(f"\n {U}")
    # print(f'\n {vt}')

    # And finally the ZCA-whitening
    epsilon = regularization_constant
    inverse_sqrt_of_S = 1/ np.sqrt( S + epsilon) # It is more numerically stable to use the sqrt of the eigenvalues at this step
    
    V = vt.T # This is the U or V matrix from the ZCA formula

    zca_whitened_images = centered_flattened_image @ V @ np.diag( inverse_sqrt_of_S ) @ V.T # The ZCA whitening
    # zca_whitened_images =  V @ np.diag( inverse_sqrt_of_S ) @ V.T @ centered_flattened_image # The ZCA whitening
    # pca =  PCA(whiten=True, n_components=centered_flattened_image.shape[1])
    # zca_whitened_images = pca.fit_transform(centered_flattened_image) # PCA whitening
    # print(zca_whitened_images)

    # Reshaping the zca_whitened images back to the original shape of the image
    zca_images = zca_whitened_images.reshape( number_of_images, height, width, color_channels )
    # print(zca_images)

    # Now, we save these images
    for index, image_array in enumerate(zca_images):
        # convert back to the original datatype for saving
        normalized_image = normalize_image_robust(image_array)
        image_array = np.clip(image_array, 0, 255).astype(original_image_datatype)
        image = Image.fromarray(normalized_image)
        output_file_path = os.path.join(output_path, f"zca_{index:04d}.png") # :04d tells python to format these as 4 digit integers
        image.save(output_file_path)
        if index % 10 == 0:
            print(f"  Saved {index}/{number_of_images} images")
    
    print(f"Done! Saved {number_of_images} whitened images to {output_path}")

    return zca_images



if __name__ == "__main__":

    # image_path = "random_images_from_the_internet"  # Uncomment the code below when working with images of different shapes
    # output_path = "ZCA_whitened_random_images"

    image_path = "rgb_training_data" # These images are of the same dimensions as the filter_cutouts function makes them all of the same height and width
    output_path = "ZCA_whitened_images"

    zca_images = ZCA_whitening(image_path, output_path)
    # print(zca_images)

    plt.figure(figsize=(10, 10))
    plt.imshow(zca_images[0][:,:,1], cmap='viridis')  # Shows first channel
    plt.colorbar()
    plt.title("ZCA Whitened - First Channel")
    plt.show()