''' 
Author: Chidiebere N. Okafor

Workflow for this file:
1. Match catalogue by RA/Dec to get a consistent list of galaxies to work with
2. Ask Prof. McGrath which filters to use for the R, G, B filters
3. Make cut outs for each filter.
4. Then make an RGB composite
5. Afterwards, get Galaxy zoo labels as indicated from the paper and
6. Extract the required Morphology classes from the votes
'''


''' 
Match catalogue by RA/Dec to get a consistent list of galaxies to work with
'''

# Begin by importing all required modules into this file
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import make_lupton_rgb, make_rgb, LogStretch, SqrtStretch, ManualInterval
from match_catalogue import catalogue_matcher
from filter_cutouts import make_filter_cutouts
from label_galaxy_morphology import add_morphology_classes
from psf_matching import match_psfs_with_provided_psfs
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List # For proper type annotation

"""
Main workflow for creating RGB images from multi-filter cutouts.

Workflow:
    Input:  Raw CANDELS FITS images + Galaxy Zoo catalog
    Output: RGB PNG images + label files for each galaxy
"""



BASE_PSF_PATH = "/Volumes/Research/emcgrath/CANDELS/psfs/" 

psf_red_path = BASE_PSF_PATH + "gds_60mas_wfc3_hybrid/" + "gs_deep_f160w_v0.5_psf.fits"
psf_green_path = BASE_PSF_PATH + "gds_60mas_wfc3_hybrid/" + "gs_deep_f125w_v0.5_psf.fits"
psf_blue_path = BASE_PSF_PATH + "gds_60mas_acs_yicheng/" + "gs_psf_ss_acs_i_bkgsub.fits"



#  Find the optimal combination of values for stretch and Q to best bring out wanted features for CNN training
def make_rgb_from_matched_catalog(red_dir, green_dir, blue_dir, output_dir, 
                                  matched_catalog, stretch=0.5, Q=10,
                                  psf_red_path=psf_red_path,
                                  psf_green_path=psf_green_path,
                                  psf_blue_path=psf_blue_path) -> List[str]:
    """
    Create RGB images from filter cutouts for galaxies in matched catalog.
    
    This function takes three directories of FITS cutouts (one per filter)
    and combines them into RGB PNG images suitable for CNN training.
    
    Parameters
    ----------
    red_dir : str
        Directory containing red filter cutouts (For red, we use/reserve F160W)
    green_dir : str
        Directory containing green filter cutouts (For green, we use/reserve F125W)
    blue_dir : str
        Directory containing blue filter cutouts (For blue, we use/reserve F606W)
    output_dir : str
        Directory where RGB PNGs and label files will be saved
    matched_catalog : astropy.table.Table
        Catalog of matched galaxies with morphology labels
    stretch : float, optional
        Lupton RGB stretch parameter (higher = more contrast)
        Default is 5
    Q : float, optional
        Lupton RGB asinh softening parameter (lower = more log-like)
        Default is 8
    
    Returns
    -------
    rgb_paths : list
        List of paths to created RGB images
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n MAKING RGB IMAGES FROM CUTOUTS")
    
    # Determine ID column name
    id_col = None

    # "ID" and "gz_id" columns are two different columns
    for possible_name in ['ID', 'id', 'NUMBER', 'Number']:
        if possible_name in matched_catalog.colnames:
            id_col = possible_name
            break
    
    if id_col is None:
        raise ValueError("Catalog has no ID column!")
    
    # print(f"Processing {len(matched_catalog)} galaxies...")
    # print(f"Using stretch={stretch}, Q={Q}")
    
    rgb_paths = []
    successful = 0
    missing_files = 0
    
    for idx in range(len(matched_catalog)):
        # print(f'\n \n { matched_catalog[id_col] } Code for "galaxy_id = Matched_catalog[id_col]" ') # comment out when needed
        galaxy_id = matched_catalog[id_col][idx]
        # print(f'\n \n \n {galaxy_id} vs { matched_catalog["gz_id"] if 't00_smooth_or_featured_a0_smooth_weighted_frac' in matched_catalog.colnames else 0}')
        
        # Construct file paths for each filter
        # Note: Adjust the filename pattern to match file's actual naming convention
        red_file = os.path.join(red_dir, f"candels.{galaxy_id}.f160w.fits")
        green_file = os.path.join(green_dir, f"candels.{galaxy_id}.f125w.fits")
        blue_file = os.path.join(blue_dir, f"candels.{galaxy_id}.f606w.fits")
        
        # Check all files exist
        if not all(os.path.exists(f) for f in [red_file, green_file, blue_file]):
            missing_files += 1
            if missing_files < 10:  # Only print first few missing
                print(f"  Missing files for galaxy {galaxy_id}, skipping...")
            continue
        
        # Load FITS data. Coverts the .fits files into numpy arrays.
        red_data = fits.getdata(red_file)
        green_data = fits.getdata(green_file)
        blue_data = fits.getdata(blue_file)

        # Do psf_matching here before moving on with the rest of the image processing.
        try:
            red_data, green_data, blue_data = match_psfs_with_provided_psfs(
                red_data, green_data, blue_data,
                psf_red_path, psf_green_path, psf_blue_path)
            
            print("PSF matching is working. Keep at it...")
        except Exception as e:
            print(f"PSF matching failed for galaxy {galaxy_id}: {e}")
            continue
        
        # # Clean the data (remove NaNs, infinities, negative values)
        red_data = np.nan_to_num(red_data, nan=0.0, posinf=0.0, neginf=0.0)
        green_data = np.nan_to_num(green_data, nan=0.0, posinf=0.0, neginf=0.0)
        blue_data = np.nan_to_num(blue_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # # Clip negative values to zero (flux should be non-negative). Uncomment this after checking the effect on the data
        # red_data = np.maximum(red_data, 0)
        # green_data = np.maximum(green_data, 0)
        # blue_data = np.maximum(blue_data, 0)
        
        
        # vmin_r = np.percentile(red_data, 1)
        # vmax_r = np.percentile(red_data, 99.5)
        # vmin_g = np.percentile(green_data, 1)
        # vmax_g = np.percentile(green_data, 99.5)
        # vmin_b = np.percentile(blue_data, 1)
        # vmax_b = np.percentile(blue_data, 99.5)

        # red_data = np.clip(red_data, vmin_r, vmax_r) # Rescaled array by 1.2: 1 : 1.2
        # green_data = np.clip(green_data, vmin_g, vmax_g)
        # blue_data = np.clip(blue_data, vmin_b, vmax_b)

        # # Normalize each filter to [0, 1] range for consistent scaling
        # red_data = (red_data - vmin_r) / (vmax_r - vmin_r)
        # green_data = (green_data - vmin_g) / (vmax_g - vmin_g)
        # blue_data = (blue_data - vmin_b) / (vmax_b - vmin_b)
        
        


        # Use the maximum value of the 99.5% percentile over all three filters
        # as the maximum value:
        # Borrowed code from lines 153-160 from https://docs.astropy.org/en/latest/visualization/rgb.html
        pctl = 99.5 
        maximum = 0.

        # # Clip extreme outliers to prevent scaling issues (removes bottom 1% and top 0.5% of pixels)
        for img in [red_data,green_data,blue_data]:
            val = np.percentile(img,pctl)
            if val > maximum:
                maximum = val

        # rgb_array = make_rgb(red_data, green_data, blue_data, interval=ManualInterval(vmin=0, vmax=maximum) )
        rgb_array = make_lupton_rgb(red_data, green_data, blue_data, stretch=stretch, Q=Q)

        # # Make RGB using Lupton's asinh scaling # I changed it to implement the make_rgb() function
        # rgb_array = make_rgb(red_data, green_data, blue_data, # Using LogStretch() function
        #                              stretch=LogStretch(a=10), interval=ManualInterval(vmin=0, vmax=maximum))
        
        # # rgb_array = make_rgb(red_data, green_data, blue_data, # Using SqrtStretch() function
        # #                              stretch=SqrtStretch(), interval=ManualInterval(vmin=0, vmax=maximum))

        

        # Save as PNG
        print(matched_catalog["gz_id"][idx])
        output_path = os.path.join(output_dir, f"galaxy_{matched_catalog["gz_id"][idx].strip()}.png") # names the pdf with the more informative ref_catalog id which includes the field.
        plt.imsave(output_path, rgb_array) # The .strip function above removes all leading and trailing spaces. Keeps the naming clean and tight.
        rgb_paths.append(output_path)
        
        # Save morphology label in companion file for CNN training
        if 'morphology' in matched_catalog.colnames:
            morphology = matched_catalog['morphology'][idx]
            label_file = output_path.replace('.png','_label.txt') # Removes all ".png" within the string entirely with "_label.txt".
            with open(label_file, 'w') as f:
                f.write(morphology)
        
        successful += 1
        
        # Print progress every 100 galaxies
        if successful % 100 == 0:
            print(f"  Processed {successful} galaxies...")
    

    # print("\n Some data to keep track of the RGB creation")
    # print(f"  Total galaxies in catalog: {len(matched_catalog)}")
    # print(f"  Successful RGB images: {successful}")
    # print(f"  Missing cutout files: {missing_files}")
    # print(f"  Output directory: {output_dir}")
    
    return rgb_paths


def test_small_sample(catalog, sample_size=5) -> List[str]:
    """
    Extract a small sample of galaxies for testing.
    
    Parameters
    ----------
    catalog : astropy.table.Table
        Full matched catalog
    sample_size : int, optional
        Number of galaxies to select (default 10)
    
    Returns
    -------
    sample_catalog : astropy.table.Table
        Small sample of the catalog for testing
    """
    
    # Take first N galaxies (or random sample)
    # Using first N for reproducibility
    if len(catalog) > sample_size:
        sample_catalog = catalog[:sample_size]
    else:
        sample_catalog = catalog
    
    # print(f"\n Created test sample with {len(sample_catalog)} galaxies")
    # print("Sample galaxy IDs:", sample_catalog['ID'][:5].tolist())
    
    return sample_catalog






if __name__ == "__main__":
    # Set up for testing the code for files stored on the SMB server:
    # First connect to colby smb serve and login so that the drive mounts to /Volumes/Research/

    BASE_MOUNT = "/Volumes/Research/emcgrath/Research/CANDELS_data/mosaics/gds/" # This is readable by Python. For GDS field
    # BASE_MOUNT = "/Volumes/Research/emcgrath/Research/CANDELS_data/mosaics/cos/" # This is readable by Python. For COS field
    # BASE_MOUNT = "/Volumes/Research/emcgrath/Research/CANDELS_data/mosaics/uds/" # This is readable by Python. For UDS field


    # Input data paths
    MAIN_CATALOG_PATH = "/Users/holyphysics/Desktop/Galaxy_Classification/gds_merged_v1.1.fits" # For GDS
    # MAIN_CATALOG_PATH = "/Users/holyphysics/Desktop/Galaxy_Classification/cos_merged_v1.1.fits" # For COS
    # MAIN_CATALOG_PATH = "/Users/holyphysics/Desktop/Galaxy_Classification/uds_merged_v1.1.fits" # For UDS

    REF_CATALOG_PATH = "/Users/holyphysics/Desktop/Galaxy_Classification/gz_candels_table_2_main_release.fits"
    
    # Filter information - TODO: Ask Prof. McGrath which filters to use
    RED_FILTER = "f160w"      # Example - confirm with Prof. McGrath
    GREEN_FILTER = "f125w"    # Example - confirm with Prof. McGrath
    BLUE_FILTER = "f814w"     # Example - confirm with Prof. McGrath
    
    # Paths to FITS images for each filter 
    RED_IMAGE_PATH = BASE_MOUNT + "goodss_all_wfc3_ir_f160w_060mas_v1.0_drz.fits" # For GDS field
    GREEN_IMAGE_PATH = BASE_MOUNT + "goodss_all_wfc3_ir_f125w_060mas_v1.0_drz.fits"
    BLUE_IMAGE_PATH = BASE_MOUNT + "goodss_all_acs_wfc_f814w_060mas_v1.5_drz.fits"

    # RED_IMAGE_PATH = BASE_MOUNT + "30mas/"+ "cos_2epoch_wfc3_f160w_030mas_v1.0_drz.fits" # For COS field
    # GREEN_IMAGE_PATH = BASE_MOUNT + "30mas/"+ "cos_2epoch_wfc3_f125w_030mas_v1.0_drz.fits"
    # BLUE_IMAGE_PATH = BASE_MOUNT + "cos_2epoch_acs_f606w_060mas_v1.0_drz.fits"

    # RED_IMAGE_PATH = BASE_MOUNT + "goodss_all_wfc3_ir_f160w_060mas_v1.0_drz.fits" # For UDS field
    # GREEN_IMAGE_PATH = BASE_MOUNT + "goodss_all_wfc3_ir_f125w_060mas_v1.0_drz.fits"
    # BLUE_IMAGE_PATH = BASE_MOUNT + "goodss_all_acs_wfc_f814w_060mas_v1.5_drz.fits"

    # Check if files exist before running
    for img_path, name in [(RED_IMAGE_PATH, "RED"), (GREEN_IMAGE_PATH, "GREEN"), (BLUE_IMAGE_PATH, "BLUE")]:
        if os.path.exists(img_path):
            print(f" {name}: {img_path} exists. File path correctly written. ")
        else:
            print(f" {name} NOT FOUND at: {img_path}")
    
    # Output directories
    CUTOUTS_RED_DIR = "cutouts_red"
    CUTOUTS_GREEN_DIR = "cutouts_green"
    CUTOUTS_BLUE_DIR = "cutouts_blue"
    RGB_OUTPUT_DIR = "rgb_training_data"
    
    # Testing parameters
    TEST_MODE = True           # Set to False for full run
    TEST_SAMPLE_SIZE = 6      # Number of galaxies to test with
    
    # Match catalogues and add morphology labels
    print("Matching catalogues")

    
    matched_catalog = catalogue_matcher(
        main_catalog_path=MAIN_CATALOG_PATH,
        ref_catalog_path=REF_CATALOG_PATH,
        field_filter='GDS',           # Appropriate field should be entered
        # max_separation=0.5,           # 0.5 arcseconds
        save_output=True,
        output_filename="matched_catalog.fits"
    )
    
    # Note: You'll need to add morphology classification here
    # For now, we'll proceed with the matched catalog
    # Create test sample (for debugging)
    morphology_matched_catalog = add_morphology_classes(matched_catalog)
    
    
    if TEST_MODE:
        print(" Test mode is on and will use a small sample of galaxies")
        working_catalog = test_small_sample(morphology_matched_catalog, TEST_SAMPLE_SIZE)
    else:
        working_catalog = morphology_matched_catalog
    

    # Make cutouts for each filter
    box_radius: int = 48 # So that images are 96 by 96

    print("Making cutouts for each filter")
    # # Red filter cutouts
    red_files, red_count = make_filter_cutouts(
        catalog=working_catalog,
        image_path=RED_IMAGE_PATH,
        band=RED_FILTER,
        output_dir=CUTOUTS_RED_DIR,
        box_radius=box_radius
    )
    # 
    # # Green filter cutouts
    green_files, green_count = make_filter_cutouts(
        catalog=working_catalog,
        image_path=GREEN_IMAGE_PATH,
        band=GREEN_FILTER,
        output_dir=CUTOUTS_GREEN_DIR,
        box_radius=box_radius
    )
    # 
    # # Blue filter cutouts
    blue_files, blue_count = make_filter_cutouts(
        catalog=working_catalog,
        image_path=BLUE_IMAGE_PATH,
        band=BLUE_FILTER,
        output_dir=CUTOUTS_BLUE_DIR,
        box_radius=box_radius
    )
    

    print("Making RGB images ")

    
   # And this makes the rgb images
    rgb_images = make_rgb_from_matched_catalog(
        red_dir=CUTOUTS_RED_DIR,
        green_dir=CUTOUTS_GREEN_DIR,
        blue_dir=CUTOUTS_BLUE_DIR,
        output_dir=RGB_OUTPUT_DIR,
        matched_catalog=working_catalog
    )
    

    print("Workflow is now completed.")