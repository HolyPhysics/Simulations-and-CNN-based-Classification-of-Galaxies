''' Workflow for this file:
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
from match_catalogue import catalogue_matcher
from filter_cutouts import make_filter_cutouts
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
import os

def make_rgb_from_matched_catalog(red_dir, green_dir, blue_dir, output_dir, matched_catalog):
    """
    Create RGB images ONLY for galaxies in your matched catalog.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for galaxy_id in matched_catalog['ID']:
        # Construct file paths (adjust naming convention as needed)
        red_file = os.path.join(red_dir, f"candels.{galaxy_id}.f160w.fits")
        green_file = os.path.join(green_dir, f"candels.{galaxy_id}.f125w.fits")
        blue_file = os.path.join(blue_dir, f"candels.{galaxy_id}.f606w.fits")
        
        # Check all files exist
        if not all(os.path.exists(f) for f in [red_file, green_file, blue_file]):
            print(f"Missing files for galaxy {galaxy_id}, skipping...")
            continue
        
        # Load data
        red_data = fits.getdata(red_file)
        green_data = fits.getdata(green_file)
        blue_data = fits.getdata(blue_file)
        
        # Clean and make RGB
        rgb = make_lupton_rgb(red_data, green_data, blue_data, stretch=5, Q=8)
        
        # Save as PNG
        output_path = os.path.join(output_dir, f"galaxy_{galaxy_id}.png")
        plt.imsave(output_path, rgb)
        
        # Also save the morphology label in a companion file for CNN training
        morphology = matched_catalog[matched_catalog['ID'] == galaxy_id]['morphology'][0]
        label_file = output_path.replace('.png', '_label.txt')
        with open(label_file, 'w') as f:
            f.write(morphology)
    
    print(f"Created {len(os.listdir(output_dir))//2} RGB images with labels")



if __name__ == "__main__":
    print(0) # Delete this eventually

# Now, we simply make the rgb images as desired


    # main_catalogue = 
    # reference_catalogue = 
    # matched_catalogue = catalogue_matcher(main_catalogue, reference_catalogue)


    # ra_list = matched_catalog['RA']
    # dec_list = matched_catalog['DEC']
    # id_list = matched_catalog['ID']

    # Make the filter cutouts for each filter down here

    # red_cutouts = 
    # green_cutouts = 
    # blue_cutouts = 


    # # Call the function
    # make_rgb_from_matched_catalog(
    #     red_dir="cutouts_red",
    #     green_dir="cutouts_green",
    #     blue_dir="cutouts_blue",
    #     output_dir="rgb_training_data",
    #     matched_catalog=matched_catalogue
    # )










































# # ============================================================================
# # STEP 1: MATCH CATALOGS FIRST
# # ============================================================================

# from astropy.table import Table
# from astropy.coordinates import SkyCoord, match_coordinates_sky
# import astropy.units as u
# import numpy as np

# # Load your main catalog (all galaxies in the field)
# main_catalog = Table.read("/path/to/your/main_catalog.fits")

# # Load your reference catalog (e.g., Galaxy Zoo, LRD catalog, reliable IDs)
# reference_catalog = Table.read("/path/to/reference_catalog.fits")

# # Match by position
# main_coords = SkyCoord(ra=main_catalog['RA'], dec=main_catalog['DEC'], unit='deg')
# ref_coords = SkyCoord(ra=reference_catalog['RA'], dec=reference_catalog['DEC'], unit='deg')

# # Find closest matches
# idx, d2d, d3d = match_coordinates_sky(main_coords, ref_coords)

# # Keep only matches within 0.3 arcseconds
# max_sep = 0.3 * u.arcsec
# good_matches = d2d < max_sep

# # Create your FINAL matched catalog (only the good ones)
# matched_catalog = main_catalog[good_matches]

# # Save matched catalog for later use (and for making cutouts)
# matched_catalog.write("matched_galaxies.fits", overwrite=True)

# # Also save a simple text file with IDs and coordinates for cutout making
# with open("galaxy_list_for_cutouts.txt", "w") as f:
#     f.write("# galaxy_id    RA        DEC        morphology\n")
#     for i in range(len(matched_catalog)):
#         f.write(f"{matched_catalog['ID'][i]}    {matched_catalog['RA'][i]}    {matched_catalog['DEC'][i]}    {matched_catalog['morphology'][i]}\n")

# print(f"Matched {len(matched_catalog)} galaxies out of {len(main_catalog)} total")
# print("Now you're ready to make cutouts ONLY for these matched galaxies!")

# # ============================================================================
# # STEP 2: MAKE CUTOUTS USING THE MATCHED CATALOG
# # ============================================================================

# # Now use your matched catalog to make cutouts
# ra_list = matched_catalog['RA']
# dec_list = matched_catalog['DEC']
# id_list = matched_catalog['ID']

# # Define box size (in pixels, will be doubled to side length)
# # For CANDELS, typical cutout sizes:
# # - 200-300 pixels for single galaxies
# # - 400-600 pixels if you want to see surroundings
# box_size = np.full_like(ra_list, 300)  # 300 pixel radius → 600x600 cutout

# # Make cutouts for RED filter (e.g., F160W)
# mk_cutouts(
#     ra=ra_list,
#     dec=dec_list,
#     catid=id_list,
#     box=box_size,
#     image="/path/to/candels_f160w_sci.fits",
#     band="f160w",
#     rootname="candels",
#     outdir="cutouts_red"
# )

# # Make cutouts for GREEN filter (e.g., F125W)
# mk_cutouts(
#     ra=ra_list,
#     dec=dec_list,
#     catid=id_list,
#     box=box_size,
#     image="/path/to/candels_f125w_sci.fits",
#     band="f125w",
#     rootname="candels",
#     outdir="cutouts_green"
# )

# # Make cutouts for BLUE filter (e.g., F606W)
# mk_cutouts(
#     ra=ra_list,
#     dec=dec_list,
#     catid=id_list,
#     box=box_size,
#     image="/path/to/candels_f606w_sci.fits",
#     band="f606w",
#     rootname="candels",
#     outdir="cutouts_blue"
# )

# print("All cutouts created for matched galaxies!")

