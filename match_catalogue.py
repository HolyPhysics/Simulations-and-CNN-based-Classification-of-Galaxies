from astropy.io import fits
from astropy.table import Table,QTable
import numpy as np
import astropy.units as u
import os.path
#import re as regex
import matplotlib.pyplot as plt
import matplotlib as mpl
# %matplotlib inline
mpl.rcParams['savefig.dpi'] = 90
mpl.rcParams['figure.dpi'] = 90
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky


def catalogue_matcher(main_catalogue, reference_catalogue):

    # # Load your CANDELS GOODS-S catalog
    # # your_catalog = Table.read("/Users/holyphysics/Desktop/Galaxy_Classification/gds_merged_v1.1.fits")  # REPLACE with actual path

    # your_catalog  = Table.read("/Users/holyphysics/Desktop/Galaxy_Classification/gz_candels_table_2_main_release.fits")
    # # PRINT ALL COLUMN NAMES - THIS IS YOUR DEBUGGING TOOL
    # print("Column names in your catalog:")
    # print(your_catalog.colnames)
    # print("\n" + "="*50 + "\n")

    # # Also print first few rows to see the data format
    # print("First 3 rows of data:")
    # print(your_catalog[:3])

    # # Check specifically for RA and Dec columns
    # print("\n" + "="*50)
    # print("Looking for coordinate columns:")
    # for col in your_catalog.colnames:
    #     if 'ra' in col.lower() or 'dec' in col.lower():
    #         print(f"  Found: '{col}'")



    # ============================================================================
    # STEP 1: Load your CANDELS GOODS-S catalog
    # ============================================================================
    your_catalog = Table.read("/Users/holyphysics/Desktop/Galaxy_Classification/gds_merged_v1.1.fits")
    print(f"Your catalog has {len(your_catalog)} galaxies")

    # ============================================================================
    # STEP 2: Load Galaxy Zoo CANDELS catalog
    # ============================================================================
    # Download from https://data.galaxyzoo.org
    gz_catalog = Table.read("/Users/holyphysics/Desktop/Galaxy_Classification/gz_candels_table_2_main_release.fits")  # or .csv

    # Filter to GOODS-S only (IDs starting with 'GDS')
    gz_goods = gz_catalog[[str(id).startswith('GDS') for id in gz_catalog['ID']]]
    print(f"Galaxy Zoo has {len(gz_goods)} galaxies in GOODS-S")

    # ============================================================================
    # STEP 3: Match by position (same as your LRD matching code)
    # ============================================================================
    your_coords = SkyCoord(ra=your_catalog['RA'], dec=your_catalog['DEC'], unit='deg')
    gz_coords = SkyCoord(ra=gz_goods['RA'], dec=gz_goods['Dec'], unit='deg')

    idx, d2d, d3d = match_coordinates_sky(your_coords, gz_coords)

    max_sep = 0.5 * u.arcsec  # 0.5 arcsec is reasonable for CANDELS
    matched = d2d < max_sep

    # Create your matched catalog with morphology labels
    matched_catalog = your_catalog[matched]
    matched_gz = gz_goods[idx[matched]]

    final_catalog = matched_catalog

    print(f"Matched {len(final_catalog)} galaxies with reliable morphology labels")

    # Save for cutout making
    final_catalog.write("goods_galaxies_with_morphology.fits", overwrite=True)

    return 0;
    #  What you really want it to return is the matched_catalog as written below




# find a way to have this added to the catalogue_matcher() method:

# # Create your FINAL matched catalog (only the good ones)
# matched_catalog = main_catalog[good_matches]

# # Save matched catalog for later use (and for making cutouts)
# matched_catalog.write("matched_galaxies.fits", overwrite=True)