"""
Author: Chidiebere N. Okafor; Prof. Elizabeth McGrath
"""

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
from astropy.coordinates import SkyCoord, match_coordinates_sky
from typing import List # For proper type annotation

"""
match_catalogue.py
Module for matching astronomical catalogs by RA/Dec coordinates.

This module takes a main catalog (for example, CANDELS GOODS-S) and a reference catalog
(for example, Galaxy Zoo CANDELS) and returns a matched catalog containing only galaxies
found in both with reliable coordinates.

Purpose:
    - Match galaxies by position (not by unreliable IDs)
    - Create a clean, matched catalog for further processing
    - Preserve all relevant columns from the main catalog
    - Add morphology labels from the reference catalog
"""


#  max_separation should not be greater than 2
def catalogue_matcher(main_catalog_path, ref_catalog_path, field_filter='GDS', 
                      max_separation=1, save_output=True, output_filename="matched_catalog.fits") -> List[str]:
    """
    Match a main catalog against a reference catalog using RA/Dec positions according to 
    a specified observation field( "GDS", "UDS", "COS")
    
    Parameters
    ----------
    main_catalog_path : str
        Path to the main catalog FITS file (for example, CANDELS GOODS-S catalog)
    ref_catalog_path : str
        Path to the reference catalog FITS file (for example, Galaxy Zoo CANDELS catalog)
    field_filter : str, optional
        String to filter reference catalog IDs by field:
        - 'GDS' for GOODS-S (Great Observatories Deep Survey - South)
        - 'UDS' for UDS (Ultra Deep Survey)
        - 'COS' for COSMOS (Cosmic Evolution Survey)
        - None or '' for no filter (use all galaxies)
        Default is 'GDS' for GOODS-S field
    max_separation : float, optional
        Maximum angular separation in arcseconds for a valid match
        Default is 0.5 arcseconds (reasonable for CANDELS)
    save_output : bool, optional
        Whether to save the matched catalog to a FITS file
        Default is True
    output_filename : str, optional
        Name of output FITS file if save_output is True
        Default is "matched_catalog.fits"
    
    Returns: 
    matched_catalog : astropy.table.Table
        Table containing only galaxies that matched between catalogs
        Includes all columns from main catalog plus morphology labels
    """
    
    # First we load the galaxy catalogue in
    # print(f" \n Loading main catalog from: {main_catalog_path}")
    main_catalog = Table.read(main_catalog_path)
    # print(f" Main catalog has {len(main_catalog)} galaxies") # Next we check the number of galaxies involved
    
    # We perform a similar task for the reference catalogue
    # print(f"\n Loading reference catalog from: {ref_catalog_path}")
    ref_catalog = Table.read(ref_catalog_path)
    # print(f" Reference catalog has {len(ref_catalog)} galaxies")

    # print(f"\n { True if 't00_smooth_or_featured_a0_smooth_weighted_frac' in ref_catalog.colnames else False }, 't00_smooth_or_featured_a0_smooth_weighted_frac' is in reference catalog \n")
    # The commented-out code above confirms that the required classification features are present and so can be extracted
    # We will now extract these features below

    # Filter reference catalog to specific field if requested
    if field_filter:
        # print(f"\n Field specific filtering by {field_filter} is needed.")
        # print(f"\n Filtering reference catalog to field: {field_filter}")
        filtered_ref = ref_catalog[[str(id).startswith(field_filter) for id in ref_catalog['ID']]]
        print(f" Found {len(filtered_ref)} galaxies in {field_filter} field")
    else:
        filtered_ref = ref_catalog
        print(f"\n No field filter applied, using all {len(filtered_ref)} galaxies")
    
    
    # Create SkyCoord objects for position matching
    print("\n Matching catalogs by RA/Dec position...")
    
    # Note: Column names may vary between catalogs
    # Common variations: 'RA'/'ra', 'DEC'/'Dec'/'dec' I saw that the main catalogue I initialy used 
    # made use of "DEC" while the reference used "Dec" and when I used DEC for both 
    # I got an error.
    # We can djust these based on catalog column names
    
    # Get RA/Dec column names (handles common variations)
    ra_col_main = 'RA' if 'RA' in main_catalog.colnames else 'ra'
    dec_col_main = 'DEC' if 'DEC' in main_catalog.colnames else 'dec'
    if dec_col_main not in main_catalog.colnames:
        dec_col_main = 'Dec' if 'Dec' in main_catalog.colnames else 'DEC'
    
    ra_col_ref = 'RA' if 'RA' in filtered_ref.colnames else 'ra'
    dec_col_ref = 'DEC' if 'DEC' in filtered_ref.colnames else 'dec'
    if dec_col_ref not in filtered_ref.colnames:
        dec_col_ref = 'Dec' if 'Dec' in filtered_ref.colnames else 'DEC'
    
    print(f" Using columns: main(RA='{ra_col_main}', Dec='{dec_col_main}')")
    print(f"                ref(RA='{ra_col_ref}', Dec='{dec_col_ref}')")
    
    # Create coordinate objects
    main_coords = SkyCoord(ra=main_catalog[ra_col_main], dec=main_catalog[dec_col_main], unit='deg')
    ref_coords = SkyCoord(ra=filtered_ref[ra_col_ref], dec=filtered_ref[dec_col_ref], unit='deg')
    
    # Perform the matching
    idx, d2d, d3d = match_coordinates_sky(main_coords, ref_coords)
    
    # Apply separation constraint
    max_sep_deg = max_separation / 3600.0  # Perfoming conversion from arcseconds to degrees
    max_sep_constraint = d2d < (max_sep_deg * u.deg)
    
    # Get indices of good matches
    good_matches = np.where(max_sep_constraint)[0]
    
    print(f" Found {len(good_matches)} matches within {max_separation} arcseconds")
    

    # Extract the matched galaxies from main catalog
    matched_catalog = main_catalog[good_matches]
    
    # Get corresponding reference catalog entries
    matched_ref = filtered_ref[idx[good_matches]]
    


    # Create the matched catalog with morphology labels
    # print("\n Creating matched catalog...")
    # Add morphology labels from reference catalog
    # Note: This adds the raw vote fractions - this needs to be converted to classes separately
    if 't00_smooth_or_featured_a0_smooth_weighted_frac' in matched_ref.colnames:
        # Everyhting t00(for task 00)
        matched_catalog['gz_smooth_frac'] = matched_ref['t00_smooth_or_featured_a0_smooth_weighted_frac']
        matched_catalog['gz_features_frac'] = matched_ref['t00_smooth_or_featured_a1_features_weighted_frac']
        matched_catalog['gz_artifacts_frac'] = matched_ref['t00_smooth_or_featured_a2_artifact_weighted_frac']

        # Everything t02
        matched_catalog['gz_clumpy_frac'] = matched_ref['t02_clumpy_appearance_a0_yes_weighted_frac'] # Extra recommendation from Table 3 of Brooke's paper.
        matched_catalog['gz_not_clumpy_frac'] = matched_ref['t02_clumpy_appearance_a1_no_weighted_frac'] # Another from Brooke's paper 
        matched_catalog['gz_clumpy_count'] = matched_ref['t02_clumpy_appearance_count'] # Another recommendation from Table 3 of Brooke's paper

        # Everything t09
        matched_catalog['gz_not_edge_on_frac'] = matched_ref['t09_disk_edge_on_a1_no_weighted_frac']

        # Everything t12
        matched_catalog['gz_spiral_frac'] = matched_ref['t12_spiral_pattern_a0_yes_weighted_frac']
        matched_catalog['gz_spiral_count'] = matched_ref['t12_spiral_pattern_count'] # Also by Brooke's recommendation

        #Everything t16
        matched_catalog['gz_merging_frac'] = matched_ref['t16_merging_tidal_debris_a0_merging_weighted_frac']
        matched_catalog['gz_tidal_debris_frac'] = matched_ref['t16_merging_tidal_debris_a1_tidal_debris_weighted_frac']
        matched_catalog['gz_both_frac'] = matched_ref['t16_merging_tidal_debris_a2_both_weighted_frac']
        matched_catalog['gz_neither_frac'] = matched_ref['t16_merging_tidal_debris_a3_neither_weighted_frac']
        matched_catalog['gz_task16_count'] = matched_ref['t16_merging_tidal_debris_count']
        # matched_catalog['gz_irregular_frac'] = matched_ref['t04_clump_configuration_a2_cluster_or_irregular_weighted_frac']

        #Capturing the "more informative" Galaxy Zoo ID for potential image creation and naming.
        matched_catalog['gz_id'] = matched_ref['ID']

        # galaxy_id = None

        # for possible_id in ["ID", "id", "Id"]: # Extracts the non_galaxy_zoo_id
        #     if possible_id in matched_catalog.colnames:
        #         galaxy_id = possible_id


        # matched_catalog[f"{galaxy_id}"] = matched_catalog[f"{galaxy_id}"].astype(str) # This converts the entire target column to a Unicode string type before we start changing the ID
        # # THis would throw up and error if not done before changing the values

        # # print('\n Before: ')
        # # print( matched_catalog[f'{galaxy_id}'] )

        
        # for numbered_id in range( len(matched_catalog) ): ## This helps us renumbers the galaxies by their field
        #     non_galaxy_zoo_id = f"{field_filter}_{matched_catalog[f'{galaxy_id}'][numbered_id]}"
        #     matched_catalog[f'{galaxy_id}'][numbered_id] = non_galaxy_zoo_id

        
        # # print("\n After: ")
        # # print( matched_catalog['ID'] )


        
        print(f" Added Galaxy Zoo vote fractions to catalog")
        # print(f" Available columns: { matched_catalog["gz_id"] }") #Uncomment when absolutely necessary for debugging
    else:
        print(f" Warning: Expected Galaxy Zoo columns not found")
        print(f" Available columns: {matched_ref.colnames}") #Uncomment when absolutely necessary for debugging
    
    
    
    
    
    # Add separation distance (for quality checking)
    matched_catalog['match_separation_arcsec'] = d2d[good_matches].arcsec
    
    print(f"\n MATCHING COMPLETE")
    # print(f"\n Final matched catalog size: {len(matched_catalog)} galaxies")
    # print(f" - Each galaxy has RA/Dec from main catalog")
    # print(f" - Galaxy Zoo vote fractions added as columns")
    print(f" - Match separation: min={np.min(matched_catalog['match_separation_arcsec']):.2f} arcsec, "
          f"max={np.max(matched_catalog['match_separation_arcsec']):.2f} arcsec")
    # print(f"\n {matched_catalog.colnames} ")
    
    # Save the matched catalog (this is entirely optional, but I like it)
    if save_output:
        matched_catalog.write(output_filename, overwrite=True)
        print(f"\n Saved matched catalog to: {output_filename}") #Uncomment when absolutely necessary for debugging
    
    return matched_catalog



if __name__ == "__main__":
  
    MAIN_CATALOG = "/Users/holyphysics/Desktop/Galaxy_Classification/gds_merged_v1.1.fits"
    REF_CATALOG = "/Users/holyphysics/Desktop/Galaxy_Classification/gz_candels_table_2_main_release.fits"
    

    matched_catalogue = catalogue_matcher(
        main_catalog_path=MAIN_CATALOG,
        ref_catalog_path=REF_CATALOG,
        field_filter='GDS',
        max_separation=1, # Change this to 1 for larger field search
        save_output=True,
        output_filename="matched_catalog.fits"
    )

    # print(matched_catalogue.colnames)
    
    # Add morphology classes
    # matched_with_classes = add_morphology_classes(matched)
    
    # Save the fully classified catalog
    # matched_with_classes.write("classified_matched_catalog.fits", overwrite=True)
    
    print("\n Done!")