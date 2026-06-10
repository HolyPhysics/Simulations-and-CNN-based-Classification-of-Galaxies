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



def catalogue_matcher(main_catalog_path, ref_catalog_path, field_filter='GDS', 
                      max_separation=0.5, save_output=True, output_filename="matched_catalog.fits"):
    """
    Match a main catalog against a reference catalog using RA/Dec positions.
    
    Parameters
    ----------
    main_catalog_path : str
        Path to the main catalog FITS file (for example, CANDELS GOODS-S catalog)
    ref_catalog_path : str
        Path to the reference catalog FITS file (for example, Galaxy Zoo CANDELS catalog)
    field_filter : str, optional
        String to filter reference catalog IDs (for example, 'GDS' for GOODS-S, 'UDS' for UDS)
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
    print(f" \n Loading main catalog from: {main_catalog_path}")
    main_catalog = Table.read(main_catalog_path)
    print(f" Main catalog has {len(main_catalog)} galaxies") # Next we check the number of galaxies involved
    
    # We perform a similar task for the reference catalogue
    print(f"\n Loading reference catalog from: {ref_catalog_path}")
    ref_catalog = Table.read(ref_catalog_path)
    print(f" Reference catalog has {len(ref_catalog)} galaxies")
    
    # Filter reference catalog to specific field if requested
    if field_filter:
        print("\n Field specific filtering by UDS is needed.")
        print(f"\n Filtering reference catalog to field: {field_filter}")
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
    


    # # Create the matched catalog with morphology labels
    # print("\n Creating matched catalog...")
    # # Add morphology labels from reference catalog
    # # Note: This adds the raw vote fractions - this needs to be converted to classes separately
    # if 't00_smooth_or_featured_a0_smooth_weighted_frac' in matched_ref.colnames:
    #     matched_catalog['gz_smooth_frac'] = matched_ref['t00_smooth_or_featured_a0_smooth_weighted_frac']
    #     matched_catalog['gz_features_frac'] = matched_ref['t00_smooth_or_featured_a1_features_weighted_frac']
    #     matched_catalog['gz_spiral_frac'] = matched_ref['t12_spiral_pattern_a0_yes_weighted_frac']
    #     matched_catalog['gz_irregular_frac'] = matched_ref['t04_clump_configuration_a2_cluster_or_irregular_weighted_frac']
    #     matched_catalog['gz_id'] = matched_ref['ID']
        
    #     print(f" Added Galaxy Zoo vote fractions to catalog")
    # else:
    #     print(f" Warning: Expected Galaxy Zoo columns not found")
    #     print(f" Available columns: {matched_ref.colnames}") #Uncomment when absolutely necessary for debugging
    
    
    
    
    
    # Add separation distance (for quality checking)
    matched_catalog['match_separation_arcsec'] = d2d[good_matches].arcsec
    
    print(f"\n MATCHING COMPLETE")
    print(f"\n Final matched catalog size: {len(matched_catalog)} galaxies")
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


# I need this for labelling the galaxies for classification
# def add_morphology_classes(catalog, smooth_threshold=0.6, features_threshold=0.5, 
#                            spiral_threshold=0.5, irregular_threshold=0.5):
#     """
#     Convert Galaxy Zoo vote fractions into morphology classes.
    
#     Parameters
#     ----------
#     catalog : astropy.table.Table
#         Catalog containing Galaxy Zoo vote fraction columns
#     smooth_threshold : float, optional
#         Threshold for classifying as elliptical (smooth fraction > this)
#     features_threshold : float, optional
#         Threshold for classifying as spiral (features fraction > this)
#     spiral_threshold : float, optional
#         Threshold for spiral arm confirmation
#     irregular_threshold : float, optional
#         Threshold for irregular classification
    
#     Returns
#     -------
#     catalog : astropy.table.Table
#         Input catalog with added 'morphology' and 'morphology_clean' columns
#     """
    
#     print("\nAdding morphology classifications...")
    
#     # Initialize morphology column
#     morphology = []
#     clean_flags = []
    
#     for i in range(len(catalog)):
#         smooth = catalog['gz_smooth_frac'][i]
#         features = catalog['gz_features_frac'][i]
#         spiral = catalog['gz_spiral_frac'][i]
#         irregular = catalog['gz_irregular_frac'][i]
        
#         # Classify based on vote thresholds
#         if smooth > smooth_threshold and features < features_threshold:
#             morph = "Elliptical"
#             clean = True
#         elif features > features_threshold and spiral > spiral_threshold:
#             morph = "Spiral"
#             clean = True
#         elif irregular > irregular_threshold:
#             morph = "Irregular"
#             clean = True
#         else:
#             morph = "Uncertain"
#             clean = False
        
#         morphology.append(morph)
#         clean_flags.append(clean)
    
#     # Add columns to catalog
#     catalog['morphology'] = morphology
#     catalog['morphology_clean'] = clean_flags
    
#     # Print summary
#     print(f"  Classification summary:")
#     for morph in ["Elliptical", "Spiral", "Irregular", "Uncertain"]:
#         count = sum(1 for m in morphology if m == morph)
#         print(f"    {morph}: {count} galaxies ({100*count/len(catalog):.1f}%)")
    
#     print(f"  Clean sample (well-classified): {sum(clean_flags)} galaxies")
    
#     return catalog



if __name__ == "__main__":
  
    MAIN_CATALOG = "/Users/holyphysics/Desktop/Galaxy_Classification/gds_merged_v1.1.fits"
    REF_CATALOG = "/Users/holyphysics/Desktop/Galaxy_Classification/gz_candels_table_2_main_release.fits"
    

    matched_catalogue = catalogue_matcher(
        main_catalog_path=MAIN_CATALOG,
        ref_catalog_path=REF_CATALOG,
        field_filter='GDS',
        max_separation=0.5,
        save_output=True,
        output_filename="matched_catalog.fits"
    )
    
    # Add morphology classes
    # matched_with_classes = add_morphology_classes(matched)
    
    # Save the fully classified catalog
    # matched_with_classes.write("classified_matched_catalog.fits", overwrite=True)
    
    print("\n Done!")