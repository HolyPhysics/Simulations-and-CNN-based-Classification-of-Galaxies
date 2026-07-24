# I need this for labelling the galaxies for classification
"""
Author: Chidiebere N. Okafor
Purpose: This script houses a function whose sole purpose is to convert identified Galaxy Zoo vote fractions,
         following the directions in Brooke Simmon's paper, into morphology classes.

Usage
-----
python3 label_galaxy_morphology.py [required positionl arguments] [oprtions]

Positional Arguments
--------------------
catalog: str
    This is a .fits file of the "matched catalogue" the galaxies are drawn from.

Options
-------
--smooth_threshold FLOAT      smooth fraction of the galaxies to select (default: 0.8)
--features_threshold FLOAT      fraction of the galaxies with features close to "rounded with no sign og disc" (default: 0.5)
--spiral_threshold FLOAT        spiral fraction of the galaxies to select (default: 0.8)
--irregular_threshold FLOAT     irregular fraction of the galaxies to select (default: 0.5)
--help, -h

Example
-------
(Since I already have all required arguments outlined in the "if __name__ == '__main__':" part, the following is enough)
python3 label_galaxy_morphology.py

Otherwise, one can either make this part domant, by way of commenting it out, and proceed as:
python3 label_galaxy_morphology.py matched_catalog.fits --smooth_threshold 0.6 features_threshold 0.6
"""

from match_catalogue import catalogue_matcher

def add_morphology_classes(catalog, smooth_threshold=0.8, features_threshold=0.5, 
                           spiral_threshold=0.8, irregular_threshold=0.5): # Go back to Brooke's paper for clarification on the thresholds
    """
    Convert Galaxy Zoo vote fractions into morphology classes.
    
    Parameters
    ----------
    catalog : astropy.table.Table
        Catalog containing Galaxy Zoo vote fraction columns. For my usecase here, this will be the matched catalogue
    smooth_threshold : float, optional
        Threshold for classifying as elliptical (smooth fraction > this)
    features_threshold : float, optional
        Threshold for classifying as spiral (features fraction > this)
    spiral_threshold : float, optional
        Threshold for spiral arm confirmation
    irregular_threshold : float, optional
        Threshold for irregular classification
    
    Returns
    -------
    catalog : astropy.table.Table
        Input catalog with added 'morphology' and 'morphology_clean' columns
    """
    
    print("\nAdding morphology classifications...")
    
    # Initialize morphology column
    morphology = []
    clean_flags = []
    
    for i in range(len(catalog)):
        f_smooth = catalog['gz_smooth_frac'][i]
        f_features = catalog['gz_features_frac'][i]
        f_artifacts = catalog['gz_artifacts_frac'][i]

        f_spiral = catalog['gz_spiral_frac'][i]
        number_of_spiral_classifiers = catalog['gz_spiral_count'][i] # Extra recommendation from Table 3 of Brooke's paper.

        f_clumpy = catalog['gz_clumpy_frac'][i]
        f_not_clumpy = catalog['gz_not_clumpy_frac'][i]
        number_of_clumpy_classifiers = catalog['gz_clumpy_count'][i]
        
        f_not_edge_on = catalog['gz_not_edge_on_frac'][i]

        f_merging = safe_column(catalog,'gz_merging_frac')[i]
        f_tidal_debris = safe_column(catalog,'gz_tidal_debris_frac')[i]
        f_merging_tidal = f_merging + f_tidal_debris

        # As directed/identified in the paper, we first exclude artifacts.
        # if f_artifacts >= 0.5:
        #     morph = "Uncertain"
        #     clean = False
        #     morphology.append(morph)
        #     clean_flags.append(clean)
        #     continue
        
        # Classify based on vote thresholds
        # if smooth > smooth_threshold and features < features_threshold:
        if f_smooth >= smooth_threshold:
            morph = "Elliptical"
            clean = True
        elif  f_spiral > spiral_threshold: # and features > features_threshold:
        # elif (f_features >= 0.4 and 
        #     # f_not_clumpy >= 0.3 and
        #     f_not_edge_on >= 0.5 and
        #     f_spiral >= 0.8 #and number_of_spiral_classifiers >= 10
        #     ):
            morph = "Spiral"
            clean = True
        elif f_clumpy > irregular_threshold:
        # elif (f_features >= 0.4 and
        #     # f_smooth < 0.5 and
        #     # f_spiral < 0.5 and
        #     f_clumpy >= 0.4):
            morph = "Irregular"
            clean = True
        else:
            morph = "Uncertain"
            clean = False
        
        morphology.append(morph)
        clean_flags.append(clean)
    
    # Add columns to catalog
    catalog['morphology'] = morphology
    catalog['morphology_clean'] = clean_flags
    
    # Print summary
    print(f"Information on Galaxy Morphology:")
    for morph in ["Elliptical", "Spiral", "Irregular", "Uncertain"]:
        count = sum(1 for m in morphology if m == morph) #counts and sums all included galaxies
        print(f" {morph}: {count} galaxies ({100*count/len(catalog):.1f}%) ")
    
    print(f"  Clean sample (well-classified): {sum(clean_flags)} galaxies")
    
    return catalog



def safe_column(table, colname, default=0.0):
    """Return a column from an Astropy Table, or an array of `default` if missing."""
    if colname in table.colnames:
        return table[colname]
    else:
        return np.full(len(table), default)



if __name__ == "__main__":
  
    MAIN_CATALOG = "/Users/holyphysics/Desktop/Galaxy_Classification/gds_merged_v1.1.fits" ## Testing with GDS field
    REF_CATALOG = "/Users/holyphysics/Desktop/Galaxy_Classification/gz_candels_table_2_main_release.fits"
    

    matched_catalog = catalogue_matcher(
        main_catalog_path=MAIN_CATALOG,
        ref_catalog_path=REF_CATALOG,
        field_filter='GDS',
        max_separation=1, # Change this to 1 for larger field search
        save_output=True,
        output_filename="matched_catalog.fits"
    )

    # print(matched_catalog.colnames)
    
    # Add morphology classes # The name of the outpus file is "matched_catalog not"
    matched_with_classes = add_morphology_classes(matched_catalog) 
    
    # Save the fully classified catalog
    # matched_with_classes.write("matched_catalog.fits", overwrite=True)
    # print(matched_with_classes['morphology'])

    if "morphology" in matched_with_classes.colnames:
        print("Morphology exists")
    
    print("\n Done!")