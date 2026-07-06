# I need this for labelling the galaxies for classification
"""
Author: Chidiebere N. Okafor
Purpose: This script houses a function whose sole purpose is to convert identified Galaxy Zoo vote fractions
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

Otherwise, one can either make this part domant by way of commenting it out and proceed as:
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
        smooth = catalog['gz_smooth_frac'][i]
        features = catalog['gz_features_frac'][i]
        spiral = catalog['gz_spiral_frac'][i]
        irregular = catalog['gz_irregular_frac'][i]
        number_of_spiral_classifiers = catalog['gz_spiral_count'][i] # Extra recommendation from Table 3 of Brooke's paper.
        number_of_irregular_classifiers = catalog['gz_irregular_count'][i]
        
        # Classify based on vote thresholds
        # if smooth > smooth_threshold and features < features_threshold:
        if smooth >= smooth_threshold:
            morph = "Elliptical"
            clean = True
        # elif features > features_threshold and spiral > spiral_threshold:
        elif (spiral >= spiral_threshold) or (number_of_spiral_classifiers >= 10) or (features >=0.4):
            morph = "Spiral"
            clean = True
        # elif irregular > irregular_threshold:
        elif (irregular > irregular_threshold) or (number_of_irregular_classifiers >=10) or (features >=0.4):
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
    matched_with_classes.write("morphology_matched_catalog.fits", overwrite=True)
    print(matched_with_classes['morphology'])
    
    print("\n Done!")