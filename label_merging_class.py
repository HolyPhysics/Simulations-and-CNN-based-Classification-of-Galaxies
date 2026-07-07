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

def add_merging_classes(catalog, smooth_threshold=0.8, features_threshold=0.5, 
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
    
    print("\nAdding Merger/Non-merger morphology classifications...")
    
    # Initialize morphology column
    merger_morphology = []
    clean_flags = []
    
    for i in range(len(catalog)):
        f_merger = safe_column(catalog,'gz_merging_frac')[i]
        f_tidal_debris = safe_column(catalog,'gz_tidal_debris_frac')[i]
        f_both = safe_column(catalog,'gz_both_frac')[i]
        f_neither = safe_column(catalog,'gz_neither_frac')[i]
        f_count = safe_column(catalog,'gz_task16_count')[i]


        f_interacting = f_merger + f_tidal_debris + f_both
        
        # There are other ways I can do this though. I can, for example, start with a threshold for f_neither and f_count
        # if f_interacting >= 0.5 and f_count >= 5:
        #     morph = "Merger"
        #     clean = True
        # else:
        #     morph = "Nonmerger"
        #     clean = True

        if (f_neither >= 0.8 and f_count >= 10):
            morph = "Nonmerger"
            clean = True
        else:
            morph = "Merger"
            clean = True
        
        merger_morphology.append(morph)
        clean_flags.append(clean)
    
    # Add columns to catalog
    catalog['merging_morphology'] = merger_morphology
    catalog['morphology_clean'] = clean_flags
    
    # Print summary
    print(f"Information on Galaxy Morphology:")
    for morph in ["Merger", "Nonmerger"]:
        count = sum(1 for m in merger_morphology if m == morph) #counts and sums all included galaxies
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
    matched_with_classes = add_merging_classes(matched_catalog) 
    
    # Save the fully classified catalog
    matched_with_classes.write("morphology_matched_catalog.fits", overwrite=True)
    print(matched_with_classes['merging_morphology'])
    
    print("\n Done!")