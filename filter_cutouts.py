# """
# Author: Chidiebere N. Okafor; Prof. Elizabeth McGrath

# Module for creating thumbnail cutouts from astronomical FITS images.

# This module takes a catalog of galaxy positions (RA, Dec) and creates
# small cutout images around each galaxy from a given filter image.

# Purpose:
#     - Extract small FITS cutouts centered on each galaxy
#     - Save cutouts as individual FITS files for later RGB combination
#     - Handle error arrays and WCS information properly
# """

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.units as u
import os
from typing import Tuple, List
import time



def make_filter_cutouts(catalog, image_paths, bands, output_dirs, 
                             box_radius=103.5, rootname="candels", 
                             prefix="", suffix="", overwrite=False,
                             galaxy_indices=None, verbose=True):
    """
    SUPER-FAST: Create thumbnail cutouts for all galaxies across all filters simultaneously.
    
    This function opens all FITS files ONCE and keeps them open, then extracts
    cutouts for all filters for each galaxy in a single pass. This is MUCH faster
    than calling make_filter_cutouts separately for each filter.
    
    Parameters
    ----------
    catalog : astropy.table.Table
        Catalog containing galaxies with columns for RA, Dec, and ID
        Must have columns: 'RA', 'DEC', and 'ID' (or 'NUMBER')
    image_paths : dict
        Dictionary mapping filter names to FITS file paths
        Example: {'red': '/path/to/f160w.fits', 'green': '/path/to/f125w.fits'}
    bands : dict
        Dictionary mapping filter names to band names for output files
        Example: {'red': 'f160w', 'green': 'f125w', 'blue': 'f606w'}
    output_dirs : dict
        Dictionary mapping filter names to output directories
        Example: {'red': 'cutouts_red', 'green': 'cutouts_green', 'blue': 'cutouts_blue'}
    box_radius : int, optional
        Radius of cutout in pixels (final size will be 2 * box_radius)
        Default is 103.5 pixels (~207x207 pixel cutouts)
    rootname : str, optional
        Root name for output files (for example, "candels", "egs", "uds")
        Default is "candels"
    prefix : str, optional
        Optional prefix for output filenames (for example, "lrd_")
        Default is "" (no prefix)
    suffix : str, optional
        Optional suffix for output filenames (for example, "v1")
        Default is "" (no suffix)
    overwrite : bool, optional
        If True, overwrite existing cutouts. If False, skip existing files.
        Default is False
    galaxy_indices : list, optional
        List of indices to process. If None, process all galaxies.
        Default is None (process all)
    verbose : bool, optional
        If True, print progress updates. Default is True.
    
    Returns:
    -------
    output_files : dict
        Dictionary mapping filter names to lists of created cutout FITS files
    successful_counts : dict
        Dictionary mapping filter names to number of successful cutouts created
    
    Notes:
    -----
    Output filename format: rootname.prefixCATID.band.suffix.fits
    Example: candels.12345.f160w.fits
    
    Error files are saved with '.err' before .fits:
    Example: candels.12345.f160w.err.fits
    
    This function processes ALL filters simultaneously for maximum speed!
    """
    
    # Determine ID column name
    # Catalogs use different names for the ID column
    # Common possibilities: 'ID', 'id', 'NUMBER', 'Number'
    id_col = None
    
    # "ID" and "gz_id" columns are two different columns
    for possible_name in ['ID', 'id', 'NUMBER', 'Number']:
        if possible_name in catalog.colnames:
            id_col = possible_name
            break
    
    if id_col is None:
        raise ValueError("Catalog has no ID column! Found: " + str(catalog.colnames))
    
    # Now, extract Coordinates and IDS from catalog 
    # Get RA, Dec, and IDs (handling different column name variations)
    ra_col = 'RA' if 'RA' in catalog.colnames else 'ra'
    dec_col = 'DEC' if 'DEC' in catalog.colnames else 'dec'
    
    # Also check for 'Dec' (capital D, rest lowercase). Again, necessary since I've seen this 
    # very variation used and had to debug an error descended from it.
    if dec_col not in catalog.colnames:
        dec_col = 'Dec' if 'Dec' in catalog.colnames else 'DEC'
    
    ra_list = catalog[ra_col]
    dec_list = catalog[dec_col]
    id_list = catalog[id_col]
    
    # If no indices specified, process all
    if galaxy_indices is None:
        galaxy_indices = list(range(len(catalog)))
    
    total_galaxies = len(galaxy_indices)
    
    if verbose:
        print(f"Galaxies to process: {total_galaxies}")
        print(f"Box size: {box_radius*2}x{box_radius*2} pixels")
        print(f"Filters: {', '.join(bands.values())}")
        print(f"Overwrite mode: {overwrite}")
    
    # Create output directories
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Open ALL FITS files at once with memmap (stays open for all galaxies)
    if verbose:
        print("Opening FITS files...")
    
    hdus = {}
    science_data = {}
    wcs_objects = {}
    shapes = {}
    
    start_time = time.time() # Starts counting the time for opening the files
    
    for filter_name, img_path in image_paths.items():
        band = bands[filter_name]
        if verbose:
            print(f"  Loading {band}...", end=" ", flush=True)
        
        try:
            # memmap=True is of dire importance for large FITS files!
            hdu = fits.open(img_path, memmap=True)
            hdus[filter_name] = hdu
            
            # Check for different formats
            # Figured the previous code was specific to Candels codes with 
            # fits file in "SCI". However, when testing with GOODS_south data where
            # that was not the case, the previous code threw errors. Searched and found the fits files were in 
            # something called "PRIMARY HDU (extension 0)", not 'SCI'
            # So it's much better to use hdu[0] directly
            if 'SCI' in hdu:
                science_data[filter_name] = hdu['SCI'].data
                wcs_objects[filter_name] = WCS(hdu[0].header, hdu)
                if verbose:
                    print(" CANDELS format")
            elif len(hdu) > 1 and 'SCI' in hdu[1].name:
                science_data[filter_name] = hdu[1].data
                wcs_objects[filter_name] = WCS(hdu[1].header)
                if verbose:
                    print(" JWST format")
            else:
                # HLSP and standard FITS files have data in HDU 0
                science_data[filter_name] = hdu[0].data
                wcs_objects[filter_name] = WCS(hdu[0].header)
                if verbose:
                    print(" Primary HDU format")
            
            shapes[filter_name] = science_data[filter_name].shape
            if verbose:
                print(f"  Dimensions: {shapes[filter_name][1]}x{shapes[filter_name][0]} pixels")
            
        except Exception as e:
            if verbose:
                print(f" ERROR: {e}")
            raise
    
    if verbose:
        print(f"\nAll files opened in {time.time() - start_time:.1f} seconds\n")
    
    # Process each galaxy
    box_size_pixels = int(box_radius * 2)
    
    # Initialize counters for each filter
    output_files = {filter_name: [] for filter_name in image_paths.keys()}
    successful_counts = {filter_name: 0 for filter_name in image_paths.keys()}
    skipped_counts = {filter_name: 0 for filter_name in image_paths.keys()}
    skipped_outside = 0
    
    if verbose:
        print("Extracting cutouts...")
    
    for i, idx in enumerate(galaxy_indices):
        galaxy_id = id_list[idx]
        ra = ra_list[idx]
        dec = dec_list[idx]
        
        # Progress update
        if verbose and (i < 5 or i % 10 == 0 or i == total_galaxies - 1):
            percent = (i + 1) / total_galaxies * 100
            print(f"  [{i+1:4d}/{total_galaxies}] ({percent:5.1f}%) Galaxy {galaxy_id}...", end=" ", flush=True)
        
        # Create SkyCoord object for the galaxy center
        center = SkyCoord(ra, dec, frame='icrs', unit='deg')
        
        # Check if galaxy is within bounds for ALL filters
        within_bounds = True
        for filter_name in image_paths.keys():
            try:
                x_pixel, y_pixel = center.to_pixel(wcs_objects[filter_name])
                ny, nx = shapes[filter_name]
                if x_pixel < 0 or x_pixel >= nx or y_pixel < 0 or y_pixel >= ny:
                    within_bounds = False
                    break
            except:
                within_bounds = False
                break
        
        if not within_bounds:
            if verbose and (i < 5 or i % 10 == 0):
                print(" outside bounds")
            skipped_outside += 1
            continue
        
        # Check if cutouts already exist for ALL filters (skip if overwrite=False)
        if not overwrite:
            all_exist = True
            for filter_name in output_dirs.keys():
                band = bands[filter_name]
                if suffix:
                    science_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.{suffix}.fits"
                    error_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.{suffix}.err.fits"
                else:
                    science_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.fits"
                    error_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.err.fits"
                
                science_path = os.path.join(output_dirs[filter_name], science_filename)
                error_path = os.path.join(output_dirs[filter_name], error_filename)
                
                if not (os.path.exists(science_path) and os.path.exists(error_path)):
                    all_exist = False
                    break
            
            if all_exist:
                if verbose and (i < 5 or i % 10 == 0):
                    print(" already exists")
                # Add to output files and increment counters
                for filter_name in image_paths.keys():
                    band = bands[filter_name]
                    if suffix:
                        science_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.{suffix}.fits"
                    else:
                        science_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.fits"
                    science_path = os.path.join(output_dirs[filter_name], science_filename)
                    output_files[filter_name].append(science_path)
                    successful_counts[filter_name] += 1
                continue
        
        # Extract and save cutouts for each filter
        success = True
        for filter_name in image_paths.keys():
            try:
                # Create cutout
                science_cutout = Cutout2D(
                    science_data[filter_name],
                    center,
                    (box_size_pixels, box_size_pixels),
                    wcs=wcs_objects[filter_name]
                )
                
                # Construct output filename
                band = bands[filter_name]
                if suffix:
                    science_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.{suffix}.fits"
                    error_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.{suffix}.err.fits"
                else:
                    science_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.fits"
                    error_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.err.fits"
                
                science_path = os.path.join(output_dirs[filter_name], science_filename)
                error_path = os.path.join(output_dirs[filter_name], error_filename)
                
                # Save science cutout
                science_hdu = fits.PrimaryHDU(science_cutout.data, header=science_cutout.wcs.to_header())
                science_hdu.writeto(science_path, overwrite=True)
                
                # Save error cutout (placeholder - we don't have error data in HLSP)
                # Create a simple error array if none exists
                if 'ERR' in hdus[filter_name]:
                    error_cutout = Cutout2D(
                        hdus[filter_name]['ERR'].data,
                        center,
                        (box_size_pixels, box_size_pixels),
                        wcs=wcs_objects[filter_name]
                    )
                    error_hdu = fits.PrimaryHDU(error_cutout.data, header=error_cutout.wcs.to_header())
                    error_hdu.writeto(error_path, overwrite=True)
                
                output_files[filter_name].append(science_path)
                successful_counts[filter_name] += 1
                
            except Exception as e:
                if verbose and (i < 5 or i % 10 == 0):
                    print(f" error in {band}")
                success = False
                break
        
        if success:
            if verbose and (i < 5 or i % 10 == 0):
                print(" done")
    
    # Close FITS files
    if verbose:
        print("\n Closing FITS files...")
    for hdu in hdus.values():
        try:
            hdu.close()
        except:
            pass
    
    # For summary of filter cutouts(filter cutout details)
    if verbose:
        print("\n EXTRACTION COMPLETE")

        for filter_name in image_paths.keys():
            band = bands[filter_name]
            print(f"  {band.upper()}: {successful_counts[filter_name]} successful, {skipped_counts[filter_name]} skipped")

        print(f"  Skipped (outside bounds): {skipped_outside}")
        print(f"  Output directories:")

        for filter_name, dir_path in output_dirs.items():
            print(f"     {bands[filter_name]}: {dir_path}")
    
    return output_files, successful_counts













# def make_filter_cutouts(catalog, image_path, band, output_dir, box_radius=103.5, 
#                         rootname="candels", prefix="", suffix="", 
#                         overwrite=False) -> Tuple[List[str], int]:
#     """
#     Create thumbnail cutouts for all galaxies in a catalog from a single filter image.
    
#     This function extracts small FITS cutouts centered on each galaxy's position
#     and saves them as individual files. It handles both science and error arrays.
    
#     Parameters
#     ----------
#     catalog : astropy.table.Table
#         Catalog containing galaxies with columns for RA, Dec, and ID
#         Must have columns: 'RA', 'DEC', and 'ID' (or 'NUMBER')
#     image_path : str
#         Path to the FITS file containing the science image for this filter
#         Example: "/data/candels_f160w_sci.fits"
#     band : str
#         Filter name (for example, 'f160w', 'f125w', 'f606w')
#         This will appear in output filenames
#     output_dir : str
#         Directory where cutout FITS files will be saved
#         Example: "cutouts_red"
#     box_radius : int, optional
#         Radius of cutout in pixels (final size will be 2 * box_radius)
#         Default is 300 pixels( THat is 600x600 pixel cutouts)
#     rootname : str, optional
#         Root name for output files (for example, "candels", "egs", "uds")
#         Default is "candels"
#     prefix : str, optional
#         Optional prefix for output filenames (for example, "lrd_")
#         Default is "" (no prefix)
#     suffix : str, optional
#         Optional suffix for output filenames (for example, "v1")
#         Default is "" (no suffix)
#     overwrite : bool, optional
#         If True, overwrite existing cutouts. If False, skip existing files.
#         Default is False
    
#     Returns:
#     -------
#     output_files : list
#         List of paths to created cutout FITS files
#     cutout_count : int
#         Number of successful cutouts created
    
#     Notes:
#     -----
#     Output filename format: rootname.prefixCATID.band.suffix.fits
#     Example: candels.12345.f160w.fits
    
#     Error files are saved with '.err' before .fits:
#     Example: candels.12345.f160w.err.fits
#     """
    
#     # Create output directory
#     # print(f" Making cutouts for band: {band}")
    
#     # Create output directory if it doesn't exist
#     if os.path.exists(output_dir):
#         print(f"Output directory '{output_dir}' already exists")
#     else:
#         os.makedirs(output_dir)
#         print(f"Created output directory: {output_dir}")
    
#     # Next we need to determine ID column name
#     # Catalogs use different names for the ID column
#     # Common possibilities: 'ID', 'id', 'NUMBER', 'Number'
#     id_col = None
    
#     # "ID" and "gz_id" columns are two different columns
#     for possible_name in ['ID', 'id', 'NUMBER', 'Number']:
#         if possible_name in catalog.colnames:
#             id_col = possible_name
#             break
    
#     if id_col is None:
#         raise ValueError("Catalog has no ID column! Found: " + str(catalog.colnames))
    
#     # print(f" Using '{id_col}' as galaxy ID column")
    
#     # Now, extract Coordinates and IDS from catalog 
#     # Get RA, Dec, and IDs (handling different column name variations)
#     ra_col = 'RA' if 'RA' in catalog.colnames else 'ra'
#     dec_col = 'DEC' if 'DEC' in catalog.colnames else 'dec'
    
#     # Also check for 'Dec' (capital D, rest lowercase). Again, necessary since I've seen this 
#     # very varition used and had to debug an error descended from it.
#     if dec_col not in catalog.colnames:
#         dec_col = 'Dec' if 'Dec' in catalog.colnames else 'DEC'
    
#     ra_list = catalog[ra_col]
#     dec_list = catalog[dec_col]
#     id_list = catalog[id_col]
    
#     # print(f" Using RA column: '{ra_col}', Dec column: '{dec_col}'")
#     # print(f" Processing {len(ra_list)} galaxies...")
    
   
#     # Open the fits file and extract data
#     # print(f"\n Opening FITS image: {image_path}")

#     start_time = time.time()

#     try:
#         # Use memmap=True to handle large files efficiently. 
#         # It allows reading large FITS files without loading everything into memory, fixing the "Bad file descriptor" error
#         with fits.open(image_path, memmap=True) as hdu:
#             # Check for different formats
#             # Figured the previous code was specific to Candels codes with 
#             # fits file in "SCI". However, when testing with GOODS_south data where
#             # that was not the case, the previous code threw errors. Searched and found the fits files were in 
#             # something called "PRIMARY HDU (extension 0)", not 'SCI'
#             # So it's much better to use hdu[0] directly
#             if 'SCI' in hdu:
#                 science_data = hdu['SCI'].data
#                 image_wcs = WCS(hdu[0].header, hdu)
#                 print("Format: CANDELS (SCI extension)")
#             elif len(hdu) > 1 and 'SCI' in hdu[1].name:
#                 science_data = hdu[1].data
#                 image_wcs = WCS(hdu[1].header)
#                 print("Format: JWST (extension 1)")
#             else:
#                 # HLSP and standard FITS files have data in HDU 0
#                 science_data = hdu[0].data
#                 image_wcs = WCS(hdu[0].header)
#                 print("Format: Primary HDU (HLSP or standard)")
            
#             # Get error data if available
#             error_data = hdu['ERR'].data if 'ERR' in hdu else np.zeros_like(science_data)
            
#             ny, nx = science_data.shape
#             print(f"Image dimensions: {nx} x {ny} pixels")
#             print(f"Data type: {science_data.dtype}")
#             print(f"Data range: {science_data.min():.2f} to {science_data.max():.2f}")
            
#     except OSError as e:
#         print(f" OS Error reading FITS file: {e}")
#         print(" This could be due to:")
#         print("  - File corruption (try re-downloading)")
#         print("  - Insufficient memory (try using memmap=True)")
#         print("  - File system issues")
#         raise
#     except Exception as e:
#         print(f" Error reading FITS file: {e}")
#         raise

#     print(f"\n All files opened in {time.time() - start_time:.1f} seconds\n")
    
#     # Create cutouts for each galaxy 
#     output_files = []
#     successful_cutouts = 0
#     skipped_outside = 0
#     skipped_existing = 0
#     box_size_pixels = box_radius * 2  # Convert radius to side length
    
#     print(f"\n Creating cutouts (box size: {box_size_pixels}x{box_size_pixels} pixels)...")
#     print(f" Overwrite mode: {overwrite}")
        
#     for i in range(len(ra_list)):
#         galaxy_id = id_list[i]
#         ra = ra_list[i]
#         dec = dec_list[i]
        
#         # Print progress every 100 galaxies or for first few. Just as a tracker/a self-sufficient 
#         # progress indicator of some sort.
#         if i < 5 or i % 100 == 0:
#             print(f" Processing galaxy {i+1}/{len(ra_list)}: ID={galaxy_id}")
        
#         # Construct output filename
#         if suffix:
#             science_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.{suffix}.fits"
#             error_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.{suffix}.err.fits"
#         else:
#             science_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.fits"
#             error_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.err.fits"
        
#         science_path = os.path.join(output_dir, science_filename)
#         error_path = os.path.join(output_dir, error_filename)
        
#         # Check if cutout already exists (skip if overwrite=False)
#         if not overwrite and os.path.exists(science_path) and os.path.exists(error_path):
#             if i < 5 or i % 100 == 0:
#                 print(f" already exists")
#             output_files.append(science_path)
#             successful_cutouts += 1
#             skipped_existing += 1
#             continue
        
#         # Create SkyCoord object for the galaxy center
#         center = SkyCoord(ra, dec, frame='icrs', unit='deg')
        
#         # Convert to pixel coordinates to check if inside image
#         try:
#             x_pixel, y_pixel = center.to_pixel(image_wcs)
#         except Exception as e:
#             print(f"\n Warning: Could not convert coordinates for galaxy {galaxy_id}: {e}")
#             skipped_outside += 1
#             continue
        
#         # Check if galaxy falls within image bounds(the mosiac)
#         if (x_pixel < 0 or x_pixel >= nx or y_pixel < 0 or y_pixel >= ny):
#             if i < 5 or i % 100 == 0:
#                 print(f" outside bounds")
#             skipped_outside += 1
#             continue

#         # Is this code above doing exactly as this code below. Hmmmm... I'll need to return to this later, Chidiebere.
#         # if((cutout_cenpix[0]+1 < nx) & (cutout_cenpix[0]+1 > 0) & (cutout_cenpix[1]+1 < ny) & (cutout_cenpix[1]+1 > 0)): 
#         #     box[i]*=2
#         #     if(box[i] > 666): 
#         #         box[i]=666
#         #     if(box[i] < 200):
#         #         box[i]=200
#         # 
        
#         # Create the cutouts
#         try:
#             science_cutout = Cutout2D(science_data, center, (box_size_pixels, box_size_pixels), wcs=image_wcs)
            
#             error_cutout = Cutout2D(error_data, center, (box_size_pixels, box_size_pixels), wcs=image_wcs)
#         except Exception as e:
#             print(f"    Error creating cutout for galaxy {galaxy_id}: {e}")
#             continue
        
#         # Save science cutout
#         science_hdu = fits.PrimaryHDU(science_cutout.data, header=science_cutout.wcs.to_header())
#         science_hdu.writeto(science_path, overwrite=True)
        
#         # Save error cutout
#         error_hdu = fits.PrimaryHDU(error_cutout.data, header=error_cutout.wcs.to_header())
#         error_hdu.writeto(error_path, overwrite=True)
        
#         if i < 5 or i % 100 == 0:
#             print(f" done")
        
#         output_files.append(science_path)
#         successful_cutouts += 1
    
#     print("\n Some data to keep track of things: ")
#     print(f" Band: {band}")
#     print(f" Total galaxies in catalog: {len(ra_list)}")
#     print(f" Successful cutouts: {successful_cutouts}")
#     print(f" Skipped (already existed): {skipped_existing}")
#     print(f" Skipped (outside image): {skipped_outside}")
#     print(f" Output directory: {output_dir}")
    
#     return output_files, successful_cutouts


if __name__ == "__main__":
    # Set up for testing the code:
    # BASE_MOUNT = "/Volumes/Research/emcgrath/Research/CANDELS_data/mosaics/gds/"
    BASE_MOUNT = "/Research/emcgrath/Research/CANDELS_data/mosaics/gds/" # This is how to run it for NSCC(Natural Science Computing Cluster)

    
    MATCHED_CATALOG = "matched_catalog.fits"
    
    RED_FILTER = "f160w"
    GREEN_FILTER = "f125w"
    BLUE_FILTER = "f814w"
    
    RED_IMAGE = BASE_MOUNT + "goodss_all_wfc3_ir_f160w_060mas_v1.0_drz.fits"
    GREEN_IMAGE = BASE_MOUNT + "goodss_all_wfc3_ir_f125w_060mas_v1.0_drz.fits"
    BLUE_IMAGE = BASE_MOUNT + "goodss_all_acs_wfc_f814w_060mas_v1.5_drz.fits"
    
    image_paths = {
        'red': RED_IMAGE,
        'green': GREEN_IMAGE,
        'blue': BLUE_IMAGE
    }

    bands = {
        'red': RED_FILTER,
        'green': GREEN_FILTER,
        'blue': BLUE_FILTER
    }

    output_dirs = {
        'red': "cutouts_red",
        'green': "cutouts_green",
        'blue': "cutouts_blue"
    }
    
    for name, path in image_paths.items():
        if os.path.exists(path):
            print(f"{name.upper()}: {path} exists. File path correctly written.")
        else:
            print(f"{name.upper()} NOT FOUND at: {path}")
    
    catalog = Table.read(MATCHED_CATALOG)
    test_sample = catalog[:5]
    
    results = make_filter_cutouts(
        catalog=test_sample,
        image_paths=image_paths,
        bands=bands,
        output_dirs=output_dirs,
        box_radius=103.5,
        rootname="candels",
        overwrite=True,
        verbose=True
    )
    