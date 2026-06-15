"""
Author: Chidiebere N. Okafor

Module for creating thumbnail cutouts from astronomical FITS images.

This module takes a catalog of galaxy positions (RA, Dec) and creates
small cutout images around each galaxy from a given filter image.

Purpose:
    - Extract small FITS cutouts centered on each galaxy
    - Save cutouts as individual FITS files for later RGB combination
    - Handle error arrays and WCS information properly
"""

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.units as u
import os
from typing import Tuple, List



def make_filter_cutouts(catalog, image_path, band, output_dir, box_radius=103.5, 
                        rootname="candels", prefix="", suffix="") -> Tuple[ List[str] , int]: # I'm probably making some mistake in the type annotation but I'll digress...
    """
    Create thumbnail cutouts for all galaxies in a catalog from a single filter image.
    
    This function extracts small FITS cutouts centered on each galaxy's position
    and saves them as individual files. It handles both science and error arrays.
    
    Parameters
    ----------
    catalog : astropy.table.Table
        Catalog containing galaxies with columns for RA, Dec, and ID
        Must have columns: 'RA', 'DEC', and 'ID' (or 'NUMBER')
    image_path : str
        Path to the FITS file containing the science image for this filter
        Example: "/data/candels_f160w_sci.fits"
    band : str
        Filter name (for example, 'f160w', 'f125w', 'f606w')
        This will appear in output filenames
    output_dir : str
        Directory where cutout FITS files will be saved
        Example: "cutouts_red"
    box_radius : int, optional
        Radius of cutout in pixels (final size will be 2 * box_radius)
        Default is 300 pixels( THat is 600x600 pixel cutouts)
    rootname : str, optional
        Root name for output files (for example, "candels", "egs", "uds")
        Default is "candels"
    prefix : str, optional
        Optional prefix for output filenames (for example, "lrd_")
        Default is "" (no prefix)
    suffix : str, optional
        Optional suffix for output filenames (for example, "v1")
        Default is "" (no suffix)
    
    Returns:
    -------
    output_files : list
        List of paths to created cutout FITS files
    cutout_count : int
        Number of successful cutouts created
    
    Notes:
    -----
    Output filename format: rootname.prefixCATID.band.suffix.fits
    Example: candels.12345.f160w.fits
    
    Error files are saved with '.err' before .fits:
    Example: candels.12345.f160w.err.fits
    """
    
    # Create output directory
    print(f" Making cutouts for band: {band}")
    
    # Create output directory if it doesn't exist
    if os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' already exists")
    else:
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Next we need to determine ID column name
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
    
    print(f"Using '{id_col}' as galaxy ID column")
    
    # Now, extract Coordinates and IDS from catalog 
    # Get RA, Dec, and IDs (handling different column name variations)
    ra_col = 'RA' if 'RA' in catalog.colnames else 'ra'
    dec_col = 'DEC' if 'DEC' in catalog.colnames else 'dec'
    
    # Also check for 'Dec' (capital D, rest lowercase). Again, necessary since I've seen this 
    # very varition used and had to debug an error descended from it.
    if dec_col not in catalog.colnames:
        dec_col = 'Dec' if 'Dec' in catalog.colnames else 'DEC'
    
    ra_list = catalog[ra_col]
    dec_list = catalog[dec_col]
    id_list = catalog[id_col]
    
    print(f" Using RA column: '{ra_col}', Dec column: '{dec_col}'")
    print(f" Processing {len(ra_list)} galaxies...")
    
   
    # Open the fits file and extract data
    print(f"\n Opening FITS image: {image_path}")
    
    with fits.open(image_path) as hdu:
        # Figured the previous code was specific to Candels codes with 
        # fits file in "SCI". However, when testing with GOODS_south data where
        # that was not the case, the previous code threw errors. Searched and found the fits files were in 
        # something called "PRIMARY HDU (extension 0)", not 'SCI'
        # So it's much better to use hdu[0] directly

        if 'SCI' in hdu:
            science_data = hdu['SCI'].data
            image_wcs = WCS(hdu[0].header, hdu)
            print("Format: CANDELS (SCI extension)")
        elif 'SCI' in hdu[1].name if len(hdu) > 1 else False:
            science_data = hdu[1].data
            image_wcs = WCS(hdu[1].header)
            print("Format: JWST (extension 1)")
        else:
            science_data = hdu[0].data
            image_wcs = WCS(hdu[0].header)
            print("Format: Primary HDU")
            
        
        error_data = hdu['ERR'].data if 'ERR' in hdu else np.zeros_like(science_data)
        
        ny, nx = science_data.shape
        print(f"Image dimensions: {nx} x {ny} pixels")
    
    # Create cutouts for each galaxy 
    output_files = []
    successful_cutouts = 0
    skipped_outside = 0
    box_size_pixels = box_radius * 2  # Convert radius to side length
    
    print(f"\n Creating cutouts (box size: {box_size_pixels}x{box_size_pixels} pixels)...")
    
    for i in range(len(ra_list)):
        galaxy_id = id_list[i]
        ra = ra_list[i]
        dec = dec_list[i]
        
        # Print progress every 100 galaxies or for first few. Just as a tracker/a self-sufficient 
        # progress indicator of some sort.
        if i < 5 or i % 100 == 0:
            print(f" Processing galaxy {i+1}/{len(ra_list)}: ID={galaxy_id}")
        
        # Create SkyCoord object for the galaxy center
        center = SkyCoord(ra, dec, frame='icrs', unit='deg')
        
        # Convert to pixel coordinates to check if inside image
        try:
            x_pixel, y_pixel = center.to_pixel(image_wcs)
        except Exception as e:
            print(f"\n Warning: Could not convert coordinates for galaxy {galaxy_id}: {e}")
            skipped_outside += 1
            continue
        
        # Check if galaxy falls within image bounds(the mosiac)
        if (x_pixel < 0 or x_pixel >= nx or y_pixel < 0 or y_pixel >= ny):
            print(f" Skipping galaxy {galaxy_id}: outside image bounds")
            skipped_outside += 1
            continue

        # Is this code above doing exactly as this code below. Hmmmm... I'll need to return to this later, Chidiebere.
        # if((cutout_cenpix[0]+1 < nx) & (cutout_cenpix[0]+1 > 0) & (cutout_cenpix[1]+1 < ny) & (cutout_cenpix[1]+1 > 0)): 
        #     box[i]*=2
        #     if(box[i] > 666): 
        #         box[i]=666
        #     if(box[i] < 200):
        #         box[i]=200
        # 
        
        # Create the cutouts
        try:
            science_cutout = Cutout2D(science_data, center, (box_size_pixels, box_size_pixels), wcs=image_wcs)
            
            error_cutout = Cutout2D(error_data, center, (box_size_pixels, box_size_pixels), wcs=image_wcs)
        except Exception as e:
            print(f"    Error creating cutout for galaxy {galaxy_id}: {e}")
            continue
        
        # Construct output filename
        if suffix:
            science_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.{suffix}.fits"
            error_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.{suffix}.err.fits"
        else:
            science_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.fits"
            error_filename = f"{rootname}.{prefix}{galaxy_id}.{band}.err.fits"
        
        science_path = os.path.join(output_dir, science_filename) # I can also proceed with Prof. McGrath's concatenation style. 
        error_path = os.path.join(output_dir, error_filename) #  I can also proceed with Prof. McGrath's concatenation style. 
        
        # Save science cutout
        science_hdu = fits.PrimaryHDU(science_cutout.data, header=science_cutout.wcs.to_header())
        science_hdu.writeto(science_path, overwrite=True)
        
        # Save error cutout
        error_hdu = fits.PrimaryHDU(error_cutout.data, header=error_cutout.wcs.to_header())
        error_hdu.writeto(error_path, overwrite=True)
        
        output_files.append(science_path)
        successful_cutouts += 1
    
  

    print("\n Some data to keep track of things: ")
    print(f" Band: {band}")
    print(f" Total galaxies in catalog: {len(ra_list)} == {len(dec_list)} == {len(id_list)}")
    print(f" Successful cutouts: {successful_cutouts}")
    print(f" Skipped (outside image): {skipped_outside}")
    print(f" Output directory: {output_dir}")
    
    return output_files, successful_cutouts




if __name__ == "__main__":
    # Set up for testing the code:
    # First connect to colby smb serve and login so that the drive mounts to /Volumes/Research/

    BASE_MOUNT = "/Volumes/Research/emcgrath/Research/CANDELS_data/mosaics/gds/"
    
    MATCHED_CATALOG = "matched_catalog.fits"
    
    # Filter information - TODO: Ask Prof. McGrath which filters to use
    RED_FILTER = "f160w"
    GREEN_FILTER = "f125w"
    BLUE_FILTER = "f814w"
    
    # Paths to your FITS images
    RED_IMAGE = BASE_MOUNT + "goodss_all_wfc3_ir_f160w_060mas_v1.0_drz.fits"
    GREEN_IMAGE = BASE_MOUNT + "goodss_all_wfc3_ir_f125w_060mas_v1.0_drz.fits"
    BLUE_IMAGE = BASE_MOUNT + "goodss_all_acs_wfc_f814w_060mas_v1.5_drz.fits"

    # Check if files exist before running
    for img_path, name in [(RED_IMAGE, "RED"), (GREEN_IMAGE, "GREEN"), (BLUE_IMAGE, "BLUE")]:
        if os.path.exists(img_path):
            print(f" {name}: {img_path} exists. File path correctly written. ")
        else:
            print(f" {name} NOT FOUND at: {img_path}")
    

    
    
    # Load the matched catalog
    catalog = Table.read(MATCHED_CATALOG)
    
    # Make cutouts for each filter
    # make_filter_cutouts(
    #     catalog=catalog,
    #     image_path=RED_IMAGE,
    #     band=RED_FILTER,
    #     output_dir="cutouts_red",
    #     box_radius=300
    # )
    
    # make_filter_cutouts( # That was a lot of .fits file. Jeez
    #     catalog=catalog,
    #     image_path=GREEN_IMAGE,
    #     band=GREEN_FILTER,
    #     output_dir="cutouts_green",
    #     box_radius=300
    # )
    
    # make_filter_cutouts(
    #     catalog=catalog,
    #     image_path=BLUE_IMAGE,
    #     band=BLUE_FILTER,
    #     output_dir="cutouts_blue",
    #     box_radius=300
    # )
    
    print("\n All cutouts complete!")
    # I'll need to transfer the running of this file over to a google colab script