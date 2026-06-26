"""
PSF Matching using Professor McGrath's Provided PSF Files. 
I had to, first, install the photutils package.
Come back and update the return types of each functions.
"""

from photutils.psf_matching import make_wiener_kernel, TukeyWindow
from scipy.signal import fftconvolve # from this site: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html
# The fftconvolve is much faster than traditional fourier convolution functions because it uses 
# the Fast Fourier Transform(fft) algorithm which aids its speed up of the processes involved. 
from astropy.io import fits
import numpy as np
import os
from typing import List, Tuple


def load_psf_from_file(psf_path) -> List[float]:
    """
    Load a PSF from a FITS file and normalize it.
    
    Parameters
    ----------
    psf_path : str
        Path to the PSF FITS file
    
    Returns
    -------
    psf : np.ndarray
        Normalized PSF array
    """
    if not os.path.exists(psf_path):
        raise FileNotFoundError(f"PSF file not found: {psf_path}")
    
    psf_data = fits.getdata(psf_path)
    
    # Remove/cleans any NaN or inf values
    psf_data = np.nan_to_num(psf_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize to sum to 1. A requirement from the psf documentation at: https://photutils.readthedocs.io/en/latest/user_guide/psf_matching.html#window-functions
    psf_data = psf_data / np.sum(psf_data)
    
    return psf_data


def match_psfs_with_provided_psfs(red_data, green_data, blue_data,
                                  psf_red_path, psf_green_path, psf_blue_path,
                                  regularization=1e-4,
                                  window_alpha=0.3) -> Tuple[ List[float], List[float], List[float] ]:
    """
    Match PSFs of three filter images using provided PSF FITS files. 

    Parameters
    ----------
    red_data : np.ndarray
        Red channel image data (F160W)
    green_data : np.ndarray
        Green channel image data (F125W)
    blue_data : np.ndarray
        Blue channel image data (F606W)
    psf_red_path : str
        Path to PSF file for red filter (F160W)
    psf_green_path : str
        Path to PSF file for green filter (F125W)
    psf_blue_path : str
        Path to PSF file for blue filter (F606W)
    regularization : float, optional
        Regularization parameter (default 1e-4)
    window_alpha : float, optional
        Window alpha parameter (default 0.3)
    
    Returns
    -------
    tuple : (matched_red, matched_green, matched_blue)
        PSF-matched images
    """
    
    print("\n Loading PSF files...")
    
    # This loads the PSFs
    psf_red = load_psf_from_file(psf_red_path)
    psf_green = load_psf_from_file(psf_green_path)
    psf_blue = load_psf_from_file(psf_blue_path)
    
    # print(f"  Red PSF shape: {psf_red.shape}, sum: {psf_red.sum():.3f}")
    # print(f"  Green PSF shape: {psf_green.shape}, sum: {psf_green.sum():.3f}")
    # print(f"  Blue PSF shape: {psf_blue.shape}, sum: {psf_blue.sum():.3f}")
    
    target_psf = psf_red
    # Create window
    window = TukeyWindow(alpha=window_alpha)
    
    # print("\n Computing matching kernels...")
    
    # Match each image to target
    def match_one(source_psf, image) -> List[str]: # I can delete the field for filter_name; Old version: match_one(source_psf, image, filter_name)
        # print(f"  Matching {filter_name}...")
        kernel = make_wiener_kernel(source_psf, target_psf,
                                   regularization=regularization,
                                   window=window)
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        # Apply convolution
        matched = fftconvolve(image, kernel, mode='same') # With mode="same", returns a matrix of size equal to the size of image, which is what we want.
        return matched
    
    matched_red = red_data # No convolution needed since we're matching the Psf's to the psf_red
    matched_green = match_one(psf_green, green_data)
    matched_blue = match_one(psf_blue, blue_data)
    
    # print("\n PSF matching complete!")
    
    return matched_red, matched_green, matched_blue
      