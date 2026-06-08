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

#  Want a different catalogue from the egs_merged because galaxy_zoo doesn't have that
#  UDS, cosmos, and good south work as far as galaxy_zoo catalogue are concerned.

unicorn_cat = "/Users/holyphysics/Desktop/Galaxy_Classification/egs_merged_v1.1.fits" # Change this path to one of the galaxy_zoo catalogues
uds = Table.read(unicorn_cat)
# lrd_ppg = "/Users/emcgrath/data/jwst/LRDs/stack_lrd_prism-clear.v3_homog_mast.20260116.all.lrds.txt"
# lrds = Table.read(lrd_ppg, format="ascii")



















# # Explore your CANDELS data
# from astropy.io import fits
# import numpy as np

# # Open one of your FITS files
# fits_file = "path/to/your/candels_f160w.fits" 
# hdu = fits.open(fits_file)

# # Print information about the FITS file
# hdu.info()  # Shows all HDUs in the file

# # Look at the header of the science image
# print(hdu['SCI'].header)  # Shows WCS, filter, exposure time, etc.

# # Check the shape of the image
# print(f"Image shape: {hdu['SCI'].data.shape}")

# # Check the data type and range
# print(f"Data type: {hdu['SCI'].data.dtype}")
# print(f"Min value: {np.min(hdu['SCI'].data)}")
# print(f"Max value: {np.max(hdu['SCI'].data)}")

# hdu.close()