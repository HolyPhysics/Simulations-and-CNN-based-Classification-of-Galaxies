import numpy as np
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
from astroML.datasets import fetch_sdss_sspp 
## Convert from numpy to pytorch tensors at the end of the function to be defined below.

try:
    galaxy_file_path = "gz_candels_table_2_main_release.csv" # Alternatively, I can use "galaxyquasar.csv" and make the code tailored to it so it can actually work!
    galaxy_container = Table.read(galaxy_file_path)
    print(galaxy_container.columns)
    
except Exception as error_1:
    print(f" { type(error_1).__name__ } occured! ")


# Since there are 105783 stars and 327260 quasars, to avoid class imbalance, we'll select equal number of samples from the stars and the quasars. This number will be kept below 105783 because of the stars.


# def processed_data(sample_size_per_class = 6000): # change the batch size here

#     star_u_color = star_container["upsf"][: sample_size_per_class]
#     star_g_color = star_container["gpsf"][: sample_size_per_class]
#     star_r_color = star_container["rpsf"][: sample_size_per_class]
#     star_i_color = star_container["ipsf"][: sample_size_per_class]
#     star_z_color = star_container["zpsf"][: sample_size_per_class]
#     # print(star_z_color)

#     star_data = Table({"u_g_magnitude": star_u_color - star_g_color,
#                     "g_r_magnitude": star_g_color - star_r_color,
#                     "r_i_magnitude": star_r_color - star_i_color,
#                     "i_z_magnitude": star_i_color - star_z_color})

#     star_data["class"] = "star"

#     # print(star_data)
#     quasar_u_color = quasar_container["UMAG"][: sample_size_per_class]
#     quasar_g_color = quasar_container["GMAG"][: sample_size_per_class]
#     quasar_r_color = quasar_container["RMAG"][: sample_size_per_class]
#     quasar_i_color = quasar_container["IMAG"][: sample_size_per_class]
#     quasar_z_color = quasar_container["ZMAG"][: sample_size_per_class]
#     # print(star_z_color)

#     quasar_data = Table({"u_g_magnitude": quasar_u_color - quasar_g_color,
#                     "g_r_magnitude": quasar_g_color - quasar_r_color,
#                     "r_i_magnitude": quasar_r_color - quasar_i_color,
#                     "i_z_magnitude": quasar_i_color - quasar_z_color})

#     quasar_data["class"] = "quasar"

#     # print(quasar_data)
#     combined_data = vstack([star_data, quasar_data])
#     # print(combined_data)

#     # now, I want to shuffle the data so the models don't learn the order in which stars and quasars appear in the data
#     np.random.seed(41)
#     permuted_indices = np.random.permutation(2*sample_size_per_class)
#     # print(permuted_indices)

#     u_g_magnitude = combined_data["u_g_magnitude"][permuted_indices]
#     g_r_magnitude = combined_data["g_r_magnitude"][permuted_indices]
#     r_i_magnitude = combined_data["r_i_magnitude"][permuted_indices]
#     i_z_magnitude = combined_data["i_z_magnitude"][permuted_indices]
#     class_label = combined_data["class"][permuted_indices]

#     X = np.vstack([u_g_magnitude, g_r_magnitude, r_i_magnitude, i_z_magnitude]).T
#     # class_value = combined_data["class"][permuted_indices]
#     y = np.array( class_label == "quasar", dtype = int)
#     # print(X)
#     # print(y)

#     return X, y, star_data, quasar_data


# if __name__ == "__main__":
#     processed_data()