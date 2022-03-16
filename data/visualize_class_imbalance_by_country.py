import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Image Folder
image_folder = 'data/images_sorted_by_country'

# Get Num Images Per Country
country_images_map = {}
for folder in glob.glob(image_folder + "/*"):
    folder_name = folder.split('\\')[-1]
    num_items = len(glob.glob(folder + "/*"))
    country_images_map[folder_name] = num_items

# Create Bar Graph Matplotlib

bar = plt.bar(np.arange(len(country_images_map)), country_images_map.values())
plt.bar_label(bar, country_images_map.keys())
plt.show()