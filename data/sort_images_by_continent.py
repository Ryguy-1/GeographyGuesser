import os
import glob
import reverse_geocoder
import shutil
import json

# Globals
raw_image_folder = "data/images"
sorted_images_folder = "data/images_sorted_by_continent"

# Image Locations
image_locations = glob.glob(raw_image_folder + "/*")

# Countries in Geoguesser
country_to_continent_map = json.load(open('data/countries_to_continents.json'))

# Iterate Through Images and Sort by Country
transfer_counter = 0
for file_loc in image_locations:
    try:
        # Get Latitude and Longitude of png
        lat = float(file_loc.split("\\")[-1].split(".p")[0].split("_")[0])
        long = float(file_loc.split("\\")[-1].split(".p")[0].split("_")[-1])
        # Get Country of Image
        location = reverse_geocoder.search((lat, long), mode=1)
        country = location[0]['cc']
        # Get Continent
        continent = country_to_continent_map[country]
        # Get Corrosponding continent
        if not os.path.exists(sorted_images_folder + "/" + continent):
            os.makedirs(sorted_images_folder + "/" + continent)
        # Make Copy of Image in New Folder
        shutil.copy(file_loc, sorted_images_folder + "/" + continent)
        transfer_counter += 1
        if transfer_counter % 100 == 0:
            print("Transfered: " + str(transfer_counter) + " Images")
    except Exception:
        print("Error: " + file_loc)