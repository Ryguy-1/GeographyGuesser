import os
import glob
from socketserver import ThreadingTCPServer
from geopy.geocoders import Nominatim
import shutil

# Globals
raw_image_folder = "data/images"
sorted_images_folder = "data/images_sorted_by_country"

# initialize Nominatim API
geolocator = Nominatim(user_agent="geoapiExercises")

# Image Locations
image_locations = glob.glob(raw_image_folder + "/*")

# Iterate Through Images and Sort by Country
transfer_counter = 0
for file_loc in image_locations:
    # Get Latitude and Longitude of png
    lat = str(float(file_loc.split("\\")[-1].split(".p")[0].split("_")[0]))
    long = str(float(file_loc.split("\\")[-1].split(".p")[0].split("_")[-1]))
    location = geolocator.reverse(lat+","+long)
    # Get Country
    print(location.raw['address']['country'] + "|")
    country = str(location.raw['address']['country']).lower().strip().replace(" ", "_").replace("/", "-")
    if not os.path.exists(sorted_images_folder + "/" + country):
        os.makedirs(sorted_images_folder + "/" + country)
    # Make Copy of Image in New Folder
    shutil.copy(file_loc, sorted_images_folder + "/" + country)
    transfer_counter += 1
    if transfer_counter % 100 == 0:
        print("Transfered: " + str(transfer_counter) + " Images")


