# Get Rid of Warnings and Info Tensorflow Messages
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# Tensorflow 2.7.0
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import image_dataset_from_directory
# Datetime
import datetime
# Time
import time
# Glob
import glob
# Numpy
import numpy as np
# OpenCV
import cv2
# Pickle
import pickle
# Keyboard
import keyboard
# OS
import os
# Json
import json
# Mss
from mss import mss
# Text Country 
from specified_classification.text_classification.image_with_text_to_country_new import geolocation_and_language_from_image_location
# Selenium
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Continent Classificaiton Model
classification_model_path = "models/continent_classifier_model/continent_classifier_250_250.h5"
# Live Images
images_to_analyze_folder = "live_images"
# Resizing Size for Image Folder
resize_size = (150, 150)

# Countries in Geoguesser
geoguesser_countries = json.load(open('data/countries_in_geoguesser.json')).keys()

# Country to Languages
country_to_lang_arr_map = json.load(open('data/country_to_languages.json'))


def run_live():
    # Load Model
    custom_classification_model = keras.models.load_model(classification_model_path)

    # Images Currently Held
    image_counter = 0
    while True:
        
        # Predict Image Folder
        if keyboard.is_pressed('p'):
            if len(glob.glob(images_to_analyze_folder + "/*")) == 0:
                print("Press 'S' To Save an Image First")
                time.sleep(1)
                continue
            print(); print()
            print("---------------------------- Predictions ----------------------------")
            # Predict Likely Countries, Along With Addresses
            top_countries, specific_locations = get_likely_countries(custom_classification_model)
            print()
            print("-- Top Countries --")
            print(top_countries)
            print()
            print("-- Specific Locations --")
            print(specific_locations)
            print("---------------------------------------------------------------------")

            display_specific_locations(specific_locations, 2)

        # Save New Image Screenshot
        elif keyboard.is_pressed('s'):
            # Get Screen
            frame = get_screen(2560, 1440, mss())
            # Prompt For Crop and Return Cropped Image
            cropped_frame = Cropper().prompt_crop(frame)
            # Save Frame
            cv2.imwrite(f"live_images/image{image_counter}.jpg", cropped_frame)
            # Increment Counter
            image_counter += 1
        
        # Delete All Current Images
        elif keyboard.is_pressed('d'):
            # Get Image locations
            image_locations = glob.glob("live_images/*")
            # Remove All Images
            [os.remove(location) for location in image_locations]
            image_counter = 0

        time.sleep(0.05)




def get_likely_countries(custom_classification_model):

    # Returns ->  1) Geolocated Locations (Ultra Specific), 2) Countries Identified (General Regions with Probabilities)
    geolocated_locations = []
    top_continents_general = {}

    # ----------------------- NN CLASSIFICATION -----------------------
    # Continent Folders
    continent_folders = glob.glob('data/images_sorted_by_continent/*')
    # Load Image Dataset
    dataset = image_dataset_from_directory(images_to_analyze_folder,
                                           label_mode=None,
                                           image_size=resize_size,
                                           batch_size=len(glob.glob(images_to_analyze_folder + "/*")),
                                           shuffle = False,
                                           seed=42,
                                           color_mode="rgb")
    # Initialize Top Countries
    top_continents_nn = {}
    # Test Each Image
    for tensor in dataset.take(1):
        tensor_list = tensor.numpy()
        for tensor in tensor_list:
            # Run Through Model
            prediction = custom_classification_model.predict(tf.expand_dims(tensor, axis=0))
            # Print Continent Prediction
            for i in range(len(prediction[0])):
                likely_index = np.argsort(prediction[0])[-(i+1)]
                continent_name = continent_folders[likely_index].split('\\')[-1]
                confidence = prediction[0][likely_index]
                # Continent Name, Confidence
                if continent_name in top_continents_nn.keys():
                    top_continents_nn[continent_name] += confidence
                else:
                    top_continents_nn[continent_name] = confidence


    # ----------------------- Text Classification -----------------------

    # Keep Track of Languages Identified By Text
    languages_identified_by_text = []
    # Iterate Through Images
    for image_loc in glob.glob(images_to_analyze_folder + "/*"):
        # Get Address and Language Predicted By Google
        addresses_matching_google_language, language = geolocation_and_language_from_image_location(image_loc)
        # Check if None
        if addresses_matching_google_language is None and language is None:
            continue
        # If Text, Add Addresses to Master List
        for location in addresses_matching_google_language:
            geolocated_locations.append(location)
        # Append to Languages Identified By Text
        languages_identified_by_text.append(language)
    
    # Convert Languages Found to List of Possible Countries
    countries_identified_by_text = []
    # Iterate Through Languages Identified
    for text_lang in languages_identified_by_text:
        # Iterate Through Country with Corrosponding Language
        for country, lang_arr in country_to_lang_arr_map.items():
            # Check if that Country Speaks the Language
            if text_lang in lang_arr and country in geoguesser_countries:
                # Append the Country if So
                countries_identified_by_text.append(country)

    # --------------------- Combine Variables --------------------

    top_continents_general = top_continents_nn

    # Return
    return top_continents_general, geolocated_locations

# ---------------------Helper Methods--------------------

# Displays Top Guesses Using Google Maps
def display_specific_locations(specific_locations, num_displayed_max):
    # Calculate how Many to Show
    shown = 0
    if num_displayed_max < len(specific_locations):
        shown = num_displayed_max
    else:
        shown = len(specific_locations)
    # Resolution
    res_x = 2560; res_y = 1440
    # Initialize Drivers
    drivers = [Chrome("C:\\Selenium\\chromedriver.exe") for i in range(shown)]
    # All Open Google Maps
    for i in range(len(drivers)):
        driver = drivers[i]
        driver.set_window_size(res_x, int(res_y/num_displayed_max))
        driver.set_window_position(0, int(res_y/num_displayed_max)*i)
    # All Search Individual Locations and Zoom Out
    for i in range(shown):
        driver = drivers[i]
        # Search
        driver.get("https://www.google.com/maps")
        time.sleep(3)
        # Get Search Bar
        search_bar_element = driver.find_element(By.ID, "searchboxinput")
        # Clear Search Bar
        search_bar_element.clear()
        # Click on Search Bar
        search_bar_element.click()
        # Send Long/Lat
        search_bar_element.send_keys(specific_locations[i].address)
        # Search
        search_bar_element.send_keys(Keys.ENTER)
        # Wait for Load
        time.sleep(3)
        # Find Zoom Out
        zoom_out = driver.find_element(By.ID, "widget-zoom-out")
        for i in range(8):
            zoom_out.click()
            time.sleep(0.08)

    # Wait to Look at Them
    time.sleep(12)

# Crops Images for Save with Text in Them Given User Defined Box
class Cropper:
    def __init__(self):
        # Keep Track of Positions
        self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0
        # Keep Track of Current Crop
        self.cropping_done = False

    def prompt_crop(self, frame):
        # Show Image
        cv2.imshow('Crop Text of Image if Present', frame)
        # What Is Called On Click
        def mouse_callback(event, x, y, flags, param):
            # Check if Press Down For Cropping
            if event == cv2.EVENT_LBUTTONDOWN:
                self.x_start, self.y_start = x, y
            # Make Sure Get Crop During Movements
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.cropping_done == False:
                    self.x_end, self.y_end = x, y
            # Stop Crop Lift Mouse Button
            elif event == cv2.EVENT_LBUTTONUP:
                self.x_end, self.y_end = x, y
                self.cropping_done = True

        # Mouse Callback
        cv2.setMouseCallback('Crop Text of Image if Present', mouse_callback)
        # New Frame
        while True:
            if self.cropping_done:
                # Print Starting and Ending Coords of Crop
                print(f"({self.x_start}, {self.y_start}) & ({self.x_end}, {self.y_end})")
                # Destroy Windows
                cv2.destroyAllWindows()
                # Return Cropped Frame
                return frame[self.y_start:self.y_end, self.x_start:self.x_end]
            cv2.waitKey(5)


# For Loading Regression Scalar
def load_scalar(scalar_path):
    with open(scalar_path, 'rb') as f:
        return pickle.load(f)

# For Capturing Screenshot of Geoguesser
def get_screen(x_res, y_res, mss):
    monitor = {'top': int(y_res*0.2), 'left': int(x_res*0.2), 'width': int(x_res*0.6), 'height': int(y_res*0.6)}
    frame = np.array(mss.grab(monitor))
    return frame

# Main Method
if __name__ == "__main__":
    run_live()