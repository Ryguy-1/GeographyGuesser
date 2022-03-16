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
from ARCHIVE.image_with_text_to_country import text_country_identification


###############################
# Number Suggestions m
num_suggestions = 5
uses_mean_coords = True
key_activate = 'p'
# None if world, otherwise put in country code to limit search
just_one_country = None
###############################

classification_model_path = "models/country_classifier_model/country_classifier_250_250.h5"
regression_folder_path = "models/country_regression_models"
images_to_analyze_folder = "live_images"
resize_size = (250, 250)

# Countries in Geoguesser
geoguesser_countries = json.load(open('data/countries_in_geoguesser.json')).keys()

def get_likely_countries(model):
    
    # Specific Locations
    specific_locations_general = []
    top_countries_general = {}

    # =========================NN CLASSIFICATION =========================
    # Country Folders
    country_folders = glob.glob('data/images_sorted_by_country/*')
    # Load Image Dataset
    dataset = image_dataset_from_directory(images_to_analyze_folder,
                                           label_mode=None,
                                           image_size=resize_size,
                                           batch_size=len(glob.glob(images_to_analyze_folder + "/*")),
                                           shuffle = False,
                                           seed=42,
                                           color_mode="rgb")
    # Initialize Top Countries
    top_countries_nn = {}
    # Test Each Image
    for tensor in dataset.take(1):
        tensor_list = tensor.numpy()
        for tensor in tensor_list:
            # Run Through Model
            prediction = model.predict(tf.expand_dims(tensor, axis=0))
            # Print Country Prediction
            for i in range(len(prediction[0])):
                likely_index = np.argsort(prediction[0])[-(i+1)]
                country_name = country_folders[likely_index].split('\\')[-1]
                confidence = prediction[0][likely_index]
                # Country Name, Confidence
                if country_name in top_countries_nn.keys():
                    top_countries_nn[country_name] += confidence
                else:
                    top_countries_nn[country_name] = confidence

    # =========================TEXT CLASSIFICATION =========================
    top_countries_text = {}
    for image_loc in glob.glob(images_to_analyze_folder + "/*"):
        # Get Countries from Text With Score
        countries, score, location = text_country_identification(image_loc)
        # If No Text, Continue
        if countries is None and score is None and location is None:
            continue
        # If Text, Add to Countries
        for country in countries:
            top_countries_text[country] = score
        # Add Specific Locations
        specific_locations_general.append(location)

    # =========================Combination=========================
    # Only Keep Countries whose languages are the same
    for country, score in top_countries_text.items():
        if country in top_countries_nn:
            top_countries_general[country] = top_countries_nn[country]
    
    # Sort By Confidence Levels
    top_countries = sorted(top_countries_general.items(), key=lambda x: x[1], reverse=True)[:num_suggestions]
    # Return Top Countries
    return top_countries, specific_locations_general

def get_likely_coordinates(top_countries):
    # Get Just Countries
    countries = []
    for country, confidence in top_countries:
        countries.append(country)
    # Load Regression Folders
    regression_folders = glob.glob(regression_folder_path + "/*")
    # Find Folders Matching Countries
    matching_folders = []
    for country in countries:
        for folder in regression_folders:
            if country in folder:
                matching_folders.append(folder)
    # Get Coords for Image
    def test_individual_images(model, image_location, country_folder):
        def load_image(image_loc):
            # Load Images to Memory
            loaded_image = [cv2.resize(cv2.imread(image_loc), resize_size)]
            # Convert to Numpy Arrays
            loaded_image = np.array(loaded_image, dtype=np.float32)
            # Normalize Images
            loaded_image = loaded_image/255.0
            # Don't Normalize Labels
            return loaded_image
            
        # Load Image
        image = load_image(image_location)
        # Run Model
        prediction = model.predict(image)
        # Unstandardize Prediction
        lat_predicted = load_scalar(country_folder + "/scalar_lat.p").inverse_transform(prediction[0][0].reshape(1, -1))

        long_predicted = load_scalar(country_folder + "/scalar_long.p").inverse_transform(prediction[0][1].reshape(1, -1))

        # Return
        return np.array([lat_predicted[0][0]-90, long_predicted[0][0]-180], dtype=np.float32)
    # Predict Each Image Coordinates Assuming it's in each Country, then average them together
    country_coordinates = {}
    for country_folder in matching_folders:
        # Check if country folder has no_images.txt
        if os.path.isfile(country_folder + "/no_images.txt"):
            # Read No Images File
            with open(country_folder + "/no_images.txt", 'r') as f:
                no_images = f.read() # Ex: 42.6, 1.8
                # Split into Lat and Long
                lat_long_arr = no_images.split(', ')
                lat = float(lat_long_arr[0])
                long = float(lat_long_arr[1])
                # Add to Dictionary
                country_coordinates[country_folder.split('\\')[-1]] = [lat, long]
                continue
        else:   
            model = keras.models.load_model(country_folder + "/regression_250_250.h5", custom_objects={'custom_loss': custom_loss})
        lats = []; longs = []
        for image_location in glob.glob(images_to_analyze_folder + "/*"):
            coord = test_individual_images(model, image_location, country_folder)
            lats.append(coord[0])
            longs.append(coord[1])

        if uses_mean_coords:
            country_coordinates[country_folder.split('\\')[-1]] = [np.mean(lats), np.mean(longs)]
        else:
            country_coordinates[country_folder.split('\\')[-1]] = [[lat, long] for lat, long in zip(lats, longs)]
    return country_coordinates
            
def custom_loss(y_actual, y_pred):
    lat_pred = y_pred[:, 0]
    lon_pred = y_pred[:, 1]
    lat_actual = y_actual[:, 0]
    lon_actual = y_actual[:, 1]
    # Calculate sqrt((lat_pred - lat_actual)^2 + (lon_pred - lon_actual)^2)
    distance_lat_long = tf.sqrt(tf.square(lat_pred - lat_actual) + tf.square(lon_pred - lon_actual))
    # Convert Lat and Lon to Distance in kilometers
    distance_lat_long = distance_lat_long * 111.12
    # Implement Geoguessr Scoring Algorithm (y=4999.91(0.998036)^x) (ish)
    loss = tf.constant(5000, dtype=tf.float32) - tf.constant(5000, dtype=tf.float32) * tf.pow(tf.constant(0.999, dtype=tf.float32), distance_lat_long)
    # Reduce Mean of Losses
    loss = tf.reduce_mean(loss)
    # Return Loss
    return loss

def load_scalar(scalar_path):
    with open(scalar_path, 'rb') as f:
        return pickle.load(f)

# Not Using Yet
def get_screen(x_res, y_res, mss):
    monitor = {'top': int(y_res*0.2), 'left': int(x_res*0.2), 'width': int(x_res*0.6), 'height': int(y_res*0.6)}
    frame = np.array(mss.grab(monitor))
    return frame

def run_live():

    # Load Model
    model = keras.models.load_model(classification_model_path)

    # Images Currently Held
    image_counter = 0
    while True:
        if keyboard.is_pressed('p'):
            if just_one_country is None:
                top_countries, specific_locations = get_likely_countries(model)
                coord_suggestions = get_likely_coordinates(top_countries)
                print(top_countries)
                print(coord_suggestions)
                print(specific_locations)
            else:
                coord_suggestions = get_likely_coordinates([(just_one_country, 1)])        
                print(coord_suggestions)

        # Save New Image Screenshot
        elif keyboard.is_pressed('s'):
            # Get Screen
            frame = get_screen(2560, 1440, mss())
            # Save Frame
            cv2.imwrite(f"live_images/image{image_counter}.jpg", frame)
            # Increment Counter
            image_counter += 1
        
        # Delete All Current Images
        elif keyboard.is_pressed('d'):
            # Get Image locations
            image_locations = glob.glob("live_images/*")
            # Remove All Images
            [os.remove(location) for location in image_locations]

        time.sleep(0.05)


if __name__ == "__main__":                            
    run_live()