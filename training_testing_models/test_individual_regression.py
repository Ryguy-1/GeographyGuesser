# Tensorflow 2.7.0
from matplotlib import units
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential, optimizers, losses
import tensorflow.keras as keras
# Tensorboard
import tensorboard as tb
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
# Datetime
import datetime
# Glob
import glob
# Numpy
import numpy as np
# OpenCV
import cv2
# Scikit-Learn traintestsplit
from sklearn.model_selection import train_test_split
# Scikit-Learn MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
# Pickle
import pickle

model_folder = "models/country_regression_models/US"
resize_size = (250, 250)
real_image_folder = "testing_images"

def load_scalar(scalar_name):
    with open(model_folder + "/" + scalar_name, 'rb') as f:
        return pickle.load(f)

def test_individual_images(model, image_index_in_image_archive, dataset_directory):
    def load_image(image_index_in_image_archive):
        image_locations = glob.glob(dataset_directory + "/*")
        image_loc = None
        try:
            image_loc = image_locations[image_index_in_image_archive]
        except:
            print("No image found at index: " + str(image_index_in_image_archive))
        del image_locations
        # Load Images to Memory
        loaded_image = [cv2.resize(cv2.imread(image_loc), resize_size)]
        try:
            # For png files
            label = (float(image_loc.split("\\")[-1].split(".p")[0].split("_")[0]), float(image_loc.split("\\")[-1].split(".p")[0].split("_")[-1]))
        except:
            # For jpg files
            label = (float(image_loc.split("\\")[-1].split(".j")[0].split("_")[0]), float(image_loc.split("\\")[-1].split(".j")[0].split("_")[-1]))
        # Convert to Numpy Arrays
        loaded_image = np.array(loaded_image, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        # Normalize Images
        loaded_image = loaded_image/255.0
        # Don't Normalize Labels
        return loaded_image, label
        
    # Load Image
    image, label_unstandardized = load_image(image_index_in_image_archive)
    # Predict
    print(f"Image Shape: {image.shape}")
    prediction = model.predict(image)
    # Unstandardize Prediction
    lat_predicted = load_scalar("scalar_lat.p").inverse_transform(prediction[0][0].reshape(1, -1))

    long_predicted = load_scalar("scalar_long.p").inverse_transform(prediction[0][1].reshape(1, -1))

    # Return
    return np.array([lat_predicted[0][0]-90, long_predicted[0][0]-180], dtype=np.float32), label_unstandardized, image, prediction

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
    loss = tf.constant(5000, dtype=tf.float32) - tf.constant(5000, dtype=tf.float32) * tf.pow(tf.constant(0.993, dtype=tf.float32), distance_lat_long)
    # Reduce Mean of Losses
    loss = tf.reduce_mean(loss)
    # Return Loss
    return loss

if __name__ == "__main__":
    # Load Model
    model = keras.models.load_model(model_folder + "/regression_250_250.h5", custom_objects={'custom_loss': custom_loss})
    for i in range(len(glob.glob(real_image_folder + "/*"))):
        # Test Individual Image
        predicted, label_unstandardized, image, prediction_raw = test_individual_images(model, i, real_image_folder)

        # Print Information
        print("Predicted: " + str(predicted))
        print("Label: " + str(label_unstandardized))
        print("Prediction Raw: " + str(prediction_raw))
        # Show Image
        cv2.imshow("Image", image[0])
        cv2.waitKey(0)
