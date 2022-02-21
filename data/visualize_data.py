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


# Image Folder
image_folder = "data/images"

# Load Images
image_locations = glob.glob(image_folder + "/*")

# Image Latitudes
lats = []
longs = []
for image_location in image_locations:
    lats.append(float(image_location.split("\\")[-1].split(".p")[0].split("_")[0]))
    longs.append(float(image_location.split("\\")[-1].split(".p")[0].split("_")[-1]))

# Graph Density of Latitudes and Longitudes
import matplotlib.pyplot as plt
plt.scatter(longs, lats)
# Label 
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Density of Latitudes and Longitudes")
plt.show()
