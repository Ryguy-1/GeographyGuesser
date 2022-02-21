# Tensorflow 2.7.0
from matplotlib import units
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential, optimizers, losses
import tensorflow.keras as keras
from tensorflow.keras.utils import image_dataset_from_directory
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
# Imblearn
from imblearn.over_sampling import SMOTE
# Shutil
import shutil
# OS
import os

# Sorted Images Folders
images_sorted_by_country_folder = "data/images_sorted_by_country"
balanced_sorted_by_country_folder = "data/balanced_images_sorted_by_country"
resize_size = (250, 250)

# Image Folders
image_folders = glob.glob(images_sorted_by_country_folder + "/*")

# Find Most Amount of Images in a Country (target for SMOTE oversampling)
max_images_per_country = 0
# Get How Many Files in Each Folder
for folder in image_folders:
    print(folder + ": " + str(len(glob.glob(folder + "/*"))))
    if len(glob.glob(folder + "/*")) > max_images_per_country:
        max_images_per_country = len(glob.glob(folder + "/*"))
print(max_images_per_country)


# Reduce Every Class to Same Size
for folder in image_folders:
    if not os.path.exists(balanced_sorted_by_country_folder + "/" + folder.split("\\")[-1]):
        os.makedirs(balanced_sorted_by_country_folder + "/" + folder.split("\\")[-1])
        print("Created New Image Folder: " + balanced_sorted_by_country_folder + "/" + folder.split("\\")[-1])
    # Get All Images in Folder
    images = glob.glob(folder + "/*")
    # Shuffle Images
    images = shuffle(images)
    # Load images
    loaded_images = []
    for image in images:
        loaded_images.append(cv2.resize(cv2.imread(image), resize_size))
    images = np.array(loaded_images, dtype=np.float32)
    # Get Labels
    labels = []
    for image in images:
        labels.append(folder.split("\\")[-1])
    # Oversample
    sm = SMOTE(random_state=42)
    images, labels = sm.fit_resample(images, labels)
    print(len(images))
    print(len(images))
    quit()
