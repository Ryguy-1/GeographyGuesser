# Tensorflow 2.7.0
from re import T
from matplotlib import units
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
# Time
import time
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
# Keyboard
import keyboard


classification_model_path = "/models/country_classifier_model/country_classifier_250_250.h5"
regression_folder_path = "models/country_regression_models"

def analyze():
    # in progress....
    pass
def run_live():
    while True:
        if keyboard.is_pressed('Space'):
            analyze()                                                                                                                                                                                       
        time.sleep(0.01)


if __name__ == "__main__":                            
    run_live()