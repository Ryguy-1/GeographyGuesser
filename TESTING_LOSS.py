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



# Input Two Tensors of the same shape
def custom_loss(y_actual, y_pred):
    lat_pred = y_pred[:, 0]
    lon_pred = y_pred[:, 1]
    lat_actual = y_actual[:, 0]
    lon_actual = y_actual[:, 1]
    # Calculate sqrt((lat_pred - lat_actual)^2 + (lon_pred - lon_actual)^2)
    distance_lat_long = tf.sqrt(tf.square(lat_pred - lat_actual) + tf.square(lon_pred - lon_actual))
    # Convert Lat and Lon to Distance in kilometers
    distance_lat_long = distance_lat_long * 111.12
    # Implement Geoguessr Scoring Algorithm (y=4999.91(0.998036)^x)
    loss = tf.constant(5000, dtype=tf.float32) - tf.constant(5000, dtype=tf.float32) * tf.pow(tf.constant(0.9990, dtype=tf.float32), distance_lat_long)
    # Reduce Mean of Losses
    loss = tf.reduce_mean(loss)
    # Return Loss
    return loss

# Make Tensor From Numpy Array
def make_tensor(x):
    return tf.convert_to_tensor(x, dtype=tf.float32)

y_actual = make_tensor(np.array([[0, 0], [10, 0], [20, 0], [80, 0]], dtype=np.float32))
y_pred = make_tensor(np.array([[0, 0], [0, -10], [0, -20], [0, -100]], dtype=np.float32))

print(custom_loss(y_actual, y_pred))