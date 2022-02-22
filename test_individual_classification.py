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


model_folder = "models/country_classifier_model"
resize_size = (250, 250)
real_image_folder = "testing_images"


if __name__ == "__main__":
    # Load Model
    model = keras.models.load_model(model_folder + "/country_classifier_250_250.h5")
    # Country Folders
    country_folders = glob.glob('data/images_sorted_by_country/*')
    # Load Image Dataset
    dataset = image_dataset_from_directory(real_image_folder,
                                           label_mode=None,
                                           image_size=resize_size,
                                           batch_size=len(glob.glob(real_image_folder + "/*")),
                                           shuffle = False,
                                           seed=42,
                                           color_mode="rgb")
    # Get Tensors
    # Test Each Image
    for tensor in dataset.take(1):
        tensor_list = tensor.numpy()
        for tensor in tensor_list:
            # Print Tensor Shape
            print(tensor.shape)
            # Run Through Model
            prediction = model.predict(tf.expand_dims(tensor, axis=0))

            # Print Country Prediction
            max_index = np.argsort(prediction[0])[-1]
            country_name = country_folders[max_index].split('\\')[-1]
            print(f"Predicted Country: {country_name}")
            # Print Confidence
            print(f"Confidence: {prediction[0][max_index]}")

            # Second Most Confident
            second_max_index = np.argsort(prediction[0])[-2]
            country_name = country_folders[second_max_index].split('\\')[-1]
            print(f"Second Most Confident Predicted Country: {country_name}")
            # Print Confidence
            print(f"Confidence: {prediction[0][second_max_index]}")

            # Show Image
            cv2.imshow("Image", tensor.astype(np.uint8))
            cv2.waitKey(0)
        
