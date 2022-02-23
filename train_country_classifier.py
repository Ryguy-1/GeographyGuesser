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
from sklearn.metrics import ConfusionMatrixDisplay
# Scikit-Learn MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
# Pickle
import pickle
# Matplotlib
import matplotlib.pyplot as plt

# Globals
images_sorted_by_country_folder = "data/images_sorted_by_country"
country_classifier_model_folder = "models/country_classifier_model"
resize_size = (250, 250)
test_size = 0.2

# (250, 250) CountryClassifierModel
class CountryClassifierModel:

    def __init__(self, num_countries, input_shape=(250, 250, 3)):
        self.input_shape = input_shape
        self.num_countries = num_countries
        self.model = Sequential()

        self.model.add(layers.Conv2D(filters = 128, kernel_size = (7, 7), strides=(3, 3), data_format="channels_last", activation=None, input_shape=self.input_shape))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.7))
        assert self.model.output_shape == (None, 82, 82, 128)

        self.model.add(layers.Conv2D(filters = 128, kernel_size = (7, 7), strides=(3, 3), data_format="channels_last", activation=None))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.6))
        assert self.model.output_shape == (None, 26, 26, 128)

        self.model.add(layers.Conv2D(filters = 128, kernel_size = (5, 5), strides=(3, 3), data_format="channels_last", activation=None))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.5))
        assert self.model.output_shape == (None, 8, 8, 128)

        self.model.add(layers.Flatten())
        assert self.model.output_shape == (None, 8192)

        self.model.add(layers.Dense(units=4096, activation=None))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.4))
        assert self.model.output_shape == (None, 4096)

        self.model.add(layers.Dense(units=1024, activation=None))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.4))
        assert self.model.output_shape == (None, 1024)

        self.model.add(layers.Dense(units=256, activation=None))
        self.model.add(layers.Activation("relu"))
        assert self.model.output_shape == (None, 256)

        self.model.add(layers.Dense(units=self.num_countries, activation='softmax'))
        assert self.model.output_shape == (None, num_countries)

        self.optimizer = optimizers.Adam(learning_rate=0.00001)
        self.loss_function = losses.CategoricalCrossentropy()

        self.model.compile(
            optimizer = self.optimizer,
            loss = self.loss_function,
            metrics = ['accuracy']
        )

    def summary(self, ):
        self.model.summary()

    def save_model(self, model_name):
        self.model.save(model_name)
    
    def load_model(self, model_name):
        self.model = models.load_model(model_name)

def get_class_weights(image_folder):
    # Get Folders
    folders = glob.glob(image_folder + "/*")
    # Total number of images
    total_images = 0
    for folder in folders:
        total_images += len(glob.glob(folder + "/*"))
    # Get Total Classes
    total_classes = len(folders)
    print(folders)
    class_weights = {}
    for i in range(len(folders)):
        # wj=n_samples / (n_classes * n_samplesj)
        class_weights[i] = total_images / (total_classes * len(glob.glob(folders[i] + "/*")))
    return class_weights

def train():
    # New Model
    cnn = CountryClassifierModel(num_countries=len(glob.glob(images_sorted_by_country_folder + "/*")))

    # Load Dataset
    dataset_train = image_dataset_from_directory(images_sorted_by_country_folder,
                                           label_mode="categorical",
                                           image_size=resize_size,
                                           batch_size=32,
                                           seed=42,
                                           shuffle=True,
                                           validation_split=test_size,
                                           subset="training",
                                           color_mode="rgb")

    dataset_test = image_dataset_from_directory(images_sorted_by_country_folder,
                                                label_mode="categorical",
                                                image_size=resize_size,
                                                batch_size=32,
                                                seed=42,
                                                shuffle=True,
                                                validation_split=test_size,
                                                subset="validation",
                                                color_mode="rgb",
                                                )
    

    # Get Class Weights (Weight All Classes Equally)
    # (Assumes image_dataset_from_directory orders labels same as folders with glob)
    class_weights = get_class_weights(images_sorted_by_country_folder)
    print(class_weights)

    # Train Model
    cnn.model.fit(
        x = dataset_train,
        batch_size = 32,
        epochs = 2000,
        validation_data = dataset_test,
        verbose = 1,
        callbacks = [ModelCheckpoint(country_classifier_model_folder + "/country_classifier_250_250.h5", save_best_only=True, save_weights_only=False, monitor='val_accuracy', mode='max')],
        class_weight = class_weights,
        )

  
def test():
    # Load Model
    cnn = CountryClassifierModel(num_countries=len(glob.glob(images_sorted_by_country_folder + "/*")))
    cnn.load_model(country_classifier_model_folder + "/country_classifier_250_250.h5")
    # Plot Confusion Matrix
    test_dataset = image_dataset_from_directory(images_sorted_by_country_folder,
                                                label_mode="categorical",
                                                image_size=resize_size,
                                                batch_size=32,
                                                seed=42,
                                                shuffle=True,
                                                validation_split=test_size,
                                                subset="validation",
                                                color_mode="rgb",
                                                )
    predictions = cnn.model.predict(test_dataset)
    predictions = np.array([np.argmax(pred) for pred in predictions], dtype=np.int32)
    print(predictions.shape)
    labels = np.concatenate([y for x, y in test_dataset], axis=0)
    labels = np.array([np.argmax(label) for label in labels], dtype=np.int32)
    print(labels.shape)
    print(f"Labels: {labels[0]}")
    print(f"Predictions: {predictions[0]}")
    # Plot Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(labels, predictions)
    plt.show()


# Train Model
# train()
test()