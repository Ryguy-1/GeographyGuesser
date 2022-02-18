# Tensorflow 2.7.0
from matplotlib import units
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential, optimizers, losses
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
# Scikit-Learn Standard Scaler
from sklearn.preprocessing import StandardScaler
# Pickle
import pickle

# Globals
image_folder = "data/images"
model_folder = "model"
resize_size = (250, 250)
test_size = 0.2

def load_dataset(dataset_directory):
    # Image Locations
    image_locations = glob.glob(dataset_directory + "/*")[:10_000]
    # Load Images to Memory
    images_loaded = []
    labels = []
    # Iterate and Populate
    for file_loc in image_locations:
        loaded_image = cv2.resize(cv2.imread(file_loc), resize_size)
        images_loaded.append(loaded_image)
        labels.append((float(file_loc.split("\\")[-1].split(".p")[0].split("_")[0]), float(file_loc.split("\\")[-1].split(".p")[0].split("_")[-1])))
    # Convert to Numpy Arrays
    images_loaded = np.array(images_loaded, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    # Standardize Images
    images_loaded = images_loaded/255.0
    # Standardize Latitude
    standard_scalar_lat = StandardScaler()
    standard_scalar_lat.fit(labels[:,0].reshape(-1, 1))
    standardized_lat = standard_scalar_lat.transform(labels[:,0].reshape(-1, 1))
    labels[:, 0] = standardized_lat.reshape(-1)
    # Standardize Long (betwen -1 and 1)
    standard_scalar_long = StandardScaler()
    standard_scalar_long.fit(labels[:,1].reshape(-1, 1))
    standardized_long = standard_scalar_long.transform(labels[:,1].reshape(-1, 1))
    labels[:, 1] = standardized_long.reshape(-1)
    # Save Scalars
    save_scalar(standard_scalar_lat, "standard_scalar_lat.p")
    save_scalar(standard_scalar_long, "standard_scalar_long.p")
    # Return
    return images_loaded, labels

# (250, 250) CNN
class CNN_250_250:

    def __init__(self, input_shape=(250, 250, 3)):
        self.input_shape = input_shape
        self.model = Sequential()

        self.model.add(layers.Conv2D(filters = 128, kernel_size = (7, 7), strides=(3, 3), data_format="channels_last", activation=None, input_shape=self.input_shape))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.6))
        assert self.model.output_shape == (None, 82, 82, 128)

        self.model.add(layers.Conv2D(filters = 128, kernel_size = (7, 7), strides=(3, 3), data_format="channels_last", activation=None))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.6))
        assert self.model.output_shape == (None, 26, 26, 128)

        self.model.add(layers.Conv2D(filters = 128, kernel_size = (5, 5), strides=(3, 3), data_format="channels_last", activation=None))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.6))
        assert self.model.output_shape == (None, 8, 8, 128)

        self.model.add(layers.Flatten())
        assert self.model.output_shape == (None, 8192)

        self.model.add(layers.Dense(units=2048, activation=None))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.5))
        assert self.model.output_shape == (None, 2048)

        self.model.add(layers.Dense(units=512, activation=None))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.5))
        assert self.model.output_shape == (None, 512)

        self.model.add(layers.Dense(units=64, activation=None))
        self.model.add(layers.Activation("sigmoid"))
        assert self.model.output_shape == (None, 64)

        self.model.add(layers.Dense(units=2, activation='sigmoid'))
        assert self.model.output_shape == (None, 2)


        self.optimizer = optimizers.Adadelta(learning_rate=0.01) # was 0.01 -> got to 0.9975 mse
        self.loss_function = losses.MeanSquaredError()

        self.model.compile(
            optimizer = self.optimizer,
            loss = self.loss_function,
            metrics = ["mse"]
        )

    def save_model(self, model_name):
        self.model.save(model_name)
    
    def load_model(self, model_name):
        self.model = models.load_model(model_name)

def save_scalar(scalar, scalar_name):
    with open(model_folder + "/" + scalar_name, 'wb') as f:
        pickle.dump(scalar, f)

def train_model(data_x, data_y, test_size):
    # Split Data
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_size, random_state=42, shuffle=True)
    print(f"Train Shape: {train_x.shape}")
    print(f"Test Shape: {test_x.shape}")
    print(f"Train Label Shape: {train_y.shape}")
    print(f"Test Label Shape: {test_y.shape}")

    log_dir = f"{model_folder}/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train Model
    model = CNN_250_250()
    # Save model parameters every epoch by adding a callback that saves the model's weights to disk using the `ModelCheckpoint` callback.
    model.model.fit(
        train_x,
        train_y,
        batch_size = 64,
        epochs = 2000,
        validation_data = (test_x, test_y),
        verbose = 1,
        callbacks = [tensorboard_callback, ModelCheckpoint(model_folder + "/model_250_250.h5", save_best_only=True, save_weights_only=False)]
    )

if __name__ == "__main__":
    # Load Data
    data_x, data_y = load_dataset(dataset_directory=image_folder)
    print(data_x[0])
    print(data_y[0])
    # Print Shapes
    print(data_x.shape)
    print(data_y.shape)
    # Train Model
    train_model(data_x, data_y, test_size)
