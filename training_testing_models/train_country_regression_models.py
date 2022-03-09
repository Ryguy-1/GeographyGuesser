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
# Os
import os


# Globals
images_sorted_by_country_folder = "data/images_sorted_by_country"
country_regression_model_folder = "models/country_regression_models"
resize_size = (250, 250)
test_size = 0.2


# (250, 250) CountryRegressionModel
class CountryRegressionModel:

    def __init__(self, input_shape=(250, 250, 3)):
        self.input_shape = input_shape
        self.model = Sequential()

        self.model.add(layers.Conv2D(filters = 8, kernel_size = (7, 7), strides=(3, 3), data_format="channels_last", activation=None, input_shape=self.input_shape))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.7))
        assert self.model.output_shape == (None, 82, 82, 8)

        self.model.add(layers.Conv2D(filters = 16, kernel_size = (7, 7), strides=(3, 3), data_format="channels_last", activation=None))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        assert self.model.output_shape == (None, 26, 26, 16)
        # Pool Later
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
        # Dropout
        self.model.add(layers.Dropout(0.5))
        assert self.model.output_shape == (None, 13, 13, 16)

        self.model.add(layers.Flatten())
        assert self.model.output_shape == (None, 13 * 13 * 16)

        self.model.add(layers.Dense(units=1024, activation=None))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.7))
        assert self.model.output_shape == (None, 1024)

        self.model.add(layers.Dense(units=512, activation=None))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.5))
        assert self.model.output_shape == (None, 512)

        self.model.add(layers.Dense(units=2, activation='sigmoid'))
        assert self.model.output_shape == (None, 2)

        self.optimizer = optimizers.Adam(learning_rate=0.00001)
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
            # Implement Geoguessr Scoring Algorithm (y=4999.91(0.998036)^x) (ish)
            loss = tf.constant(5000, dtype=tf.float32) - tf.constant(5000, dtype=tf.float32) * tf.pow(tf.constant(0.999, dtype=tf.float32), distance_lat_long)
            # Reduce Mean of Losses
            loss = tf.reduce_mean(loss)
            # Return Loss
            return loss

        self.model.compile(
            optimizer = self.optimizer,
            loss = custom_loss,
            # Custom loss metric
            metrics = [custom_loss]
        )

    def summary(self, ):
        self.model.summary()

    def save_model(self, model_name):
        self.model.save(model_name)
    
    def load_model(self, model_name):
        self.model = models.load_model(model_name)


# Load Indieces of Dataset from Image Folder to Memory and Parse into Numpy Arrays
def load_dataset(image_locations, model_folder):
    # Image Locations
    image_locations = image_locations

    # Load Images to Memory
    images_loaded = []
    labels = []
    # Iterate and Populate
    images_loaded_counter = 0
    for file_loc in image_locations:
        try:
            loaded_image = cv2.resize(cv2.imread(file_loc), resize_size)
        except Exception:
            print("Error Loading Image: " + file_loc)
        images_loaded.append(loaded_image)
        labels.append((float(file_loc.split("\\")[-1].split(".p")[0].split("_")[0]), float(file_loc.split("\\")[-1].split(".p")[0].split("_")[-1])))
        images_loaded_counter += 1
        if images_loaded_counter % 1000 == 0:
            print(f"Loaded {images_loaded_counter} Images")
    
    # Convert to Numpy Arrays
    images_loaded = np.array(images_loaded, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # Make All Latitudes Positive
    labels[:, 0] = labels[:, 0] + 90

    # make All Longitudes Positive
    labels[:, 1] = labels[:, 1] + 180

    # Normalize Images
    images_loaded = images_loaded/255.0

    # Normalize Latitude (between 0 and 1)
    standard_scalar_lat = MinMaxScaler()
    standard_scalar_lat.fit(labels[:,0].reshape(-1, 1))
    standardized_lat = standard_scalar_lat.transform(labels[:,0].reshape(-1, 1))
    labels[:, 0] = standardized_lat.reshape(-1)

    # Normalize Long (betwen 0 and 1)
    standard_scalar_long = MinMaxScaler()
    standard_scalar_long.fit(labels[:,1].reshape(-1, 1))
    standardized_long = standard_scalar_long.transform(labels[:,1].reshape(-1, 1))
    labels[:, 1] = standardized_long.reshape(-1)

    # Save Scalars
    save_scalar(standard_scalar_lat, "scalar_lat.p", model_folder)
    save_scalar(standard_scalar_long, "scalar_long.p", model_folder)

    # Return
    return images_loaded, labels

# Save Scalars
def save_scalar(scalar, scalar_name, model_folder):
    with open(model_folder + "/" + scalar_name, 'wb') as f:
        pickle.dump(scalar, f)

# Load Scalars
def load_scalar(scalar_name, model_folder):
    with open(model_folder + "/" + scalar_name, 'rb') as f:
        return pickle.load(f)


# Train Models In Order
def train_model(model, images_loaded, labels, epochs, batch_size, model_folder):

    # Train Test Split
    train_images, test_images, train_labels, test_labels = train_test_split(images_loaded, labels, train_size=0.8, test_size=0.2, random_state=42)
    print(train_images.shape)
    print(test_images.shape)

    # Train Model
    model.model.fit(
        x = train_images,
        y = train_labels,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = (test_images, test_labels),
        verbose = 1,
        callbacks = [ModelCheckpoint(model_folder + "/regression_250_250.h5", save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min')],
    )



if __name__ == "__main__":
    # Load All Country Folders
    country_folders = glob.glob(images_sorted_by_country_folder + "/*")
    # Train Model For Each Country
    for country_folder in country_folders:
        # Initialize Model Folder Location
        if not os.path.exists(country_regression_model_folder + "/" + country_folder.split("\\")[-1]):
            os.makedirs(country_regression_model_folder + "/" + country_folder.split("\\")[-1])
        model_folder = country_regression_model_folder + "/" + country_folder.split("\\")[-1]
        # Get Image Locations
        image_locatons = glob.glob(country_folder + "/*")
        # Check if image locations greater than 1
        if len(image_locatons) == 1:
            # Write Text File to Indicate Latitude and Longitude of One Image
            with open(model_folder + "/no_images.txt", 'w') as f:
                lat = image_locatons[0].split('\\')[-1].split(".p")[0].split('_')[0]
                long = image_locatons[0].split('\\')[-1].split(".p")[0].split('_')[1]
                string_write = f"{lat}, {long}"
                f.write(string_write)
            continue
        # Load Dataset
        images_loaded, labels = load_dataset(image_locatons, model_folder)
        # Train Model
        train_model(CountryRegressionModel(), images_loaded, labels, 200, 16, model_folder)