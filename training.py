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

# Globals
image_folder = "data/images"
model_folder = "model"
resize_size = (250, 250)
test_size = 0.2

def load_dataset(dataset_directory):
    # Image Locations
    image_locations = glob.glob(dataset_directory + "/*")
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
    # Make All Latitudes Positive
    labels[:, 0] = labels[:, 0] + 90
    # make All Longitudes Positive
    labels[:, 1] = labels[:, 1] + 180
    # Normalize Images
    images_loaded = images_loaded/255.0
    # Normalize Latitude
    standard_scalar_lat = MinMaxScaler()
    standard_scalar_lat.fit(labels[:,0].reshape(-1, 1))
    standardized_lat = standard_scalar_lat.transform(labels[:,0].reshape(-1, 1))
    labels[:, 0] = standardized_lat.reshape(-1)
    # Normalize Long (betwen -1 and 1)
    standard_scalar_long = MinMaxScaler()
    standard_scalar_long.fit(labels[:,1].reshape(-1, 1))
    standardized_long = standard_scalar_long.transform(labels[:,1].reshape(-1, 1))
    labels[:, 1] = standardized_long.reshape(-1)
    print(labels)
    # Save Scalars
    save_scalar(standard_scalar_lat, "scalar_lat.p")
    save_scalar(standard_scalar_long, "scalar_long.p")
    print(load_scalar("scalar_long.p").inverse_transform(labels[0, 1].reshape(-1, 1)))
    # Return
    return images_loaded, labels

# (250, 250) CNN
class CNN_250_250:

    def __init__(self, input_shape=(250, 250, 3)):
        self.input_shape = input_shape
        self.model = Sequential()

        self.model.add(layers.Conv2D(filters = 32, kernel_size = (7, 7), strides=(3, 3), data_format="channels_last", activation=None, input_shape=self.input_shape))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.2))
        assert self.model.output_shape == (None, 82, 82, 32)

        self.model.add(layers.Conv2D(filters = 64, kernel_size = (7, 7), strides=(3, 3), data_format="channels_last", activation=None))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.2))
        assert self.model.output_shape == (None, 26, 26, 64)

        self.model.add(layers.Conv2D(filters = 64, kernel_size = (5, 5), strides=(3, 3), data_format="channels_last", activation=None))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.2))
        assert self.model.output_shape == (None, 8, 8, 64)

        self.model.add(layers.Flatten())
        assert self.model.output_shape == (None, 4096)

        self.model.add(layers.Dense(units=1024, activation=None))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.4))
        assert self.model.output_shape == (None, 1024)

        self.model.add(layers.Dense(units=256, activation=None))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.4))
        assert self.model.output_shape == (None, 256)

        self.model.add(layers.Dense(units=64, activation=None))
        self.model.add(layers.Activation("sigmoid"))
        assert self.model.output_shape == (None, 64)

        self.model.add(layers.Dense(units=2, activation='sigmoid'))
        assert self.model.output_shape == (None, 2)


        self.optimizer = optimizers.Adam(learning_rate=0.00001)
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

def load_scalar(scalar_name):
    with open(model_folder + "/" + scalar_name, 'rb') as f:
        return pickle.load(f)

def train_model(data_x, data_y, test_size):
    # Split Data
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_size, random_state=42, shuffle=True)
    print(f"Train Shape: {train_x.shape}")
    print(f"Test Shape: {test_x.shape}")
    print(f"Train Label Shape: {train_y.shape}")
    print(f"Test Label Shape: {test_y.shape}")

    log_dir = f"{model_folder}/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Convert to Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    # Create a generator for the training data
    train_generator = train_dataset.shuffle(buffer_size=1000).batch(16)
    # Create a generator for the test data
    test_generator = test_dataset.batch(16)

    # Train Model
    model = CNN_250_250()
    # Save model parameters every epoch by adding a callback that saves the model's weights to disk using the `ModelCheckpoint` callback.
    model.model.fit(
        train_generator,
        batch_size = 16,
        epochs = 2000,
        validation_data = test_generator,
        verbose = 1,
        callbacks = [tensorboard_callback, ModelCheckpoint(model_folder + "/model_250_250.h5", save_best_only=True, save_weights_only=False)]
    )

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
        label = (float(image_loc.split("\\")[-1].split(".p")[0].split("_")[0]), float(image_loc.split("\\")[-1].split(".p")[0].split("_")[-1]))
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
    prediction = model.model.predict(image)
    # Unstandardize Prediction
    lat_predicted = load_scalar("scalar_lat.p").inverse_transform(prediction[0][0].reshape(1, -1))

    long_predicted = load_scalar("scalar_long.p").inverse_transform(prediction[0][1].reshape(1, -1))

    # Return
    return np.array([lat_predicted[0][0]-90, long_predicted[0][0]-180], dtype=np.float32), label_unstandardized, image, prediction


if __name__ == "__main__":
    # Load Data
    data_x, data_y = load_dataset(dataset_directory=image_folder)
    # Print Shapes
    print(data_x.shape)
    print(data_y.shape)
    # # Train Model
    train_model(data_x, data_y, test_size)

    # Test
    model = CNN_250_250()
    model.load_model(model_folder + "/model_250_250.h5")
    for i in range(0, 10_000, 10):
        predicted, true, image, prediction_raw = test_individual_images(model, i, image_folder)
        print(f"Predicted: {predicted}")
        print(f"True: {true}")
        print(f"Prediction Raw: {prediction_raw}")
        cv2.imshow("Image", image[0])
        cv2.waitKey(0)
