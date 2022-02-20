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
records_folder = "data/records"
model_folder = "model"
resize_size = (250, 250)
test_size = 0.2

# (250, 250) CNN
class CNN_250_250:

    def __init__(self, input_shape=(250, 250, 3)):
        self.input_shape = input_shape
        self.model = Sequential()

        self.model.add(layers.Conv2D(filters = 128, kernel_size = (7, 7), strides=(3, 3), data_format="channels_last", activation=None, input_shape=self.input_shape))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.2))
        assert self.model.output_shape == (None, 82, 82, 128)

        self.model.add(layers.Conv2D(filters = 128, kernel_size = (7, 7), strides=(3, 3), data_format="channels_last", activation=None))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.2))
        assert self.model.output_shape == (None, 26, 26, 128)

        self.model.add(layers.Conv2D(filters = 128, kernel_size = (5, 5), strides=(3, 3), data_format="channels_last", activation=None))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.2))
        assert self.model.output_shape == (None, 8, 8, 128)

        self.model.add(layers.Flatten())
        assert self.model.output_shape == (None, 8192)

        self.model.add(layers.Dense(units=4096, activation=None))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.4))
        assert self.model.output_shape == (None, 4096)

        self.model.add(layers.Dense(units=1024, activation=None))
        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.4))
        assert self.model.output_shape == (None, 1024)

        self.model.add(layers.Dense(units=256, activation=None))
        self.model.add(layers.Activation("sigmoid"))
        assert self.model.output_shape == (None, 256)

        self.model.add(layers.Dense(units=2, activation='sigmoid'))
        assert self.model.output_shape == (None, 2)


        self.optimizer = optimizers.Adam(learning_rate=0.00001)
        # self.loss_function = losses.MeanSquaredError()
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

        self.model.compile(
            optimizer = self.optimizer,
            # loss = self.loss_function,
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

def load_scalar(scalar_name):
    with open(model_folder + "/" + scalar_name, 'rb') as f:
        return pickle.load(f)

def load_tf_records_datasets(epochs, batch_size, records_directory):
    # Names of Records
    test_record_names = glob.glob(records_directory + "/*test*")
    train_record_names = glob.glob(records_directory + "/*train*")
    print(test_record_names)
    print(train_record_names)

    def parse_tfr_element(element):
        #use the same structure as above; it's kinda an outline of the structure we now want to create
        data = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width':tf.io.FixedLenFeature([], tf.int64),
            # Changed Label to float
            'label':tf.io.FixedLenFeature([], tf.string),
            'raw_image' : tf.io.FixedLenFeature([], tf.string),
            'depth':tf.io.FixedLenFeature([], tf.int64),
            }

            
        content = tf.io.parse_single_example(element, data)
        
        height = content['height']
        width = content['width']
        depth = content['depth']
        label = content['label']
        raw_image = content['raw_image']
        
        
        #get our 'feature'-- our image -- and reshape it appropriately
        feature = tf.io.parse_tensor(raw_image, out_type=tf.float32)
        feature = tf.reshape(feature, shape=[height,width,depth])
        # # Parse Label
        label = tf.io.parse_tensor(label, out_type=tf.float32)
        label = tf.reshape(label, shape=[2])
        return (feature, label)

    def get_dataset(filenames, epochs, batch_size):
        #create the dataset
        dataset = tf.data.TFRecordDataset(filenames)

        #pass every single feature through our mapping function
        dataset = dataset.map(
            parse_tfr_element
        )

        dataset.prefetch(10)
        dataset.repeat(epochs)
        dataset.shuffle(buffer_size=10 * batch_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        return dataset
    
    train_records = get_dataset(train_record_names, epochs, batch_size)
    test_records = get_dataset(test_record_names, epochs, batch_size)   

    return train_records, test_records

def train_model(records_directory):

    # Hyperparameters
    epochs = 100
    batch_size = 32

    log_dir = f"{model_folder}/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    
    # Records Datasets
    train_dataset, test_dataset = load_tf_records_datasets(epochs=epochs, batch_size=batch_size, records_directory=records_directory)


    # Train Model
    model = CNN_250_250()
    model.summary()
    # Save model parameters every epoch by adding a callback that saves the model's weights to disk using the `ModelCheckpoint` callback.
    model.model.fit(
        train_dataset,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = test_dataset,
        verbose = 1,
        callbacks = [tensorboard_callback, ModelCheckpoint(model_folder + "/model_250_250.h5", save_best_only=True, save_weights_only=False)]
    )

if __name__ == "__main__":
    # Train Model
    train_model(records_folder)

