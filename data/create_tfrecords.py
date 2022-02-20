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
records_folder = "data/records"
model_folder = "model"
resize_size = (250, 250)

# TF Record Info
record_size = 10000
test_size = 0.2

# From https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
class TfRecordsHelpers:
    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))): # if value ist tensor
            value = value.numpy() # get value of tensor
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a floast_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def serialize_array(array):
        array = tf.io.serialize_tensor(array)
        return array

    # Modified Method
    @staticmethod
    def parse_single_image(image, label):
  
        #define the dictionary -- the structure -- of our single example
        data = {
                'height' : TfRecordsHelpers._int64_feature(image.shape[0]),
                'width' : TfRecordsHelpers._int64_feature(image.shape[1]),
                'depth' : TfRecordsHelpers._int64_feature(image.shape[2]),
                'raw_image' : TfRecordsHelpers._bytes_feature(TfRecordsHelpers.serialize_array(image)),
                # Serialize Prediction Array -> 'label'
                'label' : TfRecordsHelpers._bytes_feature(TfRecordsHelpers.serialize_array(label)),
            }
        #create an Example, wrapping the single features
        out = tf.train.Example(features=tf.train.Features(feature=data))
        return out
    
    @staticmethod
    def write_images_to_tfr_short(images, labels, filename:str="images"):
        filename= filename+".tfrecords"
        writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
        count = 0

        for index in range(len(images)):

            #get the data we want to write
            current_image = images[index] 
            current_label = labels[index]

            out = TfRecordsHelpers.parse_single_image(image=current_image, label=current_label)
            writer.write(out.SerializeToString())
            count += 1

        writer.close()
        return count

    #++++++++++++++++++Reading TfRecord++++++++++++++++++++
    @staticmethod
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

    @staticmethod
    def get_dataset(filename):
        #create the dataset
        dataset = tf.data.TFRecordDataset(filename)

        #pass every single feature through our mapping function
        dataset = dataset.map(
            TfRecordsHelpers.parse_tfr_element
        )
            
        return dataset

# Save Scalars
def save_scalar(scalar, scalar_name):
    with open(model_folder + "/" + scalar_name, 'wb') as f:
        pickle.dump(scalar, f)

# Load Indieces of Dataset from Image Folder to Memory and Parse into Numpy Arrays
def load_dataset(dataset_directory, start_index, stop_index):
    # Image Locations
    image_locations = glob.glob(dataset_directory + "/*")[start_index:stop_index]

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
    save_scalar(standard_scalar_lat, "scalar_lat.p")
    save_scalar(standard_scalar_long, "scalar_long.p")

    # Return
    return images_loaded, labels

def load_scalar(scalar_name):
    with open(model_folder + "/" + scalar_name, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    
    # Get Total Number of Images
    num_images_training = int(len(glob.glob(image_folder + "/*")) * float(1-test_size))
    num_images_test = int(len(glob.glob(image_folder + "/*")) * float(test_size))
    print(f"Total Number of Training Images: {num_images_training}")
    print(f"Total Number of Test Images: {num_images_test}")
    print(f"Total Number of Images: {num_images_training + num_images_test}")

    # Keeps Track of Total Images Added to TfRecords
    total_count = 0
    naming_counter = 0
    # Iterate Through Images (Training)
    for i in range(0, num_images_training, record_size):
        if i <= num_images_training-record_size:
            # Load Data
            data_x, data_y = load_dataset(dataset_directory=image_folder, start_index=i, stop_index=i+record_size)
        else:
            print("Last Training Record")
            data_x, data_y = load_dataset(dataset_directory=image_folder, start_index=i, stop_index=num_images_training)
        # Print Shapes
        print(f"Data X Shape: {data_x.shape}")
        print(f"Data Y Shape: {data_y.shape}")

        # Write TFRecord With Data
        count = TfRecordsHelpers.write_images_to_tfr_short(data_x, data_y, filename=f"{records_folder}/record_train_{int(naming_counter)}")
        print(f"Wrote {count} elements to Training TFRecord {int(naming_counter)}")
        # Increase Naming Counter
        naming_counter += 1
        # Increase Total Count
        total_count += count

    # Iterate Through Images (Testing)
    naming_counter = 0
    for i in range(num_images_training, num_images_training+num_images_test, record_size):
        if i <= num_images_training+num_images_test-record_size:
            # Load Data
            data_x, data_y = load_dataset(dataset_directory=image_folder, start_index=i, stop_index=i+record_size)
        else:
            print("Last Testing Record")
            data_x, data_y = load_dataset(dataset_directory=image_folder, start_index=i, stop_index=num_images_training+num_images_test)
        # Print Shapes
        print(f"Data X Shape: {data_x.shape}")
        print(f"Data Y Shape: {data_y.shape}")

        # Write TFRecord With Data
        count = TfRecordsHelpers.write_images_to_tfr_short(data_x, data_y, filename=f"{records_folder}/record_test_{int(naming_counter)}")
        print(f"Wrote {count} elements to Testing TFRecord {int(naming_counter)}")
        # Increase Naming Counter
        naming_counter += 1
        # Increase Total Count
        total_count += count

    print(f"Total Number of Images Added to TFRecords: {total_count}")