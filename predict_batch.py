# Imports

# TensorFlow ≥2.0 is required for this notebook
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Activation, Input, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

import argparse
import os


# Run example:
# python predict_batch.py -dataset trash -image_dir data/trash/images_maix



# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-dataset", help="Name of actual dataset being used for training", required=True)
parser.add_argument("-image_dir", help="Path to image directory for prediction", required=True)
args = parser.parse_args()

dataset_name = args.dataset



# Load data and preprocess
train_dir = 'data/' + dataset_name + '/images_train'
validation_dir = 'data/' + dataset_name + '/images_valid'
savepath = 'train/' + dataset_name + '/'
imagedir = args.image_dir


from tensorflow.keras.applications.mobilenet import preprocess_input

# Data generators
test_datagen = ImageDataGenerator(
    validation_split=0.1,

	preprocessing_function=preprocess_input
	)

predict_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
    )


print('')
validation_generator = test_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        class_mode='sparse',
        subset='validation'
    	)

print('')
print('Search for prediction images in subfolders of path <<' + imagedir + '>>: ')
prediction_generator = predict_datagen.flow_from_directory(
        imagedir,
        target_size=(224, 224),
        shuffle=False,
        class_mode=None,
        )


classes = validation_generator.class_indices


# Print class labels and indices
print('')
print('class names: ', classes)
n_classes = len(classes)
print('number of classes: ', n_classes)
print('')



# Transfer learning implementation of MobileNet model with freezed convolution layers
# and a fully connected classifier
base_model=tf.keras.applications.mobilenet.MobileNet(alpha = 0.75,depth_multiplier = 1, dropout = 0.001,include_top = False, weights = "imagenet", input_shape=(224,224,3))
model = GlobalAveragePooling2D()(base_model.output)
model = Dropout(0.001)(model)
output_layer = Dense(n_classes, activation='softmax')(model)
model = Model(base_model.input, output_layer)

# Load saved weights
model.load_weights(savepath + 'weights.h5', by_name=True)


# Make predictions
preds = model.predict(prediction_generator)

preds_class_indices = preds.argmax(axis=-1)

# Convert labels and predictions to dictionaries
labels = dict((v,k) for k,v in classes.items())
predictions = [labels[k] for k in preds_class_indices]

# Put them together with the filenames into a dataframe
filenames = prediction_generator.filenames
results = pd.DataFrame({"Filename":filenames, "Prediction":predictions})

print('')
print(results)
print('')
print('')
