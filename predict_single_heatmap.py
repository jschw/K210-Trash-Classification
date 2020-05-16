# Imports

# TensorFlow ≥2.0 is required for this notebook
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Activation, Input, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, GaussianNoise
from tensorflow.keras.models import Model
#from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

import argparse
import os

# Run example:
# python predict_single_heatmap.py -dataset trash -image data/trash/logged_images/jpg/1273_001.jpg


tf.compat.v1.disable_eager_execution()

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-dataset", help="Name of actual dataset being used for training", required=True)
parser.add_argument("-image", help="Path to image for prediction", required=True)
args = parser.parse_args()

dataset_name = args.dataset


# Load data and preprocess
train_dir = 'data/' + dataset_name + '/images_train'
validation_dir = 'data/' + dataset_name + '/images_valid'
savepath = 'train/' + dataset_name + '/'
imagepath = args.image

# Transfer learning implementation of MobileNet model with freezed convolution layers
# and a fully connected classifier
base_model=tf.keras.applications.mobilenet.MobileNet(alpha = 0.75,depth_multiplier = 1, dropout = 0.001,include_top = False, weights = "imagenet", input_shape=(224,224,3))
model = GlobalAveragePooling2D()(base_model.output)
model = Dropout(0.001)(model)
output_layer = Dense(6, activation='softmax')(model)
model = Model(base_model.input, output_layer)

# Load saved weights
model.load_weights(savepath + 'weights.h5', by_name=True)


# Load and preprocess image
img = image.load_img(imagepath, target_size=(224, 224))

z = image.img_to_array(img)
z = np.expand_dims(z, axis=0)
z = preprocess_input(z)

# Make prediction
preds = model.predict(z)

maximum_model_output = model.output[:, 0]

last_conv_layer =  model.layers[83]

# Pooled grads of last convolutional layer and iterate over image
grads = K.gradients(model.output[:, 0], last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input],
                     [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([z])

# Extract 768 pooled grads of last convolutional layer
for i in range(768):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# Create and convert heatmap
heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

img = cv2.imread(imagepath)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Create superimposed image with heatmap
result = cv2.addWeighted(img, 0.8, heatmap, 0.4, 0)

# Format and show the plot
fig, ax = plt.subplots()
plt.imshow(result)
ax.tick_params(labelbottom=False, labelleft=False) 
plt.show()

