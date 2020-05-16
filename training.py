# Imports

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sn
import pandas as pd

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Activation, Input, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import load_model

from sklearn.metrics import classification_report, confusion_matrix

import argparse
import os


# Run example:
# python training.py -dataset trash_small_nosplit -val_split 0.1


# General functions
def plot_learning_curve(
    title: str, x: int, y: int, y_test: int, ylim: float = 0.6, path: str = '') -> None:
    plt.figure()
    plt.title(title)
    axes = plt.gca()
    axes.set_ylim([ylim, 1])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    train_sizes = x
    train_scores = y
    test_scores = y_test

    plt.grid()

    plt.plot(
        train_sizes,
        train_scores,
        "o-",
        label="Training accuracy",
    )
    plt.plot(
        train_sizes,
        test_scores,
        "o-",
        label="Validation accuracy",
    )

    plt.legend(loc="best")

    plt.savefig(path + 'train_result.png', bbox_inches='tight')



def plot_two_histories(history: "History", history_finetune: "History", path: str = '') -> None:
    y = history.history["accuracy"] + history_finetune.history["accuracy"]
    y_test = history.history["val_accuracy"] + history_finetune.history["val_accuracy"]
    plot_learning_curve("Training process", np.arange(1, 1 + len(y)), y, y_test, 0, path)



def save_tflite(model, path: str):
    # KPU V4 - nncase >= 0.2.0
    #converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #tflite_model = converter.convert()
    
    # KPU V3 - nncase = 0.1.0rc5
    #model.save(path + 'weights.h5', include_optimizer=False) # <== Save model without callback flag in mode.fit

    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(path + 'weights.h5')

    tfmodel = converter.convert()
    file = open (path + 'weights.tflite' , 'wb')
    file.write(tfmodel)




# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-dataset", help="Name of actual dataset being used for training", required=True)
parser.add_argument("-val_split", type=float, help="Split ratio for validation dataset", required=True)
args = parser.parse_args()

dataset_name = args.dataset
val_ratio = args.val_split



# Load data and preprocess
train_dir = 'data/' + dataset_name + '/images_train'
validation_dir = 'data/' + dataset_name + '/images_valid'
savepath = 'train/' + dataset_name + '/'

# Create train folder if not exist
if not os.path.exists(savepath):
    os.makedirs(savepath)

batch_size = 32


from tensorflow.keras.applications.mobilenet import preprocess_input

# Data generators
train_datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.1,
      horizontal_flip=True,
      vertical_flip=True,
      validation_split=val_ratio,
      fill_mode='nearest',

      preprocessing_function=preprocess_input

      )

# Validation data should not be augmented
test_datagen = ImageDataGenerator(
    validation_split=val_ratio,

	preprocessing_function=preprocess_input
	)



print('')
print('Search for train images: ')
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        subset='training'
    	)

print('')
print('Search for validation images: ')
validation_generator = test_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        class_mode='sparse',
        subset='validation'
    	)


classes = train_generator.class_indices


# Print class labels and indices
print('')
print('class names: ', classes)
n_classes = len(classes)
print('number of classes: ', n_classes)
print('')

# Save classnames to file
classfile = open(savepath + 'classnames.txt','w')
classfile.write( str(classes) )
classfile.close()



# Transfer learning implementation of MobileNet model with freezed convolution layers
# and a fully connected classifier
base_model=tf.keras.applications.mobilenet.MobileNet(alpha = 0.75,depth_multiplier = 1, dropout = 0.001,include_top = False, weights = "imagenet", input_shape=(224,224,3))

model = GlobalAveragePooling2D()(base_model.output)
model = Dropout(0.001)(model)
output_layer = Dense(n_classes, activation='softmax')(model)

model = Model(base_model.input, output_layer)

# Save model structure to file
with open(savepath+ 'model.txt','w') as txtfile:
    model.summary(print_fn=lambda x: txtfile.write(x + '\n'))

# Freeze layers
for layer in base_model.layers:
  layer.trainable = False


# Compile and train the classifier
model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator
)




# Unfreeze middle convolution layers and retrain with low LR

for layer in base_model.layers:
  layer.trainable = True


model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Create callback and save the model with best (=highest) validation accuracy
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                filepath=savepath + 'weights.h5',
                monitor='val_accuracy',
                save_best_only=True
                )

# Create callback for saving training step info to file
csvlogger_cb = CSVLogger(savepath + 'finetune_training_log.csv', append=True, separator=';')


history_finetune=model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[checkpoint_cb, csvlogger_cb]

)


# Plot result and save
plot_two_histories(history, history_finetune, savepath)

 
# Save as tflite to disk
save_tflite(model, savepath)


# Load best saved model for evaluation
model_eval = load_model(savepath + 'weights.h5')


# Evaluate model
val_loss, val_acc = model_eval.evaluate(validation_generator)

print('')
print('Final validation loss:')
print(val_loss)
print('')
print('Final validation accuracy:')
print(val_acc)
print('')


# Create confusion matrix
# Make predictions on validation data
preds = model_eval.predict(validation_generator)
preds_class_indices = preds.argmax(axis=1)


cm = confusion_matrix(validation_generator.classes, preds_class_indices)
print(classification_report(validation_generator.classes, preds_class_indices, target_names=classes))

# Normalize confusion matrix (optional)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm *= 100.

df_cm = pd.DataFrame(cm, classes, classes)

# Format and save plot
plt.figure(figsize = (10,7))
plt.title('Confusion matrix')
svm = sn.heatmap(df_cm, annot=True)
plt.xlabel("Predicted label")
plt.ylabel("True label")
svm.get_figure().savefig(savepath + 'confusion_matrix.png', dpi=400)
plt.show()
