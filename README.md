# Trash Classification on Kendryte K210 Embedded System

![Classification example](/classification_example.jpeg)

Simple trash image classifier, implemented on Kendryte K210 (Boards Sipeed Maix Go & M5Stick AI Camera).

See detailed project description (German): ...

## Dataset

Image dataset is forked from original dataset https://github.com/garythung/trashnet .

In this project, "plastic" class was splitted in "plastic_bottle" and "plastic_other".

## Training

Simply run:
`python3 training.py -dataset <Name of Dataset> -val_split 0.2`

(dataset name is the name of the folder containing training images in 'data')

## Evaluation

For batch predicting all images in a folder, run: `python3 predict_batch.py -dataset trash -image_dir <Test image folder>`

For single image prediction and show the heatmap, run: `python3 predict_single_heatmap.py -dataset trash -image <path to image>`

## Running on K210

Just copy all contents of "sdcard_content" to your SD card (Have to be FAT32 formatted!).

If you trained your own model, use nncase compiler (https://github.com/kendryte/nncase) to convert the *.tflite* file created after the training to the kmodel format.

Have fun! :)
