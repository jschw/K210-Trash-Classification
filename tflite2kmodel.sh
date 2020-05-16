#!/bin/bash
echo "usage: ./tflite2kmodel.sh xxx.tflite"
dataset=$1
dataset_img=data/$dataset/images_maix
name=$dataset.kmodel
out=train/$dataset/$name
in=train/$dataset/weights.tflite

./Maix_Toolbox/ncc/ncc -i tflite -o k210model --dataset $dataset_img $in $out
