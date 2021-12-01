import tensorflow as tf

from tensorflow.keras.applications.resnet50 import (ResNet50,decode_predictions,preprocess_input)


model = ResNet50(weights="imagenet")

model.save('pys/resnet50')
