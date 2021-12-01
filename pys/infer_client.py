import numpy as np
# import tensorflow as tf
from tensorflow.keras.applications.resnet50 import (decode_predictions, preprocess_input)

from tensorflow.keras.preprocessing import image

import grpc
import infer_pb2
import infer_pb2_grpc

# 准备推理数据
img = image.load_img('pys/elephant.jpeg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x) # (1*224*224*3)

# 进行推理
with grpc.insecure_channel('localhost:5000') as chan:
  stub = infer_pb2_grpc.InferStub(chan)
  req = infer_pb2.InferRequest(shape=x.shape, data=x.reshape(-1))
  res = stub.Infer(req)
  preds = np.array(res.data).reshape(res.shape)
  print(decode_predictions(preds, top=3)[0])
