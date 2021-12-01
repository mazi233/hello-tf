import io

import grpc
import numpy as np
from keras_preprocessing.image.utils import _PIL_INTERPOLATION_METHODS
from PIL import Image
from tensorflow.keras.applications.resnet50 import (decode_predictions,
                                                    preprocess_input)
from tensorflow.keras.preprocessing import image

from concurrent import futures
import infer_pb2
import infer_pb2_grpc
from infer_pb2 import Pred, PostProcessResponse, PreProcessResponse


class Processer(infer_pb2_grpc.ProcessServicer):
  def PreProcess(self, request, context):
    img = Image.open(io.BytesIO(request.image))
    img = img.convert('RGB')
    resample = _PIL_INTERPOLATION_METHODS['nearest']
    img = img.resize((224, 224), resample)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    return PreProcessResponse(shape=x.shape, data=x.reshape(-1))

  def PostProcess(self, request, context):
    preds = np.array(request.data).reshape(request.shape).astype(np.float32)
    preds = decode_predictions(preds, top=3)[0]
    preds = [Pred(name=name, probability=p) for _, name, p in preds]
    return PostProcessResponse(preds=preds)

def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=3))
  infer_pb2_grpc.add_ProcessServicer_to_server(Processer(), server)
  addr = '0.0.0.0:5001'
  server.add_insecure_port(addr)
  server.start()
  print('Listen on:', addr)
  server.wait_for_termination()

if __name__ == '__main__':
  serve()
