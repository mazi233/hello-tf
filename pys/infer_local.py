# 引入一些库
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import (decode_predictions, preprocess_input)

# 加载模型
loaded = tf.saved_model.load('pys/resnet50')
infer = loaded.signatures['serving_default']

# 准备数据
img = image.load_img('pys/elephant.jpeg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)

# 进行推理
preds = infer(tf.constant(x))['predictions'].numpy()
print(decode_predictions(preds, top=3)[0])

print('X: ', x.shape, x.dtype)
with open('pys/request', 'wb') as f:
  f.write(x.tobytes())
