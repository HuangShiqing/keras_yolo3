import tensorflow as tf
from keras.layers import *
from keras import backend as K
from keras.models import Model
from keras.regularizers import l2

import numpy as np
import cv2

img = cv2.imread('dog.jpg')
img = img[:, :, ::-1]  # RGB image
im_sized = cv2.resize(img, (416, 416))
im_sized = np.expand_dims(im_sized, axis=0)

weights_path = 'yolov3.weights'
# Load weights and config.
print('Loading weights.')
weights_file = open(weights_path, 'rb')
major, minor, revision = np.ndarray(
    shape=(3,), dtype='int32', buffer=weights_file.read(12))
if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
    seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
else:
    seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
print('Weights Header: ', major, minor, revision, seen)

conv_bias = np.ndarray(
    shape=(32,),  # (32,),
    dtype='float32',
    buffer=weights_file.read(32 * 4))
bn_weights = np.ndarray(
    shape=(3, 32),
    dtype='float32',
    buffer=weights_file.read(32 * 12))
bn_weight_list = [
    bn_weights[0],  # scale gamma
    conv_bias,  # shift beta
    bn_weights[1],  # running mean
    bn_weights[2]  # running var
]
weights_size = 32 * 3 * 3 * 3
conv_weights = np.ndarray(
    shape=[32, 3, 3, 3],  # [32, 3, 3, 3],
    dtype='float32',
    buffer=weights_file.read(weights_size * 4))
conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
conv_weights = [conv_weights]
# weights_file.close()

input_layer = Input(shape=(416, 416, 3))
conv_layer = Conv2D(
    filters=32, kernel_size=(3, 3),
    strides=(1, 1),
    kernel_regularizer=l2(5e-4),
    use_bias=False,
    weights=conv_weights,
    activation=None,
    padding='same',
    name='conv_1')(input_layer)
conv_layer = (BatchNormalization(
    weights=bn_weight_list))(conv_layer)
conv_layer = LeakyReLU(alpha=0.1)(conv_layer)
conv_layer = UpSampling2D(2)(conv_layer)
sess = K.get_session()
a = sess.run(conv_layer, feed_dict={input_layer: im_sized})

exit()
