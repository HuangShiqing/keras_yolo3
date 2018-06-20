import numpy as np
import tensorflow as tf
from keras.layers import *
from keras import backend as K
from keras.models import Model
from net import Gb_all_layers, Gb_out_index, conv2d_unit, shortcut, route, detection, upsample

from net import weights_file

num_classes = 80
print('Creating Keras model.')
print('layer     filters    size              input                output')
input_layer = Input(shape=(416, 416, 3))
net = conv2d_unit(input_layer, filters=32, kernels=3, strides=1, bn=True, activation='leaky')  # 0
net = conv2d_unit(net, filters=64, kernels=3, strides=2, bn=True, activation='leaky')  # 1
net = conv2d_unit(net, filters=32, kernels=1, strides=1, bn=True, activation='leaky')  # 2
net = conv2d_unit(net, filters=64, kernels=3, strides=1, bn=True, activation='leaky')  # 3
net = shortcut(net, 1)  # 4
net = conv2d_unit(net, filters=128, kernels=3, strides=2, bn=True, activation='leaky')  # 5
net = conv2d_unit(net, filters=64, kernels=1, strides=1, bn=True, activation='leaky')  # 6
net = conv2d_unit(net, filters=128, kernels=3, strides=1, bn=True, activation='leaky')  # 7
net = shortcut(net, 5)  # 8
net = conv2d_unit(net, filters=64, kernels=1, strides=1, bn=True, activation='leaky')  # 9
net = conv2d_unit(net, filters=128, kernels=3, strides=1, bn=True, activation='leaky')  # 10
net = shortcut(net, 8)  # 11
net = conv2d_unit(net, filters=256, kernels=3, strides=2, bn=True, activation='leaky')  # 12
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, activation='leaky')  # 13
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, activation='leaky')  # 14
net = shortcut(net, 12)  # 15
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, activation='leaky')  # 16
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, activation='leaky')  # 17
net = shortcut(net, 15)  # 18
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, activation='leaky')  # 19
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, activation='leaky')  # 20
net = shortcut(net, 18)  # 21
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, activation='leaky')  # 22
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, activation='leaky')  # 23
net = shortcut(net, 21)  # 24
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, activation='leaky')  # 25
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, activation='leaky')  # 26
net = shortcut(net, 24)  # 27
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, activation='leaky')  # 28
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, activation='leaky')  # 29
net = shortcut(net, 27)  # 30
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, activation='leaky')  # 31
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, activation='leaky')  # 32
net = shortcut(net, 30)  # 33
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, activation='leaky')  # 34
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, activation='leaky')  # 35
net = shortcut(net, 33)  # 36
net = conv2d_unit(net, filters=512, kernels=3, strides=2, bn=True, activation='leaky')  # 37
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, activation='leaky')  # 38
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, activation='leaky')  # 39
net = shortcut(net, 37)  # 40
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, activation='leaky')  # 41
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, activation='leaky')  # 42
net = shortcut(net, 40)  # 43
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, activation='leaky')  # 44
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, activation='leaky')  # 45
net = shortcut(net, 43)  # 46
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, activation='leaky')  # 47
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, activation='leaky')  # 48
net = shortcut(net, 46)  # 49
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, activation='leaky')  # 50
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, activation='leaky')  # 51
net = shortcut(net, 49)  # 52
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, activation='leaky')  # 53
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, activation='leaky')  # 54
net = shortcut(net, 52)  # 55
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, activation='leaky')  # 56
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, activation='leaky')  # 57
net = shortcut(net, 55)  # 58
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, activation='leaky')  # 59
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, activation='leaky')  # 60
net = shortcut(net, 58)  # 61
net = conv2d_unit(net, filters=1024, kernels=3, strides=2, bn=True, activation='leaky')  # 62
net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, activation='leaky')  # 63
net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, activation='leaky')  # 64
net = shortcut(net, 62)  # 65
net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, activation='leaky')  # 66
net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, activation='leaky')  # 67
net = shortcut(net, 65)  # 68
net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, activation='leaky')  # 69
net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, activation='leaky')  # 70
net = shortcut(net, 68)  # 71
net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, activation='leaky')  # 72
net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, activation='leaky')  # 73
net = shortcut(net, 71)  # 74
net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, activation='leaky')  # 75
net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, activation='leaky')  # 76
net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, activation='leaky')  # 77
net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, activation='leaky')  # 78
net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, activation='leaky')  # 79
net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, activation='leaky')  # 80
yolo_1 = conv2d_unit(net, filters=3 * (5 + num_classes), kernels=1, strides=1, bn=False, activation='linear')  # 81
detection(82)
net = route([79])  # 83
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, activation='leaky')  # 84
net = upsample(net)  # 85
net = route([85, 61])  # 86
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, activation='leaky')  # 87
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, activation='leaky')  # 88
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, activation='leaky')  # 89
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, activation='leaky')  # 90
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, activation='leaky')  # 91
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, activation='leaky')  # 92
yolo_2 = conv2d_unit(net, filters=3 * (5 + num_classes), kernels=1, strides=1, bn=False, activation='linear')  # 93
detection(94)  # 94
net = route([91])  # 95
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, activation='leaky')  # 96
net = upsample(net)  # 97
net = route([97, 36])  # 98
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, activation='leaky')  # 99
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, activation='leaky')  # 100
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, activation='leaky')  # 101
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, activation='leaky')  # 102
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, activation='leaky')  # 103
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, activation='leaky')  # 104
yolo_3 = conv2d_unit(net, filters=3 * (5 + num_classes), kernels=1, strides=1, bn=False, activation='linear')  # 105
detection(106)  # 106

model = Model(inputs=input_layer, outputs=[yolo_3, yolo_2, yolo_1])
model.save('model.h5')
# a = model.output

weights_file.close()



# sess = K.get_session()
# a = tf.global_variables()
# h = sess.run(tf.global_variables("conv2d_75/kernel")[0])
exit()
