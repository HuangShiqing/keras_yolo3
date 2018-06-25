import tensorflow as tf
from keras.layers import *
from keras import backend as K
from keras.models import Model, load_model
from keras.optimizers import Adam

from train import yolo3_loss
from data import data_generator

C = 80
# images_path = "D:/DeepLearning/data/VOCdevkit/VOC2012/JPEGImages/"
# annotations_path = "D:/DeepLearning/data/VOCdevkit/VOC2012/Annotations/"
# batch_size = 16
# pick = []

y_true_input = list()
y_true_input.append(Input(name='input0', shape=[52, 52, 3, (5 + C)]))
y_true_input.append(Input(name='input1', shape=[26, 26, 3, (5 + C)]))
y_true_input.append(Input(name='input2', shape=[13, 13, 3, (5 + C)]))

model_load = load_model('model.h5', compile=False)
loss_layer = Lambda(yolo3_loss, name='loss_layer')([*model_load.output, *y_true_input])
model = Model(inputs=[model_load.input, *y_true_input], outputs=loss_layer)

model.compile(optimizer=Adam(lr=1e-3), loss=lambda y_true, y_pred: y_pred)
model.fit_generator(generator=data_generator(), steps_per_epoch=17125 / 16, epochs=30)

exit()
