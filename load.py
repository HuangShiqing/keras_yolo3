import tensorflow as tf
from keras.layers import *
from keras import backend as K
from keras.models import Model, load_model
from keras.optimizers import Adam

from train import yolo3_loss
from data import data_generator
# def yolo3_loss(args):
#     return args


C = 80

y_true = list()
y_true.append(Input(name='input0', shape=[13, 13, 3, (5 + C)]))
y_true.append(Input(name='input1', shape=[26, 26, 3, (5 + C)]))
y_true.append(Input(name='input2', shape=[52, 52, 3, (5 + C)]))

model_load = load_model('model.h5', compile=False)
loss_layer = Lambda(yolo3_loss, name='loss_layer')([*model_load.output, *y_true])
model = Model(inputs=[model_load.input, *y_true], outputs=loss_layer)

model.compile(optimizer=Adam(lr=1e-3), loss=lambda y, y_: y_)
model.fit_generator(generator=data_generator(images_path="D:/DeepLearning/data/VOCdevkit/VOC2012/JPEGImages/",
                   annotations_path="D:/DeepLearning/data/VOCdevkit/VOC2012/Annotations/", batch_size=16),
                    )

exit()
