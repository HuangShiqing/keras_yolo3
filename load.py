import tensorflow as tf
from keras.layers import *
from keras import backend as K
from keras.models import Model, load_model

from PIL import Image
import cv2
# image = Image.open('dog.jpg')
# image = image.resize((416, 416), Image.BICUBIC)
# image_data = np.array(image, dtype='float32')
# image_data = image_data / 255.
# image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
img = cv2.imread('dog.jpg')
img = img[:, :, ::-1]  # RGB image
im_sized = cv2.resize(img, (416, 416))
image_data = np.array(im_sized, dtype='float32')
image_data /= 255.
image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

model = load_model('model.h5', compile=False)

num_classes = 80

anchors = tf.constant([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                      dtype='float', shape=[1, 1, 1, 9, 2])
input_shape = K.shape(model.output[2])[1:3] * 32

boxes = list()
scores = list()
classes = list()
for i in range(3):  # 52 26 13
    anchor = anchors[..., 3 * i:3 * (i + 1), :]
    feats = model.output[i]

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], 3, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    box_scores = box_confidence * box_class_probs
    max_scores = tf.reduce_max(box_scores, axis=-1)

    scores.append(max_scores)

sess = K.get_session()
s = sess.run(scores, feed_dict={model.input: image_data,
                                K.learning_phase(): 0})

exit()
