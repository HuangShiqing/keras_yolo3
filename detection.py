import tensorflow as tf
from keras.layers import *
from keras import backend as K
from keras.models import Model, load_model

from PIL import Image
import cv2
import matplotlib.pyplot as plt

# image = Image.open('dog.jpg')
# image = image.resize((416, 416), Image.BICUBIC)
# image_data = np.array(image, dtype='float32')
# image_data = image_data / 255.
# image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
img = cv2.imread('dog.jpg')
img = img[:, :, ::-1]  # RGB image
img_shape = img.shape[0:2][::-1]

_scale = min(416 / img_shape[0], 416 / img_shape[1])
_new_shape = (int(img_shape[0] * _scale), int(img_shape[1] * _scale))
im_sized = cv2.resize(img, _new_shape)
im_sized = np.pad(im_sized,
                  (
                      (int((416 - _new_shape[1]) / 2), 416 - _new_shape[1] - int((416 - _new_shape[1]) / 2)),
                      (int((416 - _new_shape[0]) / 2), 416 - _new_shape[0] - int((416 - _new_shape[0]) / 2)),
                      (0, 0)
                  ),
                  mode='constant')
# plt.imshow(im_sized)
# plt.show()
# im_sized = cv2.resize(img, (416, 416))
image_data = np.array(im_sized, dtype='float32')
image_data /= 255.
image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

model = load_model('model.h5', compile=False)
net_out = model.output
# net_out = [K.zeros(shape=(1, 52, 52, 3, 85)), K.zeros(shape=(1, 26, 26, 3, 85)), K.zeros(shape=(1, 13, 13, 3, 85))]

num_classes = 80
anchors = tf.constant([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                      dtype='float', shape=[1, 1, 1, 9, 2])
input_shape = K.cast(K.shape(net_out[2])[1:3] * 32, dtype='float32')[::-1]  # hw
image_shape = K.cast(img_shape, dtype='float32')[::-1]  # hw
new_shape = K.round(image_shape * K.min(input_shape / image_shape))
offset = (input_shape - new_shape) / 2. / input_shape
scale = input_shape / new_shape

# sess = K.get_session()
# a,b = sess.run([input_shape,image_shape], feed_dict={K.learning_phase(): 0})


boxes = list()
box_scores = list()
# classes = list()
for i in range(3):  # 52 26 13
    anchor = anchors[..., 3 * i:3 * (i + 1), :]
    # feats = model.output[i]
    feats = net_out[i]

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

    # box_xy = (box_xy - offset) * scale
    # box_wh *= scale
    # box_mins = box_xy - (box_wh / 2.)
    # box_maxes = box_xy + (box_wh / 2.)
    # _boxes = K.concatenate([
    #     box_mins[..., 0:1],  # x_min
    #     box_mins[..., 1:2],  # y_min
    #     box_maxes[..., 0:1],  # x_max
    #     box_maxes[..., 1:2]  # y_max
    # ])
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    box_yx = (box_yx - offset) * scale
    box_hw *= scale
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    _boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    _boxes *= K.concatenate([K.cast(image_shape, K.dtype(feats)), K.cast(image_shape, K.dtype(feats))])
    _boxes = K.reshape(_boxes, [-1, 4])

    _box_scores = box_confidence * box_class_probs
    _box_scores = K.reshape(_box_scores, [-1, num_classes])
    boxes.append(_boxes)
    box_scores.append(_box_scores)
boxes = K.concatenate(boxes, axis=0)
box_scores = K.concatenate(box_scores, axis=0)

mask = box_scores >= 0.3
max_num_boxes = K.constant(20, dtype='int32')

boxes_ = []
scores_ = []
classes_ = []
for c in range(num_classes):
    class_boxes = tf.boolean_mask(boxes, mask[:, c])
    class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
    nms_index = tf.image.non_max_suppression(
        class_boxes, class_box_scores, max_num_boxes, iou_threshold=0.5)
    class_boxes = K.gather(class_boxes, nms_index)
    class_box_scores = K.gather(class_box_scores, nms_index)
    classes = K.ones_like(class_box_scores, 'int32') * c
    boxes_.append(class_boxes)
    scores_.append(class_box_scores)
    classes_.append(classes)
boxes_ = K.concatenate(boxes_, axis=0)
scores_ = K.concatenate(scores_, axis=0)
classes_ = K.concatenate(classes_, axis=0)

sess = K.get_session()
b, s, c = sess.run([boxes_, scores_, classes_], feed_dict={model.input: image_data,
                                                           K.learning_phase(): 0})

# plt.imshow(img)
# for t in b:
#     xmin = b[1]
#     ymin = b[0]
#     xmax = b[3]
#     ymax = b[2]
#     plt.plot([xmin,ymin],[xmin,ymax])
#     plt.plot([xmin, yman], [xmin, ymax])
#     plt.plot([xmin, ymin], [xmin, ymax])
#     plt.plot([xmin, ymin], [xmin, ymax])
#
# plt.show()

img = img[:, :, ::-1]
for obj in b:
    cv2.rectangle(img, (obj[1], obj[0]), (obj[3], obj[2]), (0, 255, 0), 1)
# im_sized = im_sized[:, :, ::-1]  # RGB image
cv2.imwrite("C:/Users/john/Desktop/1.jpg", img)

exit()
