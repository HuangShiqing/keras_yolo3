import tensorflow as tf
from keras.layers import *
from keras import backend as K
from keras.regularizers import l2

Gb_all_layers = []
Gb_out_index = []

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


def conv2d_unit(prev_layer, filters, kernels=3, strides=1, bn=True, activation='linear'):
    input_shape = K.int_shape(prev_layer)
    weights_shape = (kernels, kernels, input_shape[-1], filters)
    weights_size = np.product(weights_shape)
    darknet_w_shape = (filters, weights_shape[2], kernels, kernels)

    conv_bias = np.ndarray(
        shape=(filters,),  # (32,),
        dtype='float32',
        buffer=weights_file.read(filters * 4))
    if bn:
        bn_weights = np.ndarray(
            shape=(3, filters),
            dtype='float32',
            buffer=weights_file.read(filters * 12))
        bn_weight_list = [
            bn_weights[0],  # scale gamma
            conv_bias,  # shift beta
            bn_weights[1],  # running mean
            bn_weights[2]  # running var
        ]
    conv_weights = np.ndarray(
        shape=darknet_w_shape,
        dtype='float32',
        buffer=weights_file.read(weights_size * 4))
    conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
    conv_weights = [conv_weights] if bn else [
        conv_weights, conv_bias
    ]

    padding = 'same' if strides == 1 else 'valid'
    if strides > 1:
        # Darknet uses left and top padding instead of 'same' mode
        prev_layer = ZeroPadding2D(((1, 0), (1, 0)))(prev_layer)
    conv_layer = (Conv2D(
        filters, (kernels, kernels),
        strides=(strides, strides),
        kernel_regularizer=l2(5e-4),
        use_bias=not bn,
        weights=conv_weights,
        activation=None,
        padding=padding))(prev_layer)
    if bn:
        conv_layer = (BatchNormalization(
            weights=bn_weight_list))(conv_layer)
    prev_layer = conv_layer
    if activation == 'linear':
        Gb_all_layers.append(prev_layer)
    elif activation == 'leaky':
        act_layer = LeakyReLU(alpha=0.1)(prev_layer)
        prev_layer = act_layer
        Gb_all_layers.append(act_layer)

    out_shape = K.int_shape(prev_layer)
    print(
        '   {:3} conv     {:4}  {} x {} / {}   {:3} x {:3} x {:4}   ->   {:3} x {:3} x {:4}'.format(
            len(Gb_all_layers) - 1,
            filters,
            kernels, kernels,
            strides,
            input_shape[1],
            input_shape[2],
            input_shape[3],
            out_shape[1],
            out_shape[2],
            out_shape[3]))
    return prev_layer


def shortcut(prev_layer, res=0):
    input_shape = K.int_shape(prev_layer)
    prev_layer = Add()([Gb_all_layers[res], prev_layer])
    Gb_all_layers.append(prev_layer)
    out_shape = K.int_shape(prev_layer)
    print('   {:3} res   {:2}                   {:3} x {:3} x {:4}   ->   {:3} x {:3} x {:4}'.format(
        len(Gb_all_layers) - 1, res,
        input_shape[1],
        input_shape[2],
        input_shape[3],
        out_shape[1],
        out_shape[2],
        out_shape[3]))

    return Gb_all_layers[-1]


def detection(num):
    Gb_out_index.append(len(Gb_all_layers) - 1)
    Gb_all_layers.append(None)
    # prev_layer = Gb_all_layers[-1]
    # return prev_layer
    print(
        '   {:3} detection'.format(num))


def route(ids):
    layers = [Gb_all_layers[i] for i in ids]
    if len(layers) > 1:
        # print('Concatenating route layers:', layers)
        concatenate_layer = Concatenate()(layers)
        Gb_all_layers.append(concatenate_layer)
        prev_layer = concatenate_layer
    else:
        skip_layer = layers[0]  # only one layer to route
        Gb_all_layers.append(skip_layer)
        prev_layer = skip_layer
    print('   {:3} route    {}'.format(len(Gb_all_layers) - 1, ids))

    return prev_layer


def upsample(prev_layer):
    input_shape = K.int_shape(prev_layer)
    Gb_all_layers.append(UpSampling2D(2)(prev_layer))
    prev_layer = Gb_all_layers[-1]
    out_shape = K.int_shape(prev_layer)

    print(
        '   {:3} upsample           {:4}x   {:3} x {:3} x {:4}   ->   {:3} x {:3} x {:4}'.format(len(Gb_all_layers) - 1,
                                                                                                 2,
                                                                                                 input_shape[1],
                                                                                                 input_shape[2],
                                                                                                 input_shape[3],
                                                                                                 out_shape[1],
                                                                                                 out_shape[2],
                                                                                                 out_shape[3]))
    return prev_layer
