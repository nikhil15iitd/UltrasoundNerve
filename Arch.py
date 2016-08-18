import numpy as np
import os
import theano
import theano.tensor as T
import lasagne
import time
import scipy.misc as misc
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from sklearn.cross_validation import KFold
import cv2
import matplotlib.pyplot as plt
from DataAug import ImageDataGenerator

img_rows = 64
img_cols = 80
image_rows = 420
image_cols = 580

smooth = 1.


# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def dice_coef(y_true, y_pred):
    y_true_f = T.flatten(y_true)
    y_pred_f = T.flatten(y_pred)
    intersection = T.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (T.sum(y_true_f) + T.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def load_train(load_real=True):
    if load_real == True:
        X = np.load('train.npy')
        Y = np.load('mask_train.npy')
        return X, Y
    else:
        X = np.load('train.npy')
        Y = np.load('mask_train.npy')
        Ynew = []
        for r in range(X.shape[0]):
            temp = np.zeros(X[r][0].shape)
            for h in range(X[r][0].shape[0]):
                for w in range(X[r][0].shape[1]):
                    if Y[r][0][h][w] > 0:
                        temp[h][w] = X[r][0][h][w]
            Ynew.append([temp])
        Ynew = np.array(Ynew)
        return X, Ynew


def load_test():
    X = np.load('test.npy')
    Y = np.load('id_test.npy')
    return X, Y


def iterate_minibatches(inputs, targets, batchsize, length, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, length - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def iterate_minibatches_train(inputs, targets, batchsize, start, length, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, length - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx + start:start_idx + start + batchsize]
        else:
            excerpt = slice(start_idx + start, start_idx + start + batchsize)
        yield inputs[excerpt], targets[excerpt]


def prep(img):
    img = img.astype('float32')
    img[img >= 0.7] = 1.
    img[img < 0.7] = 0.
    img = misc.imresize(img, (image_cols, image_rows))
    return img


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z + 1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s + 1, l + 1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def submission():
    imgs_test, imgs_id_test = load_test()
    imgs_test = np.load('mask_test.npy')
    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]

    total = imgs_test.shape[0]
    ids = []
    rles = []
    for i in range(total):
        img = imgs_test[i, 0]
        img = prep(img)
        rle = run_length_enc(img)
        rles.append(rle)
        ids.append(imgs_id_test[i])
        if i % 100 == 0:
            print('{}/{}'.format(i, total))

    first_row = 'img,pixels'
    file_name = 'submission.csv'

    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')


def mask_model(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 64, 80), input_var=input_var)

    conv1 = lasagne.layers.Conv2DLayer(network, num_filters=8, filter_size=(3, 3), pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv1 = lasagne.layers.BatchNormLayer(conv1)
    conv1 = lasagne.layers.NonlinearityLayer(conv1, nonlinearity=lasagne.nonlinearities.rectify)
    conv1 = lasagne.layers.Conv2DLayer(conv1, num_filters=16, filter_size=(3, 3), pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv1 = lasagne.layers.BatchNormLayer(conv1)
    conv1 = lasagne.layers.NonlinearityLayer(conv1, nonlinearity=lasagne.nonlinearities.rectify)
    pool1 = lasagne.layers.MaxPool2DLayer(conv1, pool_size=(2, 2))

    conv2 = lasagne.layers.Conv2DLayer(lasagne.layers.dropout(pool1, p=0.1), num_filters=16, filter_size=(3, 3),
                                       pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv2 = lasagne.layers.BatchNormLayer(conv2)
    conv2 = lasagne.layers.NonlinearityLayer(conv2, nonlinearity=lasagne.nonlinearities.rectify)
    conv2 = lasagne.layers.Conv2DLayer(conv2, num_filters=32, filter_size=(3, 3), pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv2 = lasagne.layers.BatchNormLayer(conv2)
    conv2 = lasagne.layers.NonlinearityLayer(conv2, nonlinearity=lasagne.nonlinearities.rectify)
    pool2 = lasagne.layers.MaxPool2DLayer(conv2, pool_size=(2, 2))

    conv3 = lasagne.layers.Conv2DLayer(lasagne.layers.dropout(pool2, p=0.1), num_filters=32, filter_size=(3, 3),
                                       pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv3 = lasagne.layers.BatchNormLayer(conv3)
    conv3 = lasagne.layers.NonlinearityLayer(conv3, nonlinearity=lasagne.nonlinearities.rectify)
    conv3 = lasagne.layers.Conv2DLayer(conv3, num_filters=64, filter_size=(3, 3), pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv3 = lasagne.layers.BatchNormLayer(conv3)
    conv3 = lasagne.layers.NonlinearityLayer(conv3, nonlinearity=lasagne.nonlinearities.rectify)
    pool3 = lasagne.layers.MaxPool2DLayer(conv3, pool_size=(2, 2))

    conv4 = lasagne.layers.Conv2DLayer(lasagne.layers.dropout(pool3, p=0.1), num_filters=64, filter_size=(3, 3),
                                       pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv4 = lasagne.layers.BatchNormLayer(conv4)
    conv4 = lasagne.layers.NonlinearityLayer(conv4, nonlinearity=lasagne.nonlinearities.rectify)
    conv4 = lasagne.layers.Conv2DLayer(conv4, num_filters=128, filter_size=(3, 3), pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv4 = lasagne.layers.BatchNormLayer(conv4)
    conv4 = lasagne.layers.NonlinearityLayer(conv4, nonlinearity=lasagne.nonlinearities.rectify)
    pool4 = lasagne.layers.MaxPool2DLayer(conv4, pool_size=(2, 2))

    conv5 = lasagne.layers.Conv2DLayer(pool4, num_filters=128, filter_size=(3, 3), pad='same',
                                       nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv5 = lasagne.layers.BatchNormLayer(conv5)
    conv5 = lasagne.layers.NonlinearityLayer(conv5, nonlinearity=lasagne.nonlinearities.rectify)
    conv5 = lasagne.layers.Conv2DLayer(conv5, num_filters=256, filter_size=(3, 3), pad='same',
                                       nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv5 = lasagne.layers.BatchNormLayer(conv5)
    conv5 = lasagne.layers.NonlinearityLayer(conv5, nonlinearity=lasagne.nonlinearities.rectify)
    conv5 = lasagne.layers.Conv2DLayer(conv5, num_filters=512, filter_size=(3, 3), pad='same',
                                       nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv5 = lasagne.layers.BatchNormLayer(conv5)
    conv5 = lasagne.layers.NonlinearityLayer(conv5, nonlinearity=lasagne.nonlinearities.rectify)
    conv5 = lasagne.layers.FeaturePoolLayer(conv5, pool_size=4)

    network = lasagne.layers.flatten(conv5)
    network = lasagne.layers.dropout(network, p=0.15)
    network = lasagne.layers.DenseLayer(network, num_units=2, nonlinearity=lasagne.nonlinearities.softmax)

    return network


def build_model(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 64, 80), input_var=input_var)

    conv1 = lasagne.layers.Conv2DLayer(network, num_filters=16, filter_size=(3, 3), pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv1 = lasagne.layers.BatchNormLayer(conv1)
    conv1 = lasagne.layers.NonlinearityLayer(conv1, nonlinearity=lasagne.nonlinearities.rectify)
    conv1 = lasagne.layers.Conv2DLayer(conv1, num_filters=16, filter_size=(3, 3), pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv1 = lasagne.layers.BatchNormLayer(conv1)
    conv1 = lasagne.layers.NonlinearityLayer(conv1, nonlinearity=lasagne.nonlinearities.rectify)
    pool1 = lasagne.layers.MaxPool2DLayer(conv1, pool_size=(2, 2))

    conv2 = lasagne.layers.Conv2DLayer(pool1, num_filters=32, filter_size=(3, 3),
                                       pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv2 = lasagne.layers.BatchNormLayer(conv2)
    conv2 = lasagne.layers.NonlinearityLayer(conv2, nonlinearity=lasagne.nonlinearities.rectify)
    conv2 = lasagne.layers.Conv2DLayer(conv2, num_filters=32, filter_size=(3, 3),
                                       pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv2 = lasagne.layers.BatchNormLayer(conv2)
    conv2 = lasagne.layers.NonlinearityLayer(conv2, nonlinearity=lasagne.nonlinearities.rectify)
    pool2 = lasagne.layers.MaxPool2DLayer(conv2, pool_size=(2, 2))

    conv3 = lasagne.layers.Conv2DLayer(pool2, num_filters=64, filter_size=(3, 3),
                                       pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv3 = lasagne.layers.BatchNormLayer(conv3)
    conv3 = lasagne.layers.NonlinearityLayer(conv3, nonlinearity=lasagne.nonlinearities.rectify)
    conv3 = lasagne.layers.Conv2DLayer(conv3, num_filters=64, filter_size=(3, 3),
                                       pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv3 = lasagne.layers.BatchNormLayer(conv3)
    conv3 = lasagne.layers.NonlinearityLayer(conv3, nonlinearity=lasagne.nonlinearities.rectify)
    pool3 = lasagne.layers.MaxPool2DLayer(conv3, pool_size=(2, 2))

    conv4 = lasagne.layers.Conv2DLayer(pool3, num_filters=128, filter_size=(3, 3), pad='same',
                                       nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv4 = lasagne.layers.BatchNormLayer(conv4)
    conv4 = lasagne.layers.NonlinearityLayer(conv4, nonlinearity=lasagne.nonlinearities.rectify)
    conv4 = lasagne.layers.Conv2DLayer(conv4, num_filters=128, filter_size=(3, 3), pad='same',
                                       nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv4 = lasagne.layers.BatchNormLayer(conv4)
    conv4 = lasagne.layers.NonlinearityLayer(conv4, nonlinearity=lasagne.nonlinearities.rectify)
    pool4 = lasagne.layers.MaxPool2DLayer(conv4, pool_size=(2, 2))

    conv = lasagne.layers.Conv2DLayer(pool4, num_filters=256, filter_size=(3, 3), pad='same',
                                       nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv = lasagne.layers.BatchNormLayer(conv)
    conv = lasagne.layers.NonlinearityLayer(conv, nonlinearity=lasagne.nonlinearities.rectify)
    conv = lasagne.layers.Conv2DLayer(conv, num_filters=512, filter_size=(3, 3), pad='same',
                                       nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv = lasagne.layers.BatchNormLayer(conv)
    conv = lasagne.layers.NonlinearityLayer(conv, nonlinearity=lasagne.nonlinearities.rectify)

    up4 = lasagne.layers.ElemwiseSumLayer(
        [lasagne.layers.TransposedConv2DLayer(conv, num_filters=128, filter_size=(2, 2),
                                              nonlinearity=lasagne.nonlinearities.rectify, stride=(2, 2),
                                              W=lasagne.init.HeNormal()), conv4])
    conv5 = lasagne.layers.Conv2DLayer(up4, num_filters=128, filter_size=(3, 3),
                                       pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv5 = lasagne.layers.BatchNormLayer(conv5)
    conv5 = lasagne.layers.NonlinearityLayer(conv5, nonlinearity=lasagne.nonlinearities.rectify)
    conv5 = lasagne.layers.Conv2DLayer(conv5, num_filters=128, filter_size=(3, 3),
                                       pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv5 = lasagne.layers.BatchNormLayer(conv5)
    conv5 = lasagne.layers.NonlinearityLayer(conv5, nonlinearity=lasagne.nonlinearities.rectify)

    up5 = lasagne.layers.ElemwiseSumLayer(
        [lasagne.layers.TransposedConv2DLayer(conv5, num_filters=64, filter_size=(2, 2),
                                              nonlinearity=lasagne.nonlinearities.rectify, stride=(2, 2),
                                              W=lasagne.init.HeNormal()), conv3])
    conv5 = lasagne.layers.Conv2DLayer(up5, num_filters=64, filter_size=(3, 3),
                                       pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv5 = lasagne.layers.BatchNormLayer(conv5)
    conv5 = lasagne.layers.NonlinearityLayer(conv5, nonlinearity=lasagne.nonlinearities.rectify)
    conv5 = lasagne.layers.Conv2DLayer(conv5, num_filters=64, filter_size=(3, 3),
                                       pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv5 = lasagne.layers.BatchNormLayer(conv5)
    conv5 = lasagne.layers.NonlinearityLayer(conv5, nonlinearity=lasagne.nonlinearities.rectify)

    up6 = lasagne.layers.ElemwiseSumLayer(
        [lasagne.layers.TransposedConv2DLayer(conv5, num_filters=32, filter_size=(2, 2),
                                              nonlinearity=lasagne.nonlinearities.rectify, stride=(2, 2),
                                              W=lasagne.init.HeNormal()), conv2])
    conv6 = lasagne.layers.Conv2DLayer(up6, num_filters=32, filter_size=(3, 3),
                                       pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv6 = lasagne.layers.BatchNormLayer(conv6)
    conv6 = lasagne.layers.NonlinearityLayer(conv6, nonlinearity=lasagne.nonlinearities.rectify)
    conv6 = lasagne.layers.Conv2DLayer(conv6, num_filters=32, filter_size=(3, 3),
                                       pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv6 = lasagne.layers.BatchNormLayer(conv6)
    conv6 = lasagne.layers.NonlinearityLayer(conv6, nonlinearity=lasagne.nonlinearities.rectify)

    up7 = lasagne.layers.ElemwiseSumLayer(
        [lasagne.layers.TransposedConv2DLayer(conv6, num_filters=16, filter_size=(2, 2),
                                              nonlinearity=lasagne.nonlinearities.rectify, stride=(2, 2),
                                              W=lasagne.init.HeNormal()), conv1])
    conv7 = lasagne.layers.Conv2DLayer(up7, num_filters=16, filter_size=(3, 3),
                                       pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv7 = lasagne.layers.BatchNormLayer(conv7)
    conv7 = lasagne.layers.NonlinearityLayer(conv7, nonlinearity=lasagne.nonlinearities.rectify)
    conv7 = lasagne.layers.Conv2DLayer(conv7, num_filters=16, filter_size=(3, 3), pad='same', nonlinearity=None,
                                       W=lasagne.init.HeNormal())
    conv7 = lasagne.layers.BatchNormLayer(conv7)
    conv7 = lasagne.layers.NonlinearityLayer(conv7, nonlinearity=lasagne.nonlinearities.rectify)

    # softmax layer
    out = lasagne.layers.Conv2DLayer(conv7, num_filters=1, filter_size=(1, 1),
                                     nonlinearity=lasagne.nonlinearities.sigmoid)
    return out


def train(num_epochs=100):
    datagen = ImageDataGenerator(
        rotation_range=8,
        shear_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')
    mask_var = T.ivector('targets')
    network = build_model(input_var)
    network_mask = mask_model(input_var)

    prediction_mask = lasagne.layers.get_output(network_mask)
    loss_mask = lasagne.objectives.categorical_crossentropy(prediction_mask, mask_var)
    loss_mask = loss_mask.mean()
    params_mask = lasagne.layers.get_all_params(network_mask, trainable=True)
    updates_mask = lasagne.updates.nesterov_momentum(loss_mask, params_mask, learning_rate=1e-2)
    train_mask = theano.function([input_var, mask_var], loss_mask, updates=updates_mask)

    test_prediction_mask = lasagne.layers.get_output(network_mask, deterministic=True)
    test_loss_mask = lasagne.objectives.categorical_crossentropy(test_prediction_mask, mask_var)
    test_loss_mask = test_loss_mask.mean()
    val_mask = theano.function([input_var, mask_var], test_loss_mask)
    predict_mask = theano.function([input_var], test_prediction_mask)

    # load:
    if os.path.isfile("network.npz"):
        f = np.load("network.npz")
        params = [f["param%d" % i] for i in range(len(f.files))]
        f.close()
        lasagne.layers.set_all_param_values(network, params)
    # load network mask:
    if os.path.isfile("network_mask.npz"):
        f = np.load("network_mask.npz")
        params_mask = [f["param%d" % i] for i in range(len(f.files))]
        f.close()
        lasagne.layers.set_all_param_values(network_mask, params_mask)

    prediction = lasagne.layers.get_output(network)
    loss = dice_coef_loss(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.1)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = dice_coef_loss(test_prediction, target_var)
    test_loss = test_loss.mean()
    val_fn = theano.function([input_var, target_var], test_loss)
    predict_fn = theano.function([input_var], test_prediction)

    X, Y = load_train(load_real=True)
    mean = 99.5130537031
    std = 49.980055585

    Y_label = np.zeros(Y.shape[0], dtype=np.uint8)
    for i in range(Y.shape[0]):
        cumsum = np.sum(Y[i])
        if cumsum > 2:
            Y_label[i] = 1
        else:
            Y_label[i] = 0

    print(mean)
    print(std)

    # train network to detect if mask exists or not
    for epoch in range(0):
        np.random.seed(epoch)
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in datagen.flow(X, Y, batch_size=32, shuffle=True):
            inputs, targets = batch
            inputs = inputs.astype(np.float32)
            masks = np.zeros(targets.shape[0], dtype=np.uint8)
            for i in range(inputs.shape[0]):
                cumsum = np.sum(targets[i])
                if cumsum > 0:
                    masks[i] = 1
                else:
                    masks[i] = 0
            inputs = (inputs - mean) / std
            train_err += train_mask(inputs, masks)
            train_batches += 1
            if train_batches > 160:
                break
        Xvalidate = X[5000:]
        Yvalidate = Y_label[5000:]
        valid_err = 0
        valid_batches = 0
        for batch in iterate_minibatches(Xvalidate, Yvalidate, 128, Xvalidate.shape[0], shuffle=False):
            inputs, targets = batch
            inputs = inputs.astype(np.float32)
            inputs = (inputs - mean) / std
            valid_err += val_mask(inputs, targets)
            valid_batches += 1
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  Training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  Validation loss: " + str(valid_err / valid_batches))

    values = lasagne.layers.get_all_param_values(network_mask)
    np.savez("network_mask.npz", **{"param%d" % i: param for i, param in enumerate(values)})
    # now train main network
    for epoch in range(num_epochs):
        np.random.seed(epoch+457)
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in datagen.flow(X, Y, batch_size=32, shuffle=True):
            inputs, targets = batch
            inputs = inputs.astype(np.float32)
            targets = targets.astype(np.float32)
            for i in range(inputs.shape[0]):
                if np.random.randint(0, high=2) > 2:
                    im = inputs[i][0]
                    im_mask = targets[i][0]
                    im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)
                    im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.1,
                                                   im_merge.shape[1] * 0.)
                    # Split image and mask
                    im_t = im_merge_t[..., 0]
                    im_mask_t = im_merge_t[..., 1]
                    inputs[i] = np.array([im_t])
                    targets[i] = np.array([im_mask_t])
            inputs = (inputs - mean) / std
            targets /= 255.
            train_err += train_fn(inputs, targets)
            train_batches += 1
            if train_batches > 160:
                break
        Xvalidate = X[:600]
        Yvalidate = Y[:600]
        valid_err = 0
        valid_batches = 0
        for batch in iterate_minibatches(Xvalidate, Yvalidate, 128, Xvalidate.shape[0], shuffle=False):
            inputs, targets = batch
            inputs = (inputs.astype(np.float32) - mean) / std
            targets = (targets.astype(np.float32)) / 255.
            valid_err += val_fn(inputs, targets)
            valid_batches += 1
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  Training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  Validation loss: " + str(valid_err / valid_batches))

    values = lasagne.layers.get_all_param_values(network)
    np.savez("network.npz", **{"param%d" % i: param for i, param in enumerate(values)})


    Xtest, Yids = load_test()
    Xmask = []
    for i in range(Xtest.shape[0]):
        Xinput = (Xtest[i].astype(np.float32)-mean)/std
        mask_val = predict_mask([Xinput])[0][1]
        img = predict_fn([Xinput])[0]
        Xmask.append(mask_val*img)
    Xmask = np.array(Xmask)
    print(str((Xmask[5].shape)))
    np.save('mask_test.npy', Xmask)
    submission()
    plt.imshow(Xmask[0, 0], cmap='gray')
    plt.show()



def train_model(model_number, num_epochs, Xtrain, Ytrain, Xtest, Ytest):
    datagen = ImageDataGenerator(
        rotation_range=8,
        shear_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')
    mask_var = T.ivector('targets')
    network = build_model(input_var)
    network_mask = mask_model(input_var)

    prediction_mask = lasagne.layers.get_output(network_mask)
    loss_mask = lasagne.objectives.categorical_crossentropy(prediction_mask, mask_var)
    loss_mask = loss_mask.mean()
    params_mask = lasagne.layers.get_all_params(network_mask, trainable=True)
    updates_mask = lasagne.updates.nesterov_momentum(loss_mask, params_mask, learning_rate=1e-2)
    train_mask = theano.function([input_var, mask_var], loss_mask, updates=updates_mask)

    test_prediction_mask = lasagne.layers.get_output(network_mask, deterministic=True)
    test_loss_mask = lasagne.objectives.categorical_crossentropy(test_prediction_mask, mask_var)
    test_loss_mask = test_loss_mask.mean()
    val_mask = theano.function([input_var, mask_var], test_loss_mask)
    predict_mask = theano.function([input_var], test_prediction_mask)

    prediction = lasagne.layers.get_output(network)
    loss = dice_coef_loss(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.1)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = dice_coef_loss(test_prediction, target_var)
    test_loss = test_loss.mean()
    val_fn = theano.function([input_var, target_var], test_loss)

    # load main & mask networks:
    networkArch = "network" + str(model_number) + ".npz"
    networkMaskArch = "network_mask" + str(model_number) + ".npz"
    if os.path.isfile(networkArch):
        f = np.load(networkArch)
        params = [f["param%d" % i] for i in range(len(f.files))]
        f.close()
        lasagne.layers.set_all_param_values(network, params)
    # load network mask:
    if os.path.isfile(networkMaskArch):
        f = np.load(networkMaskArch)
        params_mask = [f["param%d" % i] for i in range(len(f.files))]
        f.close()
        lasagne.layers.set_all_param_values(network_mask, params_mask)

    mean = np.mean(Xtrain)
    std = np.std(Xtrain)

    # train network to detect if mask exists or not
    for epoch in range(0):
        np.random.seed(epoch)
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in datagen.flow(Xtrain, Ytrain, batch_size=32, shuffle=True):
            inputs, targets = batch
            inputs = inputs.astype(np.float32)
            masks = np.zeros(targets.shape[0], dtype=np.uint8)
            for i in range(inputs.shape[0]):
                cumsum = np.sum(targets[i])
                if cumsum > 0:
                    masks[i] = 1
                else:
                    masks[i] = 0
                if np.random.randint(0, high=10) > 4 and epoch > 5:
                    im = inputs[i][0]
                    im_mask = targets[i][0]
                    im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)
                    coin = np.random.randint(0, high=2)
                    if coin == 1:
                        im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 3, im_merge.shape[1] * 0.07,
                                                       im_merge.shape[1] * 0.09)
                    else:
                        im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08,
                                                       im_merge.shape[1] * 0.08)
                    # Split image and mask
                    im_t = im_merge_t[..., 0]
                    im_mask_t = im_merge_t[..., 1]
                    inputs[i] = np.array([im_t])
                    targets[i] = np.array([im_mask_t])
            inputs = (inputs - mean) / std
            train_err += train_mask(inputs, masks)
            train_batches += 1
            if train_batches > 180:
                break
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  Training loss:\t\t{:.6f}".format(train_err / train_batches))
        Xvalidate = Xtest
        Yvalidate = Ytest
        valid_err = 0
        valid_batches = 0
        for batch in iterate_minibatches(Xvalidate, Yvalidate, 128, Xvalidate.shape[0], shuffle=False):
            inputs, targets = batch
            inputs = inputs.astype(np.float32)
            masks = np.zeros(targets.shape[0], dtype=np.uint8)
            for i in range(inputs.shape[0]):
                cumsum = np.sum(targets[i])
                if cumsum > 0:
                    masks[i] = 1
                else:
                    masks[i] = 0
            inputs = (inputs - mean) / std
            valid_err += val_mask(inputs, masks)
            valid_batches += 1
        print("  Validation loss: " + str(valid_err / valid_batches))

    values = lasagne.layers.get_all_param_values(network_mask)
    np.savez(networkMaskArch, **{"param%d" % i: param for i, param in enumerate(values)})

    # now train main network
    for epoch in range(num_epochs):
        np.random.seed(epoch)
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in datagen.flow(Xtrain, Ytrain, 32, shuffle=True):
            inputs, targets = batch
            inputs = inputs.astype(np.float32)
            targets = targets.astype(np.float32)
            for i in range(inputs.shape[0]):
                if np.random.randint(0, high=2) > 0:
                    im = inputs[i][0]
                    im_mask = targets[i][0]
                    im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)
                    im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.1,
                                                   im_merge.shape[1] * 0.)
                    # Split image and mask
                    im_t = im_merge_t[..., 0]
                    im_mask_t = im_merge_t[..., 1]
                    inputs[i] = np.array([im_t])
                    targets[i] = np.array([im_mask_t])
            inputs = (inputs - mean) / std
            targets /= 255.
            train_err += train_fn(inputs, targets)
            train_batches += 1
            if train_batches > 160:
                break
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  Training loss:\t\t{:.6f}".format(train_err / train_batches))
        Xvalidate = Xtest
        Yvalidate = Ytest
        valid_err = 0
        valid_batches = 0
        for batch in iterate_minibatches(Xvalidate, Yvalidate, 128, Xvalidate.shape[0], shuffle=False):
            inputs, targets = batch
            inputs = (inputs.astype(np.float32) - mean) / std
            targets = (targets.astype(np.float32)) / 255.
            valid_err += val_fn(inputs, targets)
            valid_batches += 1
        print("  Validation loss: " + str(valid_err / valid_batches))

    values = lasagne.layers.get_all_param_values(network)
    np.savez(networkArch, **{"param%d" % i: param for i, param in enumerate(values)})


def ensemble(num_epochs, folds):
    X, y = load_train(load_real=True)
    n = X.shape[0]
    kf = KFold(n, n_folds=folds)
    model_number = 0
    for train_index, test_index in kf:
        model_number += 1
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        print(str(model_number) + ' train length: ' + str(Xtrain.shape[0]))
        train_model(model_number, num_epochs, Xtrain, ytrain, Xtest, ytest)


def vote(folds):
    input_var = T.tensor4('inputs')
    network_mask = mask_model(input_var)
    test_prediction_mask = lasagne.layers.get_output(network_mask, deterministic=True)
    predict_mask = theano.function([input_var], test_prediction_mask)
    # load network mask:
    if os.path.isfile("network_mask.npz"):
        f = np.load("network_mask.npz")
        params_mask = [f["param%d" % i] for i in range(len(f.files))]
        f.close()
        lasagne.layers.set_all_param_values(network_mask, params_mask)
    networks = []
    params = []
    test_prediction = []
    predict_arr = []
    for i in range(folds):
        networks.append(build_model(input_var))
        params.append(lasagne.layers.get_all_params(networks[i], trainable=True))
        networkArch = "network" + str(i + 1) + ".npz"
        f = np.load(networkArch)
        params[i] = [f["param%d" % j] for j in range(len(f.files))]
        f.close()
        lasagne.layers.set_all_param_values(networks[i], params[i])
        test_prediction.append(lasagne.layers.get_output(networks[i], deterministic=True))
        predict_arr.append(theano.function([input_var], test_prediction[i]))

    Xtest, Yids = load_test()
    mean = Xtest.mean()
    std = Xtest.std()
    Xmask = []
    for i in range(Xtest.shape[0]):
        Xinput = (Xtest[i].astype(np.float32) - mean) / std
        imgs = np.zeros(Xinput[0].shape)
        for models in range(folds):
            imgs = imgs + (predict_arr[models]([Xinput])[0][0])
        imgs /= folds
        mask_val = predict_mask([Xinput])[0][1]
        imgs *= mask_val
        Xmask.append([imgs])
    Xmask = np.array(Xmask)
    print(str((Xmask.shape)))
    np.save('mask_test.npy', Xmask)
    submission()
    plt.imshow(Xmask[0, 0], cmap='gray')
    plt.show()


# ensemble(30, 8)
# vote(8)
train(20)
