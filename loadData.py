import numpy as np
import os
import scipy.misc as misc


train_path = 'C:\\Users\\nikhil\\Documents\\UltrasoundNerve\\train\\train'
test_path = 'C:\\Users\\nikhil\\Documents\\UltrasoundNerve\\test\\test'
def create_train(path):
    X = []
    Y = []
    for f in os.listdir(path):
        if 'mask' in f:
            continue
        mask_path = path + '\\' + f.split('.')[0] + '_mask.tif'
        image_path = path + '\\' + f
        img = misc.imread(image_path, flatten=True)
        img = misc.imresize(img, (64, 80))
        img_mask = misc.imread(mask_path, flatten=True)
        img_mask = misc.imresize(img_mask, (64, 80))
        image_mask = np.array([img_mask])
        image = np.array([img])
        X.append(image)
        Y.append(image_mask)
    X = np.array(X)
    Y = np.array(Y)
    np.save('train.npy', X)
    np.save('mask_train.npy', Y)

def load_train():
    X = np.load('train.npy')
    Y = np.load('mask_train.npy')
    return X, Y

def create_test(path):
    X = []
    Y = []
    for f in os.listdir(path):
        image_path = path + '\\' + f
        img = misc.imread(image_path, flatten=True)
        img = misc.imresize(img, (64, 80))
        image = np.array([img])
        X.append(image)
        Y.append(int(f.split('.')[0]))
    X = np.array(X)
    Y = np.array(Y)
    np.save('test.npy', X)
    np.save('id_test.npy', Y)

def load_test():
    X = np.load('test.npy')
    Y = np.load('id_test.npy')
    return X, Y

if __name__ == '__main__':
    create_train(train_path)
    create_test(test_path)