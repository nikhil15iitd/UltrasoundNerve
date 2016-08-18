

import numpy as np
import os
import scipy.misc as misc
import matplotlib.pyplot as plt

train_path = 'C:\\Users\\nikhil\\Documents\\UltrasoundNerve\\train\\train'
def create_train(path):
    X = []
    Y = []
    for f in os.listdir(path):
        if 'mask' in f:
            continue
        mask_path = path + '\\' + f.split('.')[0] + '_mask.tif'
        image_path = path + '\\' + f
        img = misc.imread(image_path, flatten=True)
        img = misc.imresize(img, (112, 144))
        img_mask = misc.imread(mask_path, flatten=True)
        img_mask = misc.imresize(img_mask, (112, 144))
        image_mask = np.array([img_mask])
        image = np.array([img])
        X.append(image)
        Y.append(image_mask)
    X = np.array(X)
    Y = np.array(Y)
    n = X.shape[0]
    mislabelled = 0
    for i in range(n):
        mask1 = np.sum(Y[i])
        for j in range(i+1, n, 1):
            mask2 = np.sum(Y[j])
            if (mask1 == 0 and mask2 > 0) or (mask1 > 0 and mask2 == 0):
                diff = np.abs(X[i][0] - X[j][0])
                if diff.sum() < 50000:
                    print(str(i) + ' ' + str(j))
                    mislabelled += 1
                    if mask1 == 0:
                        Y[i] = Y[j]
                    elif mask2 == 0:
                        Y[j] = Y[i]
                    '''
                    plt.figure(figsize=(16, 14))
                    plt.imshow(np.c_[np.r_[X[i][0], Y[i][0]], np.r_[X[j][0], Y[j][0]]], cmap='gray')
                    plt.show()
                    '''

    print(str(mislabelled)+' '+str(Y.shape))
    np.save('train112.npy', X)
    np.save('mask_train112.npy', Y)

create_train(train_path)

