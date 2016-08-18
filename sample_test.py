import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


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


im = misc.imread("C:\\Users\\nikhil\\Documents\\UltrasoundNerve\\train\\train\\47_3.tif", flatten=True)
im = misc.imresize(im, (64, 80))
im_mask = misc.imread("C:\\Users\\nikhil\\Documents\\UltrasoundNerve\\train\\train\\47_3_mask.tif", flatten=True)
im_mask = misc.imresize(im_mask, (64, 80))
temp = np.zeros(im.shape)
for h in range(im.shape[0]):
    for w in range(im.shape[1]):
        if im_mask[h][w] > 0:
            temp[h][w] = im[h][w]
im1 = 255-im
plt.imshow(im1, cmap='gray')
plt.show()
plt.imshow(im, cmap='gray')
plt.show()
im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)

im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.1, im_merge.shape[1] * 0.)

# Split image and mask
im_t = im_merge_t[..., 0]
im_mask_t = im_merge_t[..., 1]

# Display result
plt.figure(figsize=(16, 14))
plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t, im_mask_t]], cmap='gray')
plt.show()
