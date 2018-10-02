import cv2
import os
import os.path as osp
import numpy as np
import skimage.io

DATASET_DIR = '/home/adam/Pictures/vat/'


def read_by_opencv(image_path):
    image = cv2.imread(image_path)
    B, G, R = cv2.split(image)
    if np.all(B == G) and np.all(B == R):
        print('read_by_opencv: gray')
    else:
        print('read_by_opencv: color')


def read_by_skimage(image_path):
    image = skimage.io.imread(image_path)
    # 比较矬的方法
    # if len(image.shape) == 2:
    #     print('read_by_skimage: gray')
    # elif len(image.shape) == 3:
    #     print('read_by_skimage: color')
    if image.ndim == 3:
        print('read_by_skimage: color')
    else:
        print('read_by_skimage: gray')


# a gray-scale image
image_path = osp.join(DATASET_DIR, '1100162350_12093275_20180427_299606.jpg')
read_by_opencv(image_path)
read_by_skimage(image_path)
# a colorful image
image_path = osp.join(DATASET_DIR, '1100172320_39236552_20170915_285217.jpg')
read_by_opencv(image_path)
read_by_skimage(image_path)
