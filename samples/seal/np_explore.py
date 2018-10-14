import numpy as np
import skimage


def test_np_empty():
    print(np.empty((2, 3, 0)))
    print(np.empty((2, 3, 0)).shape)


# test_np_empty()


def test_xx():
    image = [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]], [[130, 140, 150], [160, 170, 180]]]
    print(skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255)
test_xx()
