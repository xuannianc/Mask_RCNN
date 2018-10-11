import tensorflow as tf
import numpy as np


def test_tf_where():
    """
    总结: tf.where(x) 返回一个二维数组, 第一维的长度为 x 中 True 的个数, 第二维的长度和 x 的维度相同, 如 x 是一个三维数组,
    那么 tf.where(x) 返回值的第二维的长度就是 3, 这 3 个数用于定位 x 中 True 的下标
    :return:
    """
    with tf.Session() as sess:
        # [[0]
        #  [2]]
        print(sess.run(tf.where(np.array([True, False, True, False]))))
        # [0 2]
        print(sess.run(tf.where(np.array([True, False, True, False]))[:, 0]))
        # [[0]
        #  [2]
        #  [4]]
        print(sess.run(tf.where(np.array([True, False, True, False, True]))))
        # [0 2 4]
        print(sess.run(tf.where(np.array([True, False, True, False, True]))[:, 0]))
        # [[0 0]
        #  [0 2]
        #  [1 1]
        #  [1 2]
        #  [1 3]]
        print(sess.run(tf.where(np.array([[True, False, True, False], [False, True, True, True]]))))
        # [0 0 1 1 1]
        print(sess.run(tf.where(np.array([[True, False, True, False], [False, True, True, True]]))[:, 0]))


# test_tf_where()

def test_tf_tile():
    """
    平铺:
    :return:
    """
    a = np.array([1, 2, 3])
    # 对 a 在 axis=0 上复制 4 次
    b = tf.tile(a, [4])
    c = np.array([[1, 2, 3], [100, 200, 300]])
    # 对 c 在 axis=0 上复制 2 次, 在 axis=1 上复制 3 次
    d = tf.tile(c, [2, 3])
    with tf.Session() as sess:
        # [1 2 3 1 2 3 1 2 3 1 2 3]
        print(sess.run(b))
        # [[  1   2   3   1   2   3   1   2   3]
        #  [100 200 300 100 200 300 100 200 300]
        #  [  1   2   3   1   2   3   1   2   3]
        #  [100 200 300 100 200 300 100 200 300]]
        print(sess.run(d))


# test_tf_tile()

def test_tf_expand_dims():
    a = np.arange(8).reshape(2, 4)
    b = tf.expand_dims(a, 1)
    with tf.Session() as sess:
        # [[[0 1 2 3]]
        #
        #  [[4 5 6 7]]]
        # shape 由 (2, 4) 变成 (2, 1, 4)
        print(sess.run(b))
        print(sess.run(tf.shape(b)))


# test_tf_expand_dims()

def test_tf_equal():
    a = np.array([[1, 2, 3, 2]])
    b = tf.equal(a, 2)
    c = tf.where(b)
    with tf.Session() as sess:
        print(sess.run(b))
        print(sess.run(c))
        print(sess.run(c[:, 0]))


# test_tf_equal()

def test_tf_nn_top():
    a = np.array([1, 7, 4, 5, 2, 6])
    b = tf.nn.top_k(a, k=a.shape[0])
    c = np.array([[1, 7, 4, 5, 2, 6], [3, 9, 10, 8, 12, 11]])
    d = tf.nn.top_k(c, k=3)
    with tf.Session() as sess:
        # [1 5 3 2 4 0]
        print(sess.run(b.indices))
        # [0 4 2 3 5 1]
        print(sess.run(b.indices[::-1]))
        print(sess.run(d.indices))
        print(sess.run(d.indices[::-1]))


# test_tf_nn_top()

def test_tf_gather():
    """
    indices ndims=2 的情况
    :return:
    """
    a = np.array([1, 1, 0])
    b = np.array([[0, 1, 0, 0, 0], [1, 1, 1, 0, 2]])

    with tf.Session() as sess:
        # [[1 1 1 1 1]
        #  [1 1 1 1 0]]
        print(sess.run(tf.gather(a, b)))


# test_tf_gather()

def test_tf_cast():
    with tf.Session() as sess:
        print(sess.run(tf.cast(tf.constant([]), tf.int64)))


test_tf_cast()
