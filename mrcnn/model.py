"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import keras.utils as KU

from mrcnn import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion

assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    # math.ceil 用于向上取整, math.floor 用于向下取整
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
          int(math.ceil(image_shape[1] / stride))]
         for stride in config.BACKBONE_STRIDES])


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                                           '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101, 传递的是 config.BACKBONE='resnet101'
        stage5: Boolean. If False, stage5 of the network is not created, 传递的是 True
        train_bn: Boolean. Train or freeze Batch Norm layers, 传递的是 config.TRAIN_BN=False, 表示 freeze, 适合较小的数据集
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    此函数的作用我猜是对已经修改过的 bbox 进行再修正, 以确保坐标都在 normalized coordinate 里面
    normalized coordinates 左上角的坐标为 (0,0), 右下角的坐标为 (1,1)
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form wy1, wx1, wy2, wx2
           (y1,x1) __________
                  |          |
                  |     _____|_____
                  |     |clip|    |
                  |_____|____|    |
                        |         |
                        |_________|(wy2,wx2)
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KE.Layer):
    """Receives anchor scores and selects a subset to pass as proposals to the second stage.
    Filtering is done based on anchor scores and non-max suppression to remove overlaps.
    It also applies bounding box refinement deltas to anchors.

    Inputs:
        rpn_probs: (batch_size, num_anchors, (bg prob, fg prob))
                   前面是 bg prob , 后面是 fg prob 这很重要
        rpn_bbox: (batch_size, num_anchors, (dy, dx, log(dh), log(dw)))
        anchors: (batch_size, num_anchors, (y1, x1, y2, x2)) anchors in normalized coordinates
                 在 self.get_anchors() 调整成 normalized coordinates 坐标

    Returns:
        Proposals in normalized coordinates (batch_size, rois, (y1, x1, y2, x2))
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        # config.POST_NMS_ROIS_TRAINING=2000 或 config.POST_NMS_ROIS_INFERENCE=1000
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence.
        # scores 的 shape 是 (batch_size, num_anchors)
        scores = inputs[0][:, :, 1]
        # Box Deltas
        # deltas 的 shape 是 (batch, num_anchors, 4)
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        # anchors 的 shape 是 (batch, num_anchors, 4)
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        # 获取 config.PRE_NMS_LIMIT=6000 和 tf.shape(anchors)[1](其实是 num_anchors=261888) 的较小值
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        # 从 scores 中获取最大的 pre_nms_limit 个数的下标
        # shape 为 (batch_size, config.PRE_NMS_LIMIT=6000)
        # tf.nn.top_k 返回 https://www.tensorflow.org/api_docs/python/tf/nn/top_k
        # scores 是二维的, 那么是对最后一维进行排序, 返回的 ix 维数也是 2, 参考 test_tf_nn_top
        # ix 第二维的第一个数表示最大数的下标, 第二个数表示第二大的数的下标, 以此类推
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        # tf.gather 的第二个参数作为第一个参数的下标,获取该下标的元素
        # config.IMAGES_PER_GPU 作为 batch_size
        # shape 为 (batch_size, config.PRE_NMS_LIMIT=6000)
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        # shape 为 (batch_size, config.PRE_NMS_LIMIT=6000, 4)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        # shape 为 (batch_size, config.PRE_NMS_LIMIT=6000, 4)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                            self.config.IMAGES_PER_GPU,
                                            names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # (batch_size, config.PRE_NMS_LIMIT=6000, (y1, x1, y2, x2))
        refined_anchors = utils.batch_slice([pre_nms_anchors, deltas],
                                            lambda x, y: apply_box_deltas_graph(x, y),
                                            self.config.IMAGES_PER_GPU,
                                            names=["refined_anchors"])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. (batch_size, config.PRE_NMS_LIMIT, (y1, x1, y2, x2))
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        refined_clipped_anchors = utils.batch_slice(refined_anchors,
                                                    lambda x: clip_boxes_graph(x, window),
                                                    self.config.IMAGES_PER_GPU,
                                                    names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy for small objects, so we're skipping it.

        # Non-max suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            # 如果得到的 proposals 的个数小于 self.proposal_count,用 0 来填充
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            # tf.pad 的第二个参数的 shape 为 (n_dims, 2), 每一个数表示的填充 0 的长度
            # 如第一个元素的两个数分别表示在第一维的开始处不填充, 在第一维度末尾填充长度为 padding 的 0
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposals = utils.batch_slice([refined_clipped_anchors, scores], nms, self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    # 换底公式
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.
    FIXME: 主要是对 batch_size > 1 而引起的混乱进行重新排列
    希望的结果是每个 batch_item 独立, 它的 rois 来自各个层的 feature_map
    而事实上使用 crop_and_resize 进行 pooling 是获取的整个 batch 在同一个 feature_map 上的 roi

    Params:
    - pool_shape: (pool_size, pool_size) of the output pooled regions. Usually [7, 7]

    Inputs:
    - rois: (batch_size, num_rois, (y1, x1, y2, x2)) in normalized coordinates.
             Possibly padded with zeros if not enough boxes to fill the array.
    - image_meta: (batch_size, (meta data)) Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is (batch_size, height, width, channels)

    Output:
    Pooled regions in the shape: (batch_size, num_rois, pool_size, pool_size, channels).
    The width and height are those specific in the pool_shape in the layer constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes (batch_size, num_rois, (y1, x1, y2, x2)) in normalized coords
        rois = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        # (batch_size, image_meta_size=14)
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the feature pyramid.
        # Each is (batch_size, height, width, channels=256)
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        # tf.split 参考 https://www.tensorflow.org/api_docs/python/tf/split
        # y1, x1, y2, x2 的 shape 为 (batch_size, num_rois, 1)
        y1, x1, y2, x2 = tf.split(rois, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper.
        # Account for the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        # 这里和论文中唯一的不同就是 224.0/tf.sqrt(image_area) 我想这是因为 rois 都被 normalized 了
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        # 这里的 4 是论文中的 k0
        # roi_level 的 shape 为 (batch_size, num_rois, 1)
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        # roi_level 的 shape 为 (batch_size, num_rois)
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        roi_to_level = []
        for i, level in enumerate(range(2, 6)):
            # ix 是一个二维数组
            # 当 roi_level 是一个二维数组, ix 第二维的长度为 2, shape 为 (num_level_rois_of_batch, 2)
            ix = tf.where(tf.equal(roi_level, level))
            # tf.gather_nd 参考 https://www.tensorflow.org/api_docs/python/tf/manip/gather_nd
            # ix 第二维的第一个元素表示 batch_item id, 第二个元素表示 roi_id, 组合起来就能获取 roi
            # shape 为 (num_level_rois_of_batch, 4)
            level_rois = tf.gather_nd(rois, ix)

            # ROI indices for crop_and_resize.
            # roi_indices 就是该 level 所有的 roi 对应的 batch_item_id, 用于后面的 crop_and_resize
            # roi_indices 的 shape 是 (num_level_rois_of_batch,)
            roi_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            roi_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            # UNCLEAR: tf.stop_gradient 是做什么?
            level_rois = tf.stop_gradient(level_rois)
            roi_indices = tf.stop_gradient(roi_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations,
            # so that we can evaluate either max or average pooling.
            # In fact, interpolating only a single value at each bin center (without pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # FIXME: pooled 的 shape 为 (num_level_rois_of_batch, pool_height, pool_width, channels)?
            # NOTE: crop_and_resize 的第二个参数 boxes 就是要 normalized cordinates 下的值
            # 参见 https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
            pooled.append(
                tf.image.crop_and_resize(feature_maps[i], level_rois, roi_indices, self.pool_shape, method="bilinear"))

        # Pack pooled features into one tensor
        # FIXME: pooled 的 shape 为 (batch_size * num_rois, pool_height, pool_width, channels)?
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another column representing the order of pooled boxes
        # 各个 level 的 roi 的下标
        # shape 为 (batch_size * num_rois, 2)
        roi_to_level = tf.concat(roi_to_level, axis=0)
        # tf.range 参见 https://www.tensorflow.org/api_docs/python/tf/range
        # shape 为 (batch_size * num_rois, 1)
        roi_range = tf.expand_dims(tf.range(tf.shape(roi_to_level)[0]), 1)
        # shape 为 (batch_size * num_rois, 3) 第二维增加了一列表示 roi 在 roi_to_level 中的序号
        roi_to_level = tf.concat([tf.cast(roi_to_level, tf.int32), roi_range], axis=1)

        # Rearrange pooled features to match the order of the original rois
        # Sort roi_to_level by batch then roi index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        # batch_item_id * 10000 + roi_id(原来的 rois 中的序号)
        sorting_tensor = roi_to_level[:, 0] * 100000 + roi_to_level[:, 1]
        # ix 表示对 sorting_tensor 进行从小到大排序, 每个元素在原来序列中的序号组成的列表
        # 假设 sorting_tensor 是 [1, 7, 4, 5, 2, 6], 排好序应该是 [1,2,4,5,6,7], 每个元素在原来序列中的序号分别是[0 4 2 3 5 1]
        # 那么 ix=[0 4 2 3 5 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(roi_to_level)[0]).indices[::-1]
        # roi_to_level[:, 2] 表示的是现在 roi 的序号, 参数 ix 表示的对 sorting_tensor 排序的 ix
        # 假设参数 ix 的第 0 个数的值为 8, 那么表示该元素是 roi_to_level 第 8 个元素
        # FIXME: 这一步是多余的?
        ix = tf.gather(roi_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(rois)[:2], tf.shape(pooled)[1:]], axis=0)
        # shape 为 (batch_size, num_rois, pool_height, pool_width, channels)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        # input_shape 是所有 input 的 shape 组成的 list
        # 所以 input_shape 第 0 个元素是 rois 的 shape 为 (batch_size, num_rois=200, 4)
        # input_shape 第 2 个元素是 feature_maps 的 shape[-1] 表示 channels
        # 综上, 返回的 shape 是 (batch_size, num_rois, pool_height, pool_width, channels]
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)


############################################################
#  Detection Target Layer
############################################################

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    # 假设 boxes1 的 shape 为 (num_proposals, 4), boxes2 的 shape 为 (num_gt_bboxes, 4)
    # expand_dims 后变成 (num_proposals, 1, 4), 我怎么感觉 expand_dims 这一步可以省啊
    # tile 之后变成 (num_proposals, 1, 4 * num_gt_bbox)
    # reshape 之后变成 (num_proposals * num_gt_bboxes, 4)
    # 每个 proposal 重复了 num_gt_bboxes 次
    # 如 [[1,2,3,4],[5,6,7,8],[9,10,11,12]] 变成 [[1,2,3,4,1,2,3,4],[5,6,7,8,5,6,7,8],[9,10,11,12,9,10,11,12]]
    # 再变成 [[1,2,3,4],[1,2,3,4],[5,6,7,8],[5,6,7,8],[9,10,11,12],[9,10,11,12]]
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    # b2 的 shape (num_gt_bboxes * num_proposals, 4), 每个 gt_bbox 重复了 num_proposals 次
    # 如 [[11,12,13,14],[15,16,17,18]] 变成
    # [[11,12,13,14],[15,16,17,18],[11,12,13,14],[15,16,17,18],[11,12,13,14],[15,16,17,18]]
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    # iou 的 shape 是 (num_proposals * num_gt_bboxes,), 0-(num_gt_bboxes-1) 个元素表示第 0 个 proposal 和所有 gt_bbox 的 iou
    # reshape 之后会变成 axis=1 的第 0 个元素
    # 同理, num_gt_bboxes - (2 * num_gt_bboxes - 1) 的元素表示第 1 个 proposal 和所有 gt_bbox 的 iou
    # reshape 之后会变成 axis=1 的第 1 个元素
    # 如 iou 为 [0.1,0.2,0.3,0.4,0.5,0.6] reshape 成 shape (2,3) 就是 [[0.1,0.2],[0.3,0.4],[0.5,0.6]]
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:

    proposals: (config.POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)) in normalized coordinates.
               Might be zero padded if there are not enough proposals.
    gt_class_ids: (config.MAX_GT_INSTANCES] int class IDs
    gt_boxes: (config.MAX_GT_INSTANCES, (y1, x1, y2, x2)) in normalized coordinates.
    gt_masks: (height, width, config.MAX_GT_INSTANCES) of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts and masks.

    rois: (config.TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)) in normalized coordinates
    class_ids: (config.TRAIN_ROIS_PER_IMAGE). Integer class IDs. Zero padded.
    deltas: (config.TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)))
    masks: (config.TRAIN_ROIS_PER_IMAGE, height, width).
           Masks cropped to bbox boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    # 参考 https://www.tensorflow.org/api_docs/python/tf/Assert
    # 第一个参数 condition,第二个参数 data. 如果不满足 condition,打印 data.
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], name="roi_assertion"),
    ]
    # ???
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")
    # tf.where 返回一个二维数组, 第一维维度表示 non_zeros 中 True 的个数, 第二维的每一个元素表示 non_zeros 中 True 的下标
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2, name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude them from training.
    # A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    crowd_masks = tf.gather(gt_masks, crowd_ix, axis=2)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    # FIXME: 表示非 crowd box?
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative proposals
    proposals_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive proposals are those with >= 0.5 IoU with a GT box
    positive_proposals_bool = (proposals_iou_max >= 0.5)
    positive_indices = tf.where(positive_proposals_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(proposals_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    # 注意: 这种切片索引方式 a[:b], 如果 a 的长度小于 b, 不会报错, 而是返回完整的 a
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    # 重新确认 positive proposals 的个数
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_proposals = tf.gather(proposals, positive_indices)
    negative_proposals = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    # unclear: 因为前面已经有了 non-max_supression
    # unclear: roi_gt_box_assignment 里面的元素是否会重复,是否存在多个 proposals 和同一个 gt_bbox 有最大 iou 的可能
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_proposals, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive proposals to GT masks
    # Permute masks to [N, height, width, 1]
    # expand_dims 之后可以认为 transposed_masks 变成了 N 个 shape 为 (height,width,1) 的 image 组成的 batch
    # N 为 gt_masks 经过 "去 padding", "去 crowd" 的数量
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each positive proposals
    # 相当于从 batch 中选择一部分的 image
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_proposals
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_proposals, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    # crop_and_resize 可以完成 RoiPooling 的操作, 参考 https://blog.csdn.net/qq_14839543/article/details/80019951
    # https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    # 第一个参数 image, 表示一个 batch 的 image
    # 第二个参数 boxes, 表示 proposals 的坐标
    # 第三个参数 box_ind, 表示 boxes 所在的 image 位于 batch 中的下标, 其长度和 boxes 一致
    # 第四个参数 crop_size, 表示 crop 和 resize 之后的大小
    # masks 的 shape 为 (N, mask_height, mask_width, 1)
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes, box_ids, config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    # tf.squeeze 参考 https://www.tensorflow.org/api_docs/python/tf/squeeze
    # 就是删除 dim_size=1 的 dim, 可以通过 axis 限制删除的 dim 的范围
    # masks 的 shape 变为 (N, mask_height, mask_width)
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with binary cross entropy loss.
    # tf.round 参考 https://www.tensorflow.org/api_docs/python/tf/round
    # round 小数到最近的整数, x.5 到最近的偶数, 如 1.5 变成 2.0, 2.5 也变成 2.0
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_proposals, negative_proposals], axis=0)
    N = tf.shape(negative_proposals)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    # 需要注意的是 masks 并没有 transpose 过来, 第一维仍表示的是 mask 的个数
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: (batch_size, N, (y1, x1, y2, x2)) in normalized coordinates.
               Might be zero padded if there are not enough proposals.
               N=proposal_count=config.POST_NMS_ROIS_TRAINING or config.POST_NMS_ROIS_INFERENCE
    gt_class_ids: (batch_size, config.MAX_GT_INSTANCES) Integer class IDs.
    gt_boxes: (batch_size, config.MAX_GT_INSTANCES, (y1, x1, y2, x2)) in normalized coordinates.
    gt_masks: (batch_size, height, width, config.MAX_GT_INSTANCES) of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: (batch_size, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)) in normalized coordinates
    target_class_ids: (batch, TRAIN_ROIS_PER_IMAGE). Integer class IDs.
    target_deltas: (batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))
    target_mask: (batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0], self.config.MASK_SHAPE[1])  # masks
        ]

    # 作甚?
    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


############################################################
#  Detection Layer
############################################################

def refine_detections_graph(rpn_rois, mrcnn_class, mrcnn_deltas, window, config):
    """Refine classified proposals and filter overlaps and return final detections.

    Args:
        rpn_rois: (num_rpn_rois, (y1, x1, y2, x2)) in normalized coordinates
        mrcnn_class: (num_rpn_rois, num_classes). Class probabilities.
        mrcnn_deltas: (num_rpn_rois, num_classes, (dy, dx, log(dh), log(dw))).
                      Class-specific bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates.
                The part of the image that contains the image excluding the padding.

    Returns:
        detections: (num_detections, (y1, x1, y2, x2, class_id, score)) where
                    coordinates are normalized.
    """
    # Class IDs per ROI
    # shape 为 (num_rpn_rois, )
    class_ids = tf.argmax(mrcnn_class, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    # shape 为 (num_rpn_rois, 2) 第二维的第一个元素是 rpn_roi_id, 第二个元素是 class_id
    indices = tf.stack([tf.range(mrcnn_class.shape[0]), class_ids], axis=1)
    # shape 为 (num_rpn_rois, ) 每个 rois 最大的 class score
    class_scores = tf.gather_nd(mrcnn_class, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(mrcnn_deltas, indices)
    # Apply bounding box deltas
    # Shape: (num_rpn_rois, (y1, x1, y2, x2)) in normalized coordinates
    refined_rois = apply_box_deltas_graph(rpn_rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        # 参见 test_tf_sets_set_intersection
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    # tf.unique 参考 https://www.tensorflow.org/api_docs/python/tf/unique
    # 返回两个数组, 第一个数组为 unique 的值, 第二个数组为原来数组的数值在 unique 数组中的 idxs
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        class_ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep_ixs = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, class_ixs),
            tf.gather(pre_nms_scores, class_ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep_ixs = tf.gather(keep, tf.gather(class_ixs, class_keep_ixs))
        # Pad with -1 so returned tensors have the same shape
        # NOTE: config.DETECTION_MAX_INSTANCES >= tf.shape(class_keep_ixs)[0]
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep_ixs)[0]
        class_keep_ixs = tf.pad(class_keep_ixs, [(0, gap)], mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep_ixs.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep_ixs

    # 2. Map over class IDs
    # tf.map_fn 参考 https://www.tensorflow.org/api_docs/python/tf/map_fn
    # 就是把 unique_pre_nms_class_ids 的第 0 维的元素分别传递 nms_keep_map,进行调用
    # 这里 nms_keep_map 返回的是一个 tensor, 那么 tf.map_fn 会把所有 tensor stack 起来
    # shape 为 (num_unique_pre_nms_class_ids, config.DETECTION_MAX_INSTANCES)
    # 每个 unique class id 关联的 rois 的下标, 下标个数不足 config.DETECTION_MAX_INSTANCES, 用 -1 代替
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    # shape 为 (num_keep,)
    keep = tf.gather(keep, top_ids)

    # Arrange output as (num_keep, (y1, x1, y2, x2, class_id, score))
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Inputs:
        rpn_rois: shape 为 (batch_size, num_rpn_rois=1000, (y1, x1, y2, x2), ProposalLayer 生成
        mrcnn_class: shape 为 (batch_size, num_rpn_rois=1000, num_classes) fpn_classifier_graph 生成
        mrcnn_deltas: shape 为 (batch_size, num_rpn_rois=1000, num_classes, (y1, x1, y2, x2))
                      fpn_classifier_graph 生成
        image_meta: shape 为 (batch_size, 12 + num_classes)

    Returns:
    (batch_size, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rpn_rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_deltas = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates.
        # Windows are the area in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = parse_image_meta_graph(image_meta)
        # 0 表示第一个 batch_item
        image_shape = m['image_shape'][0]
        # normalized_window
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rpn_rois, mrcnn_class, mrcnn_deltas, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU)

        # Reshape output
        # (batch_size, num_detections, (y1, x1, y2, x2, class_id, class_score)) in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        # None 表示 batch_size
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


############################################################
#  Region Proposal Network (RPN)
############################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: (batch_size, H * W * anchors_per_location, 2) Anchor classifier logits (before softmax)
        rpn_probs: (batch_size, H * W * anchors_per_location, 2) Anchor classifier probabilities.
        rpn_bbox: (batch_size, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))) Deltas to be applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map is not even.
    # Shared convolutional base of the RPN
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu', strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. (batch_size, height, width, anchors_per_location * 2).
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid', activation='linear', name='rpn_class_raw')(shared)

    # Reshape to (batch_size, num_anchors, 2)
    # height * width * anchors_per_location == num_anchors
    rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    # 只见过 softmax 作为 Dense 层的 Activation,没见过这种放在 Lambda 之后的用法
    rpn_probs = KL.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors_per_location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(4 * anchors_per_location, (1, 1), padding="valid", activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, num_anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared weights.

    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
                   传递的是 config.RPN_ANCHOR_STRIDE=1
    anchors_per_location: number of anchors per pixel in the feature map, 传递的是 len(config.RPN_ANCHOR_RATIOS)
    depth: Depth of the backbone feature map. 传递的是 config.TOP_DOWN_PYRAMID_SIZE=256

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, depth], name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    """Builds the computation graph of the feature pyramid network classifier and regressor heads.

    rois: (batch_size, num_rois=200|1000, (y1, x1, y2, x2)) Proposal boxes in normalized coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: (batch_size, (meta data)) Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        mrcnn_class_logits: (batch_size, num_rois=200|1000, NUM_CLASSES) classifier logits (before softmax)
        mrcnn_class: (batch_size, num_rois=200|1000, NUM_CLASSES) classifier probabilities
        mrcnn_deltas: (batch_size, num_rois=200|1000, NUM_CLASSES, (dy, dx, log(dh), log(dw)))
                     Deltas to apply to proposal boxes
    """
    # ROI Pooling
    # Shape: (batch_size, num_rois=200|1000, pool_height=7, pool_width=7, channels=256]
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)), name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # squeeze 前 x 的 shape 是 (batch_size,num_rois, 1, 1, 1024)
    # squeeze 后 x 的 shape 是 (batch_size,num_rois, 1024) 但是为什么不这样 K.squeeze(x, axis=[2,3])?
    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="pool_squeeze")(x)

    # Classifier head
    # shape 为 (batch_size, num_rois, NUM_CLASSES)
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes), name='mrcnn_class_logits')(shared)
    mrcnn_class = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")(mrcnn_class_logits)

    # Delta head
    # (batch_size, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw)))
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'), name='mrcnn_bbox_fc')(shared)
    # Reshape to (batch_size, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw)))
    s = K.int_shape(x)
    # 注意 Reshape 的参数是不包含 batch_size 的, 参见 https://keras.io/layers/core/#reshape
    mrcnn_deltas = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_class, mrcnn_deltas


def fpn_mask_graph(rois, feature_maps, image_meta,
                   pool_size, num_classes, train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: (batch_size, num_rois, (y1, x1, y2, x2)) Proposal boxes in normalized coordinates.
    feature_maps: List of feature maps from different layers of the pyramid [P2, P3, P4, P5].
                  Each has a different resolution.
    image_meta: (batch_size, (meta data)) Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
               config.MASK_POOL_SIZE
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks (batch_size, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES)
    """
    # ROI Pooling
    # Shape: (batch_size, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels)
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # Conv2DTranspose 前 x 的 shape 是 (batch_size,num_rois=200,14,14,256)
    # Conv2DTranspose 后 x 的 shape 是 (batch_size,num_rois=200,28,28,256)
    # Conv2DTranspose 参见 https://keras.io/layers/convolutional/#conv2dtranspose
    # 假设 Conv2DTranspose 的输入 shape 为 (batch_size,rows,cols,filters)
    # 那么 Conv2DTranspose 的输出 shape 为 (batch_size,new_rows,new_cols,filters)
    # new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] + output_padding[0])
    # new_rows = ((14 - 1) * 2 + 2 - 2 * 0 + 0) = 28
    # new_cols 同理
    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    # x 的 shape 变为 (batch_size,num_rois=200,28,28,num_classes=2)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask")(x)
    return x


############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.

    Return:
        loss: shape 为 (N, 4)
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: (batch_size, num_anchors, 1). Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
    rpn_class_logits: (batch_size, num_anchors, 2). RPN classifier logits for FG/BG.
    """
    # Squeeze last dim to simplify
    # shape 为 (batch_size, num_anchors)
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    # 把 -1 转成 0, (K.equal 把 -1 转成了 False,K.cast 把 False 转成了 0)
    # shape 为 (batch_size, num_anchors)
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    # rpn_match 中原来非 0 的元素 indices
    # shape 为 (num_non_zeros, 2), 第二维的第一个下标表示 batch_item_id, 第二个下标表示 anchor_id
    nonzero_indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    # 获取原来值为 -1,1 部分的 rpn_class_logits
    # shape 为 (num_nonzeros, 2)
    rpn_class_logits = tf.gather_nd(rpn_class_logits, nonzero_indices)
    # 去除掉原来值为 0 的部分,由于前面的 K.cast, 原来的值 -1 变成了现在的 0, 原来的值 1 还是现在的 1
    # shape 为 (num_nonzeros,)
    anchor_class = tf.gather_nd(anchor_class, nonzero_indices)
    # Cross entropy loss
    # sparse_categorical_crossentropy 参见 https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
    # https://github.com/keras-team/keras/issues/7749
    # FIXME: loss 的 shape 为 (num_nonzeros,)
    loss = K.sparse_categorical_crossentropy(target=anchor_class, output=rpn_class_logits, from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, input_rpn_deltas, input_rpn_match, rpn_deltas):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    input_rpn_deltas: (batch_size, num_train_anchors=256, (dy, dx, log(dh), log(dw))).
                      Uses 0 padding to fill in unsed bbox deltas.
    input_rpn_match: (batch_size, num_anchors, 1).
                     Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
    rpn_deltas: (batch_size, num_anchors, (dy, dx, log(dh), log(dw))).
                rpn_model 生成的
    """
    # Positive anchors contribute to the loss,
    # but negative and neutral anchors (match value of 0 or -1) don't.
    # shape 为 (batch_size, num_anchors)
    input_rpn_match = K.squeeze(input_rpn_match, -1)
    # shape 为 (num_pos_rpn_anchors, 2), 第二维的第一个下标表示 batch_item_id, 第二个下标表示 anchor_id
    positive_rpn_anchors_indices = tf.where(K.equal(input_rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    # shape 为 (num_pos_rpn_anchors, 4)
    rpn_deltas = tf.gather_nd(rpn_deltas, positive_rpn_anchors_indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    # shape 为 (batch_size,) 表示每个元素为 batch_item 的 positive rpn anchors 的数量
    batch_counts = K.sum(K.cast(K.equal(input_rpn_match, 1), tf.int32), axis=1)
    # batch_pack_graph 对 input_rpn_deltas 的每个 batch_item 的 256 个元素中
    # 分别提取最前面的 batch_counts[batch_item_id] 个元素
    # 此时 input_rpn_deltas 的 shape 就和 rpn_deltas 的 shape 相同了
    # shape 为 (num_pos_rpn_anchors, 4)
    input_rpn_deltas = batch_pack_graph(input_rpn_deltas, batch_counts, config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(input_rpn_deltas, rpn_deltas)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, mrcnn_class_logits, active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: (batch_size, num_rois). Integer class IDs. Uses zero padding to fill in the array.
    mrcnn_class_logits: (batch_size, num_rois, num_classes)
    active_class_ids: (batch_size, num_classes). Has a value of 1 for classes that are in the dataset of the image,
                      and 0 for classes that are not in the dataset.
    """
    # During model building, Keras calls this function with target_class_ids of type float32. Unclear why.
    # Cast it to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # mrcnn_class_logits 第 2 维的最大值的 idx 作为预测的 idx
    # shape 是 (batch_size, num_rois)
    mrcnn_class_ids = tf.argmax(mrcnn_class_logits, axis=2)
    # Find predictions of classes that are not in the dataset.
    # TODO: Update this line to work with batch > 1.
    # Right now it assumes all images in a batch have the same active_class_ids
    # active_class_idx[0] 表示第一个 batch_item 的 active_class_idx
    # NOTE: active_class_idx[0] 的 shape (num_classes,), 值为 0 表示 inactive, 值为 1 表示 active
    # mrcnn_class_ids 维度为 2, 第二维度的值作为 active_class_idx[0] 的 idx, 参见 test_tf_gather
    # mrcnn_active 的 shape 为 (batch_size, num_rois=200) 第二维的值为 0 或 1 表示是否 active
    mrcnn_active = tf.gather(active_class_ids[0], mrcnn_class_ids)

    # Loss
    # target_class_ids 的 shape 为 (batch_size, num_rois)
    # mcrnn_class_logits 的 shape 为 (batch_size, num_rois, 2)
    # FIXME: loss 的 shape 为 (batch_size, num_rois)?
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids, logits=mrcnn_class_logits)

    # Erase losses of predictions of classes that are not in the active classes of the image.
    loss = loss * mrcnn_active

    # Computer loss mean. Use only predictions that contribute to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mrcnn_active)
    return loss


def mrcnn_bbox_loss_graph(target_deltas, target_class_ids, mrcnn_deltas):
    """Loss for Mask R-CNN bounding box refinement.

    target_deltas: (batch_size, num_rois, (dy, dx, log(dh), log(dw)))
    target_class_ids: (batch_size, num_rois). Integer class IDs.
    mrcnn_deltas: (batch_size, num_rois, num_classes, (dy, dx, log(dh), log(dw)))
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    # shape 为 (batch_size * num_rois,)
    target_class_ids = K.reshape(target_class_ids, (-1,))
    # shape 为 (batch_size * num_rois, 4)
    target_deltas = K.reshape(target_deltas, (-1, 4))
    # shape 为 (batch_size * num_rois, num_classes, 4)
    mrcnn_deltas = K.reshape(mrcnn_deltas, (-1, K.int_shape(mrcnn_deltas)[2], 4))

    # Only positive ROIs contribute to the loss.
    # And only the right class_id of each ROI. Get their indices.
    # shape 为 (num_pos_rois,)
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    # shape 为 (num_pos_rois,)
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    # shape 为 (num_pos_rois,2) 第二维的第一个元素表示 pos_roi_ix, 第二维的第二个元素表示 pos_roi 对应的 class_id
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    # shape 为 (num_pos_rois, 4)
    target_deltas = tf.gather(target_deltas, positive_roi_ix)
    # tf.gather_nd 中 indices 的第一维的维度作为输出的第一维的维度, 第二维的值作为真正的下标从 target_bbox 中获取值
    # shape 为 (num_pos_rois, 4)
    mrcnn_deltas = tf.gather_nd(mrcnn_deltas, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_deltas) > 0, smooth_l1_loss(y_true=target_deltas, y_pred=mrcnn_deltas),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, mrcnn_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: (batch_size, num_rois, mask_height, mask_width).
                  A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: (batch_size, num_rois). Integer class IDs. Zero padded.
    mrcnn_masks: (batch_size, num_rois, mask_height, mask_width, num_classes) float32 tensor with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    # shape 为 (batch_size * num_rois,)
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_mask_shape = tf.shape(target_masks)
    # shape 为 (batch_size * num_rois, mask_height, mask_width)
    target_masks = K.reshape(target_masks, (-1, target_mask_shape[2], target_mask_shape[3]))
    mrcnn_masks_shape = tf.shape(mrcnn_masks)
    # shape 为 (batch_size * num_rois, mask_height, mask_width, num_classes)
    mrcnn_masks = K.reshape(mrcnn_masks, (-1, mrcnn_masks_shape[2], mrcnn_masks_shape[3], mrcnn_masks_shape[4]))
    # Permute predicted masks to (batch_size * num_rois, num_classes, mask_height, mask_width)
    mrcnn_masks = tf.transpose(mrcnn_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss.
    # And only the class specific mask of each ROI.
    # shape 为 (num_pos_rois,)
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    # shape 为 (num_pos_rois,)
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    # shape 为 (num_pos_rois, 2) 第二维的第一个元素表示 pos_roi_ix, 第二维的第二个元素表示 pos_roi 对应的 class_id
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    # shape 为 (num_pos_rois, mask_height, mask_width)
    y_true = tf.gather(target_masks, positive_roi_ix)
    # shape 为 (num_pos_rois, mask_height, mask_width)
    y_pred = tf.gather_nd(mrcnn_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: (batch_size, num_rois, num_classes)
    # UNCLEAR: 为什么是这个 shape?
    loss = K.switch(tf.size(y_true) > 0, K.binary_crossentropy(target=y_true, output=y_pred), tf.constant(0.0))
    loss = K.mean(loss)
    return loss


############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    Args:
        augment: (deprecated. Use augmentation instead). If true, apply random
                 image augmentation. Currently, only horizontal flipping is offered.
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
                      For example, passing imgaug.augmenters.Fliplr(0.5) flips images
                      right/left 50% of the time.
        use_mini_mask: If False, returns full-size masks that are the same height
                       and width as the original image. These can be big, for example
                       1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
                       224x224 and are generated by extracting the bounding box of the
                       object and resizing it to MINI_MASK_SHAPE.

    Returns:
        image: (height, width, 3), resized image or augmented image
        shape: the original shape of the image before resizing and cropping.
        class_ids: (num_instances,) Integer class IDs of the instances
        bbox: (num_instances, (y1, x1, y2, x2))
        mask: (height, width, num_instances). The height and width are those
              of the image unless use_mini_mask is True, in which case they are
              defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            # fliplr 数组左右翻转,深度方向的每一层都是左右翻转
            # flipud 上下翻转
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # Some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    # 把全是 0 的 mask 过滤掉
    # _idx 的 shape 为 (num_instances, ) 里面的值为 True or False
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Some boxes might be all zeros if the corresponding mask got cropped out.
    # bbox: (num_instances, (y1, x1, y2, x2))
    bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    # 作者可能是把一个 source 当做一个 dataset
    # dataset.num_classes 是所有 source 的 class 的数量
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    # source 的所有 class 的 class id
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    # 只把相关 source 的 class id 设置为 1 表示 active
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape, window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
    """Generate targets for training Stage 2 classifier and mask heads.
    This is not used in normal training. It's useful for debugging or to train
    the Mask RCNN heads without using the RPN head.

    Inputs:
    rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
    gt_class_ids: [instance count] Integer class IDs
    gt_boxes: [instance count, (y1, x1, y2, x2)]
    gt_masks: [height, width, instance count] Ground truth masks. Can be full
              size or mini-masks.

    Returns:
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
            bbox refinements.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
           to bbox boundaries and resized to neural network output size.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(gt_masks.dtype)

    # It's common to add GT Boxes to ROIs but we don't do that here because
    # according to XinLei Chen's paper, it doesn't help.

    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]

    # Compute areas of ROIs and ground truth boxes.
    # shape: (num_random_rois)
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * (rpn_rois[:, 3] - rpn_rois[:, 1])
    # shape: (num_gt_bboxes)
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute overlaps [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # Assign ROIs to GT boxes
    # 每一个 roi 对应的最大的 iou 的 gt_bbox 的 index
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    # 每一个 roi 与所有的 gt_bboxes 的最大 iou
    rpn_roi_iou_max = overlaps[np.arange(overlaps.shape[0]), rpn_roi_iou_argmax]
    # GT box assigned to each ROI
    # 每一个 roi 对应的最大 iou 的 gt_bbox.这里要注意 rpn_roi_iou_argmax 的 shape 是 (num_rpn_rois,)
    # 而 seal sample 中 gt_bboxes 的 shape 是 (2,4)
    # 得到的 rpn_roi_gt_bboxes 的 shape 是 (num_rpn_rois,4), 这种我还是第一次见
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    # 每一个 roi 对应的最大 iou 的 gt_bbox 的 class_id,获取的方式和 gt_bbox 一致
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]
    # Positive ROIs are those with >= 0.5 IoU with a GT box.
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
    # TODO: To hard example mine or not to hard example mine, that's the question
    # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% foreground.
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Combine indices of ROIs to keep
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Need more? 就是说 iou==0.5 的 roi 比较多
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Looks like we don't have enough samples to maintain the desired
        # balance. Reduce requirements and fill in the rest. This is
        # likely different from the Mask RCNN paper.

        # There is a small chance we have neither fg nor bg samples.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Fill the rest with repeated bg rois.
            # 从 keep_bg_ids 中再选出 remaining 个,可重复
            keep_extra_ids = np.random.choice(keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
        "keep doesn't match ROI batch size {}, {}".format(
            keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_iou_argmax = rpn_roi_iou_argmax[keep]

    # Class-aware bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.NUM_CLASSES, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    # 第二维下标要么是一个 int, 要么是 shape 为 (num_pos_ids,) 的数组
    # box_refinement 返回的 shape 是 (num_pos_ids,)
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # Normalize bbox refinements
    bboxes /= config.BBOX_STD_DEV

    # Generate class-specific target masks
    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                     dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_iou_argmax[i]
        gt_mask = gt_masks[:, :, gt_id]

        if config.USE_MINI_MASK:
            # 如果使用 MINI_MASK 把 mask 放大到和原 image 一样大小
            # Create a mask placeholder, the size of the image
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            # GT box
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            # Resize mini mask to size of GT box
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = np.round(utils.resize(gt_mask, (gt_h, gt_w))).astype(bool)
            # Place the mini batch in the placeholder
            gt_mask = placeholder

        # Pick part of the mask and resize it
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        # 获取 gt_mask 中的 roi 部分
        m = gt_mask[y1:y2, x1:x2]
        mask = utils.resize(m, config.MASK_SHAPE)
        masks[i, :, :, class_id] = mask

    return rois, roi_gt_class_ids, bboxes, masks


def build_rpn_targets(anchors, gt_class_ids, gt_bboxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    Args:
        anchors: (num_anchors=261888, (y1, x1, y2, x2))
        gt_class_ids: (num_instances,) Integer class IDs.
        gt_bboxes: (num_instances, (y1, x1, y2, x2))

    Returns:
        rpn_match: (num_anchors=261888, ) (int32) matches between anchors and GT boxes.
                   1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_deltas: (num_train_anchors=256, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    # config.RPN_TRAIN_ANCHORS_PER_IMAGE 默认为 256
    rpn_deltas = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_bboxes = gt_bboxes[crowd_ix]
        gt_bboxes = gt_bboxes[non_crowd_ix]
        # Compute overlaps with crowd boxes
        # shape 为 (num_anchors, num_crowd_bboxes)
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_bboxes)
        # np.amax 参考 https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.amax.html
        # 取某一维度的最大值, 如果没有指定 axis 获取整个数组的最大值
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        # iou 小于 0.001 认为 anchor 和 crowd_bboxes 相交
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps
    # shape 为 (num_anchors, num_gt_bboxes)
    overlaps = utils.compute_overlaps(anchors, gt_bboxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens).
    # Instead, match it to the closest anchor (even if its max IoU is < 0.3).
    # UNCLEAR: 为什么非 positive 的 anchors 也要设置 deltas?
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    # 计算每一个 anchor 和其有最大 iou 的 gt_bboxes 的 idx, shape 是 (num_anchors,)
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    # NOTE: 如果 anchor_iou_argmax 是一个数, 那么 np.arange(overlaps.shape[0]) 是可以换成 :
    # 但是这里 anchor_iou_argmax 是一个 shape 为 (num_anchors,) 的数组
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    # 如果某个 anchor 和所有 gt_bboxes 的最大 iou 小于 0.3,那么认为该 anchor 为 negative
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    # 计算每一个 gt_bbox 和其有最大 iou 的 anchor 的 idx
    # 认为该 anchor 为 positive, 不管 iou 是否大于等于 0.7
    # shape 为 (num_instances, )
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    # 如果某个 anchor 和所有 gt_bboxes 的最大 iou 大于 0.7,那么认为该 anchor 为 positive
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    # 计算 positive anchors 的个数
    positive_ids = np.where(rpn_match == 1)[0]
    extra = len(positive_ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        # 随机挑选 extra 个数的 anchors 设置为 neutral
        ids = np.random.choice(positive_ids, extra, replace=False)
        rpn_match[ids] = 0
    # 计算 negative anchors 的个数
    # Same for negative proposals
    negative_ids = np.where(rpn_match == -1)[0]
    extra = len(negative_ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(negative_ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    # 因为 positive 已经经过平衡,所以要重新计算
    positive_ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(positive_ids, anchors[positive_ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_bboxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_deltas[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        # UNCLEAR: config.RPN_BBOX_STD_DEV 是什么意思?
        rpn_deltas[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_deltas


def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """Generates ROI proposals similar to what a region proposal network
    would generate.

    image_shape: [Height, Width, Depth]
    count: Number of ROIs to generate
    gt_class_ids: [N] Integer ground truth class IDs
    gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.

    Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        # 设置 roi 四个坐标点的位置范围
        min_y = max(gt_y1 - h, 0)
        max_y = min(gt_y2 + h, image_shape[0])
        min_x = max(gt_x1 - w, 0)
        max_x = min(gt_x2 + w, image_shape[1])

        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(min_y, max_y, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(min_x, max_x, (rois_per_box * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            # 获取 rois_per_box 个 y1,y2
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >= threshold][:rois_per_box]
            # 获取 rois_per_box 个 x1,x2
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >= threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then split into x1, x2, y1, y2
        # 先排序是 x1 <= x2,y1 <= y2, 然后拆分
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        # stack into y1,x1,y2,x2, 构建成 [y1,x1,y2,x2] 的形式
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                    threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                    threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois


def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
                   num_random_rois=0, batch_size=1, detection_targets=False,
                   no_augmentation_sources=None):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.
    no_augmentation_sources: Optional. List of sources to exclude for
        augmentation. A source is string that identifies a dataset and is
        defined in the Dataset class.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The contents
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            # 在获取第一个 image 的时候,打乱 image_ids 的顺序
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]

            # If the image source is not to be augmented pass None as augmentation
            if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    load_image_gt(dataset, config, image_id, augment=augment,
                                  augmentation=None, use_mini_mask=config.USE_MINI_MASK)
            else:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    load_image_gt(dataset, config, image_id, augment=augment,
                                  augmentation=augmentation, use_mini_mask=config.USE_MINI_MASK)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            # 如果 gt_class_ids 都为 0,相当于 np.zeros(num_instances)
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            # rpn_match 的 shape 是 (num_anchors,), 每一个元素为 -1,0,1.其中 -1 表示 negative,0 表示 neutral,1 表示 positive
            # rpn_bbox 的 shape 是 (num_train_anchors,4)
            # 默认 num_train_anchors=config.RPN_TRAIN_ANCHORS_PER_IMAGE=256
            # 前 num_positive_anchors 个元素为 [delta_h,delta_w,log(gt_h/a_h),log(gt_w/a_w)]
            # 后 num_train_anchors-num_positive_anchors 个元素为 [0,0,0,0]
            rpn_match, rpn_bbox = build_rpn_targets(anchors, gt_class_ids, gt_boxes, config)

            # Mask R-CNN Targets
            if num_random_rois:
                # rpn_rois 的 shape 是 (num_random_rois, 4)
                # num_random_rois * 90% 的 rois 是在 gt_bbox 的周围, 其余 10% 则是任意的.为什么要这么生成 rois?
                rpn_rois = generate_random_rois(image.shape, num_random_rois, gt_class_ids, gt_boxes)
                if detection_targets:
                    # rois 的 shape 是 (num_train_rois, 4)
                    # num_train_rois=config.TRAIN_ROIS_PER_IMAGE=200
                    # mrcnn_class_ids 的 shape 是 (num_train_rois,)
                    # mrcnn_bbox 的 shape 是 (num_train_rois, dataset.num_classes, 4)
                    # mrcnn_mask 的 shape 是 (num_train_rois, config.MASK_SHAPE[0]=28, config.MASK_SHAPE[1]=28, dataset.num_classes)
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = \
                        build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros((batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros([batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks = np.zeros((batch_size, gt_masks.shape[0], gt_masks.shape[1],
                                           config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                if num_random_rois:
                    batch_rpn_rois = np.zeros((batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros((batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros(
                            (batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros((batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros((batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            # mold_image 是减掉像素平均数
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            if num_random_rois:
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                outputs = []

                if num_random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        # Keras requires that output and targets have the same number of dimensions
                        # 所以为什么要 expand_dims 要看一下模型最后的输出
                        # batch_mrcnn_class_ids 原来 shape 是 (batch_size, num_train_rois)
                        # expand_dims 之后的 shape 是 (batch_size, num_train_rois, 1)
                        batch_mrcnn_class_ids = np.expand_dims(batch_mrcnn_class_ids, -1)
                        # extend 不同于 append 之处就在于同时添加多个元素
                        outputs.extend([batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        # 输入图片宽高不确定, 所以用 None
        input_image = KL.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_deltas = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            # 注意 K.shape 是算上 batch 这个维度的, 所以 [1:3] 表示的是 height, width
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None], name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True, train_bn=config.TRAIN_BN)
        else:
            _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE, stage5=True, train_bn=config.TRAIN_BN)
        # Top-down Layers
        # TODO: add assert to verify feature map sizes match what's in config
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by subsampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            # shape 是 (261888, 4)
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            # shape 变成 (1, 261888, 4)
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        # layer_outputs 的形式是 [[rpn_class_logits_p2,rpn_class_p2,rpn_bbox_p2],[xx_p3,yy_p3_zz_p3],...,[xx_p6...]]
        # NOTE: rpn_bbox 换成 rpn_deltas 更合适
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        # outputs 的形式是 [[rpn_class_logits_p2,xx_p3,xx_p4...],[rpn_class_p2,yy_p3,yy_p4...],[rpn_bbox_p2,...]]
        outputs = list(zip(*layer_outputs))
        # outputs 的形式是 [rpn_class_logits_all_layers, rpn_class_all_layers, rpn_bbox_all_layers]
        outputs = [KL.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_deltas = outputs

        # Generate proposals
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" else config.POST_NMS_ROIS_INFERENCE
        # rpn_rois 的 shape (batch_size, self.proposal_count=2000|1000, (y1, x1, y2, x2))
        # in normalized coordinates and zero padded.
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_deltas, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image came from.
            active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4], name="input_roi", dtype=np.int32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero padded.
            # Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_deltas, target_mask = \
                DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_deltas = \
                fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            mrcnn_mask = fpn_mask_graph(rois, mrcnn_feature_maps,
                                        input_image_meta,
                                        config.MASK_POOL_SIZE,
                                        config.NUM_CLASSES,
                                        train_bn=config.TRAIN_BN)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_deltas, input_rpn_match, rpn_deltas])
            mrcnn_class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            mrcnn_bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_deltas, target_class_ids, mrcnn_deltas])
            mrcnn_mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_deltas, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_deltas,
                       mrcnn_class_logits, mrcnn_class, mrcnn_deltas, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
            # model.summary()
            # KU.plot_model(model, to_file='logs/model.jpg', show_shapes=True)
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_deltas = \
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # Detections
            # output is (batch_size, num_detections, (y1, x1, y2, x2, class_id, score))
            # in normalized coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_deltas, input_image_meta])

            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                        input_image_meta,
                                        config.MASK_POOL_SIZE,
                                        config.NUM_CLASSES,
                                        train_bn=config.TRAIN_BN)

            model = KM.Model([input_image, input_image_meta, input_anchors],
                             [detections, mrcnn_class, mrcnn_deltas,
                              mrcnn_mask, rpn_rois, rpn_class, rpn_deltas],
                             name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/' \
                                 'releases/download/v0.2/' \
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss", "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        # UNCLEAR: 为什么 loss 都是 None, 而不是之前计算出来的 loss
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name, layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                # 文件名的中 epoch id 是从 1 开始的, 而 keras 代码中的 epoch id 则是从 0 开始的
                # -1 表示获取 keras 的 epoch id, +1 表示下一个 epoch id
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        # 注意 : 前表示的是参数的名字, : 后表示真正的 format template
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(self.config.NAME.lower()))
        # epoch:04d 作为 placeholder, keras 会自动填充. 不想被前一个 format 填充,所以分成了两行来对 checkpoint_path 赋值
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
	    custom_callbacks: Optional. Add custom callbacks to be called
	        with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources)
        val_generator = data_generator(val_dataset, self.config, shuffle=True, batch_size=self.config.BATCH_SIZE)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            # ???: 这几个参数的意思?
            keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        # UNCLEAR: 这是干啥?
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.

        Args:
            images: List of image matrices [(height,width,depth),...]. Images can have different sizes.

        Returns:
            3 Numpy matrices:
            molded_images: (batch_size=num_images, h, w, 3). Images resized and normalized.
            image_metas: (batch_size=num_images, length_of_meta_data=14). Details about each image.
            windows: (batch_size=num_images, (y1, x1, y2, x2)).
                     The portion of the image that has the original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for idx, image in enumerate(images):
            # Resize image
            # TODO: move resizing to mold_image()
            resized_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(resized_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                idx, image.shape, molded_image.shape, window, scale,
                # UNCLEAR: 这里 active_class_ids 没有用吗?
                active_class_ids=np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the application.

        Args:
            detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
            mrcnn_mask: [N, height, width, num_classes]
            original_image_shape: [H, W, C] Original image shape before resizing
            image_shape: [H, W, C] Shape of the image after resizing and padding
            window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                    image is excluding the padding.

        Returns:
            boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
            class_ids: [N] Integer class IDs for each bounding box
            scores: [N] Float probability scores of the class_id
            masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        # 每个 box 分别取相应 class_id 的 mask
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        # FIXME: boxes 是在 image 上的坐标, 现在要转换成在 window 上的坐标, 这里都是 normalized_coordinates
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        # boxes 所在 window 是 (0,0,1,1), boxes 乘以原图的高和宽就能得到 boxes 在原图中的位置
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        if full_masks:
            full_masks = np.stack(full_masks, axis=-1)
        else:
            # full_masks 内容是空的,但是 shape 为 (h, w, 0)
            full_masks = np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        Args:
            images: List of images, potentially of different sizes.

        Returns: a list of dicts, one dict per image.
            rois: [N, (y1, x1, y2, x2)] detection bounding boxes
            class_ids: [N] int class IDs
            scores: [N] float probability scores for the class IDs
            masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ = \
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks = \
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """Runs the detection pipeline, but expect inputs that are
        molded already. Used mostly for debugging and inspecting
        the model.

        molded_images: List of images loaded using load_image_gt()
        image_metas: image meta data, also returned by load_image_gt()

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.config.BATCH_SIZE, \
            "Number of images must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ = \
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks = \
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       window)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """Runs a sub-set of the computation graph that computes the given outputs.

        Args:
            image_metas: If provided, the images are assumed to be already
                         molded (i.e. resized, padded, and normalized)

            outputs: List of tuples (name, tensor) to compute. The tensors are
                     symbolic TensorFlow tensors and the names are for easy tracking.

        Returns: an ordered dict of results.
                 Keys are the names received in the input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Prepare inputs
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        # Run inference
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +  # size=3
        list(window) +  # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +  # size=1
        list(active_class_ids)  # size=num_classes
    )
    return meta


def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: (batch_size, meta_length=12 + num_classes) where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    # non_zeros 作为 mask, 获取 boxes 中对应 mask 为 True 的 item
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row in x depending on the values in counts.
    """
    outputs = []
    # num_rows 其实等于 batch_size=config.IMAGES_PER_GPU
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: (batch_size, (y1, x1, y2, x2)) in pixel coordinates
    shape: ((height, width),) in pixels

    Note: In pixel coordinates (y2, x2) is outside the box.
    UNCLEAR: 为什么不包含?
    But in normalized coordinates it's inside the box.

    Returns:
           (batch_size, (y1, x1, y2, x2)) in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
