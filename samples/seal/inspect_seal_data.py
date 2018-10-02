import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.seal import seal

config = seal.SealConfig()
SEAL_DIR = os.path.join(ROOT_DIR, "datasets/seal")

# Load dataset
# Get the dataset from the releases page
# https://github.com/matterport/Mask_RCNN/releases
dataset = seal.SealDataset()
dataset.load_seal(SEAL_DIR, "train")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(dataset.num_images))
print("Class Count: {}".format(dataset.num_classes))
print('Class ids and names:')
for i, info in enumerate(dataset.class_info):
    print("\t{:3}. {:50}".format(i, info['name']))


def display_mask():
    # Load and display random samples
    image_ids = np.random.choice(dataset.image_ids, 4)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        # 显示面积最大的 limit(默认为4) 个 class 的 mask
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names)


# display_mask()


def display_bbox(image_id):
    # Load random image and mask.
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    # Compute Bounding box
    bboxes = utils.extract_bboxes(mask)

    # Display image and additional stats
    print("image_id ", image_id, dataset.image_reference(image_id))
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bboxes)
    # Display image and instances
    visualize.display_instances(image, bboxes, mask, class_ids, dataset.class_names)


# image_id = random.choice(dataset.image_ids)
# display_bbox(image_id)

def resize_image(image_id):
    # Load random image and mask.
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    # Resize
    image, window, scale, padding, _ = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding)
    # Compute Bounding box
    bboxes = utils.extract_bboxes(mask)

    # Display image and additional stats
    print("image_id: ", image_id, dataset.image_reference(image_id))
    print("Original shape: ", original_shape)
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bboxes)
    # Display image and instances
    visualize.display_instances(image, bboxes, mask, class_ids, dataset.class_names)


image_id = np.random.choice(dataset.image_ids, 1)[0]
resize_image(image_id)

def mini_mask(image_id):
    image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
        dataset, config, image_id, use_mini_mask=False)

    log("image", image)
    log("image_meta", image_meta)
    log("class_ids", class_ids)
    log("bbox", bbox)
    log("mask", mask)

    display_images([image] + [mask[:, :, i] for i in range(min(mask.shape[-1], 7))])

    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
    # Add augmentation and mask resizing.
    # 取 mask 的 bbox, resize 到固定大小
    image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
        dataset, config, image_id, augment=True, use_mini_mask=True)
    log("mask", mask)
    display_images([image] + [mask[:, :, i] for i in range(min(mask.shape[-1], 7))])
    # 把 mask 恢复到原来大小
    mask = utils.expand_mask(bbox, mask, image.shape)
    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)


# image_id = np.random.choice(dataset.image_ids, 1)[0]
# mini_mask(image_id)
def generate_anchors():
    # 显示所有 layer 中间的 anchors
    # Generate Anchors
    backbone_shapes = modellib.compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Print summary of anchors
    num_levels = len(backbone_shapes)
    anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
    print("Count: ", anchors.shape[0])
    print("Scales: ", config.RPN_ANCHOR_SCALES)
    print("ratios: ", config.RPN_ANCHOR_RATIOS)
    print("Anchors per Cell: ", anchors_per_cell)
    print("Levels: ", num_levels)
    anchors_per_level = []
    for l in range(num_levels):
        num_cells = backbone_shapes[l][0] * backbone_shapes[l][1]
        anchors_per_level.append(anchors_per_cell * num_cells // config.RPN_ANCHOR_STRIDE ** 2)
        print("Anchors in Level {}: {}".format(l, anchors_per_level[l]))
    return backbone_shapes, anchors, anchors_per_level


def visualize_center_anchors(image_id, anchors_per_cell):
    # Visualize anchors of one cell at the center of the feature map of a specific level
    backbone_shapes, anchors, anchors_per_level = generate_anchors()
    # Load and draw random image
    image, image_meta, _, _, _ = modellib.load_image_gt(dataset, config, image_id)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    num_levels = len(backbone_shapes)

    for level in range(num_levels):
        colors = visualize.random_colors(num_levels)
        # Compute the index of the anchors at the center of the image
        level_start = sum(anchors_per_level[:level])  # sum of anchors of previous levels
        level_anchors = anchors[level_start:level_start + anchors_per_level[level]]
        print("Level {}. Anchors: {:6}  Feature map Shape: {}".format(level, level_anchors.shape[0],
                                                                      backbone_shapes[level]))
        # backbone_shapes 为 [(256,256),(128,128),(64,64),(32,32),(16,16)]
        center_cell = backbone_shapes[level] // 2
        center_anchor = anchors_per_cell * (
                (center_cell[0] * backbone_shapes[level][1] / config.RPN_ANCHOR_STRIDE ** 2) \
                + center_cell[1] / config.RPN_ANCHOR_STRIDE)
        level_center = int(center_anchor)

        # Draw anchors. Brightness show the order in the array, dark to bright.
        for i, rect in enumerate(level_anchors[level_center:level_center + anchors_per_cell]):
            y1, x1, y2, x2 = rect
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, facecolor='none',
                                  edgecolor=(i + 1) * np.array(colors[level]) / anchors_per_cell)
            ax.add_patch(p)
    plt.show()


# image_id = np.random.choice(dataset.image_ids, 1)[0]
# visualize_center_anchors(image_id, anchors_per_cell=len(config.RPN_ANCHOR_RATIOS))

def create_data_generator(num_random_rois=2000, detection_targets=True):
    # Create data generator
    g = modellib.data_generator(dataset, config, shuffle=True, random_rois=num_random_rois, batch_size=4,
                                detection_targets=detection_targets)
    return g, num_random_rois, detection_targets


def evaluate_data_generator():
    g, num_random_rois, detection_targets = create_data_generator()
    # Get Next Image
    if num_random_rois:
        if detection_targets:
            [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, rpn_rois, rois], \
            [mrcnn_class_ids, mrcnn_bbox, mrcnn_mask] = next(g)
            log("mrcnn_class_ids", mrcnn_class_ids)
            log("mrcnn_bbox", mrcnn_bbox)
            log("mrcnn_mask", mrcnn_mask)
        else:
            [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, rpn_rois, rois], _ \
                = next(g)
        log("rois", rois)
        log("rpn_rois", rpn_rois)
    else:
        [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks], _ = next(g)
    log("gt_class_ids", gt_class_ids)
    log("gt_boxes", gt_boxes)
    log("gt_masks", gt_masks)
    log("rpn_match", rpn_match, )
    log("rpn_bbox", rpn_bbox)
    image_ids = modellib.parse_image_meta(image_meta)["image_id"]
    for image_id in image_ids:
        # image_reference 返回 image 的路径
        print("image_id: ", image_id, dataset.image_reference(image_id))


# evaluate_data_generator()

def show_anchors():
    g, _, _ = create_data_generator(num_random_rois=0)
    # Get Next Image
    [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks], _ = next(g)
    b = 0
    # Restore original image (reverse normalization)
    sample_image = modellib.unmold_image(normalized_images[b], config)
    # Compute anchor shifts.
    indices = np.where(rpn_match[b] == 1)[0]
    # Generate anchors
    backbone_shapes, anchors, anchors_per_level = generate_anchors()
    refined_anchors = utils.apply_box_deltas(anchors[indices], rpn_bbox[b, :len(indices)] * config.RPN_BBOX_STD_DEV)
    log("anchors", anchors)
    log("refined_anchors", refined_anchors)

    # Get list of positive anchors
    positive_anchor_ids = np.where(rpn_match[b] == 1)[0]
    print("Positive anchors: {}".format(len(positive_anchor_ids)))
    negative_anchor_ids = np.where(rpn_match[b] == -1)[0]
    print("Negative anchors: {}".format(len(negative_anchor_ids)))
    neutral_anchor_ids = np.where(rpn_match[b] == 0)[0]
    print("Neutral anchors: {}".format(len(neutral_anchor_ids)))

    # Show positive anchors
    fig, ax = plt.subplots(1, figsize=(16, 16))
    visualize.draw_boxes(sample_image.copy(), boxes=anchors[positive_anchor_ids],
                         refined_boxes=refined_anchors, ax=ax)
    # Show negative anchors
    visualize.draw_boxes(sample_image.copy(), boxes=anchors[negative_anchor_ids])
    # Show neutral anchors. They don't contribute to training.
    visualize.draw_boxes(sample_image.copy(), boxes=anchors[np.random.choice(neutral_anchor_ids, 100)])
    plt.show()


# show_anchors()

def show_rois():
    g, num_random_rois, detection_targets = create_data_generator()
    [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, rpn_rois, rois], \
    [mrcnn_class_ids, mrcnn_bbox, mrcnn_mask] = next(g)
    # Remove the last dim in mrcnn_class_ids. It's only added
    # to satisfy Keras restriction on target shape.
    # 原来的 mrcnn_class_idx 的 shape 是 (batch_size, num_rois, 1)
    mrcnn_class_ids = mrcnn_class_ids[:, :, 0]
    b = 0
    # Restore original image (reverse normalization)
    sample_image = modellib.unmold_image(normalized_images[b], config)
    # Class aware bboxes
    # mrcnn_bbox 的 shape 是 (batch_size,num_rois,num_classes,4)
    # mrcnn_class_idx 的 shape 是 (batch_size, num_rois)
    bbox_specific = mrcnn_bbox[b, np.arange(mrcnn_bbox.shape[1]), mrcnn_class_ids[b], :]
    # Refined ROIs
    refined_rois = utils.apply_box_deltas(rois[b].astype(np.float32), bbox_specific[:, :4] * config.BBOX_STD_DEV)
    # Class aware masks
    # mrcnn_mask 的 shape 是 (batch_size, num_rois, 28, 28, num_classes)
    mask_specific = mrcnn_mask[b, np.arange(mrcnn_mask.shape[1]), :, :, mrcnn_class_ids[b]]

    visualize.draw_rois(sample_image, rois[b], refined_rois, mask_specific, mrcnn_class_ids[b], dataset.class_names)

    # Any repeated ROIs?
    # np.ascontiguousarray 是让数组在内存中连续
    # view 方法是创建一个视图,视图和原数组数据共享
    rows = np.ascontiguousarray(rois[b]).view(np.dtype((np.void, rois.dtype.itemsize * rois.shape[-1])))
    _, idx = np.unique(rows, return_index=True)
    print("Unique ROIs: {} out of {}".format(len(idx), rois.shape[1]))
    plt.show()


# show_rois()


def show_roi_with_mask():
    g, num_random_rois, detection_targets = create_data_generator()
    [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, rpn_rois, rois], \
    [mrcnn_class_ids, mrcnn_bbox, mrcnn_mask] = next(g)
    # Remove the last dim in mrcnn_class_ids. It's only added
    # to satisfy Keras restriction on target shape.
    # 原来的 mrcnn_class_idx 的 shape 是 (batch_size, num_rois, 1)
    mrcnn_class_ids = mrcnn_class_ids[:, :, 0]
    b = 0
    # Restore original image (reverse normalization)
    sample_image = modellib.unmold_image(normalized_images[b], config)
    # Class aware bboxes
    # mrcnn_bbox 的 shape 是 (batch_size,num_rois,num_classes,4)
    # mrcnn_class_idx 的 shape 是 (batch_size, num_rois)
    bbox_specific = mrcnn_bbox[b, np.arange(mrcnn_bbox.shape[1]), mrcnn_class_ids[b], :]
    # Refined ROIs
    refined_rois = utils.apply_box_deltas(rois[b].astype(np.float32), bbox_specific[:, :4] * config.BBOX_STD_DEV)
    # Class aware masks
    # mrcnn_mask 的 shape 是 (batch_size, num_rois, 28, 28, num_classes)
    mask_specific = mrcnn_mask[b, np.arange(mrcnn_mask.shape[1]), :, :, mrcnn_class_ids[b]]
    # Dispalay ROIs and corresponding masks and bounding boxes
    ids = random.sample(range(rois.shape[1]), 8)
    images = []
    titles = []
    for i in ids:
        image = visualize.draw_box(sample_image.copy(), rois[b, i, :4].astype(np.int32), [255, 0, 0])
        image = visualize.draw_box(image, refined_rois[i].astype(np.int64), [0, 255, 0])
        images.append(image)
        titles.append("ROI {}".format(i))
        images.append(mask_specific[i] * 255)
        titles.append(dataset.class_names[mrcnn_class_ids[b, i]][:20])

    display_images(images, titles, cols=4, cmap="Blues", interpolation="none")


# show_roi_with_mask()


def check_ratio_of_positive_rois(num_random_rois):
    # Check ratio of positive ROIs in a set of images.
    limit = 10
    temp_g = modellib.data_generator(dataset, config, shuffle=True, random_rois=num_random_rois, batch_size=1,
                                     detection_targets=True)
    total = 0
    for i in range(limit):
        _, [mrcnn_class_ids, _, _] = next(temp_g)
        positive_rois = np.sum(mrcnn_class_ids[0] > 0)
        total += positive_rois
        print("{:5} {:5.2f}".format(positive_rois, positive_rois / mrcnn_class_ids.shape[1]))
    print("Average percent: {:.2f}".format(total / (limit * mrcnn_class_ids.shape[1])))

# check_ratio_of_positive_rois(num_random_rois=10000)
