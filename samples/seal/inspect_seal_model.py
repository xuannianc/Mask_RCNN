import os
import os.path as osp
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
import cv2

SEAL_DIR = os.path.join(ROOT_DIR, 'samples', 'seal')
# Directory to save logs and trained model
MODEL_DIR = os.path.join(SEAL_DIR, 'logs')
config = seal.SealConfig()
DATASET_DIR = osp.join(ROOT_DIR, 'datasets', 'seal')


# Override the training configurations with a few changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
# config.display()
# Device to load the neural network on. Useful if you're training a model on the same machine,
# in which case use CPU and leave the GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# Load validation dataset
dataset = seal.SealDataset()
dataset.load_seal(DATASET_DIR, "val")
# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode=TEST_MODE, model_dir=MODEL_DIR, config=config)
# Set path to balloon weights file

# Download file from the Releases page and set its path
# https://github.com/matterport/Mask_RCNN/releases
weights_path = osp.join(MODEL_DIR, 'seals20181012T1645/mask_rcnn_seals_0030.h5')

# Or, load the last model you trained
# weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
image_id = random.choice(dataset.image_ids)


def display_resized_image():
    resized_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    image_info = dataset.image_info[image_id]
    # Note: image_info 的 id 是 image 的 filename
    print("Image ID: {}.{} ({}) {}".format(image_info["source"], image_info["id"], image_id,
                                           dataset.image_reference(image_id)))

    # Run object detection
    # rois: [N, (y1, x1, y2, x2)] detection bounding boxes
    # class_ids: [N] int class IDs
    # scores: [N] float probability scores for the class IDs
    # masks: [H, W, N] instance binary masks
    results = model.detect([resized_image], verbose=1)

    # Display results
    ax = get_ax()
    r = results[0]
    visualize.display_instances(resized_image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], ax=ax, title="Predictions")
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    plt.show()


# display_resized_image()


def display_image():
    image = dataset.load_image(image_id)
    image_info = dataset.image_info[image_id]
    # Note: image_info 的 id 是 image 的 filename
    print("Image ID: {}.{} ({}) {}".format(image_info["source"], image_info["id"], image_id,
                                           dataset.image_reference(image_id)))

    # Run object detection
    # rois: [N, (y1, x1, y2, x2)] detection bounding boxes
    # class_ids: [N] int class IDs
    # scores: [N] float probability scores for the class IDs
    # masks: [H, W, N] instance binary masks
    results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax()
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], ax=ax, title="Predictions")
    plt.show()


# display_image()

def color_splash():
    image = dataset.load_image(image_id)
    image_info = dataset.image_info[image_id]
    # Note: image_info 的 id 是 image 的 filename
    print("Image ID: {}.{} ({}) {}".format(image_info["source"], image_info["id"], image_id,
                                           dataset.image_reference(image_id)))
    results = model.detect([image], verbose=1)
    r = results[0]
    splashed_image = seal.color_splash(image, r['masks'])
    display_images([splashed_image], titles='color_splash', cols=1)
    # cv2.namedWindow('splashed_image', cv2.WINDOW_NORMAL)
    # cv2.imshow('splashed_image', splashed_image)
    # cv2.waitKey(0)


# color_splash()

def display_rpn_targets():
    # Generate RPN trainig targets
    resized_image, image_meta, gt_class_ids, gt_bboxes, gt_masks = \
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    image_info = dataset.image_info[image_id]
    # Note: image_info 的 id 是 image 的 filename
    print("Image ID: {}.{} ({}) {}".format(image_info["source"], image_info["id"], image_id,
                                           dataset.image_reference(image_id)))
    # get_anchors 会把 pixel coordinates 赋值到 self.a
    normalized_anchors = model.get_anchors(resized_image.shape)
    anchors = model.anchors
    # target_rpn_match is 1 for positive anchors, -1 for negative anchors
    # and 0 for neutral anchors.
    target_rpn_match, target_rpn_deltas = modellib.build_rpn_targets(anchors, gt_class_ids, gt_bboxes, model.config)
    log("target_rpn_match", target_rpn_match)
    log("target_rpn_deltas", target_rpn_deltas)

    positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
    negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
    neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
    positive_anchors = model.anchors[positive_anchor_ix]
    negative_anchors = model.anchors[negative_anchor_ix]
    neutral_anchors = model.anchors[neutral_anchor_ix]
    log("positive_anchors", positive_anchors)
    log("negative_anchors", negative_anchors)
    log("neutral anchors", neutral_anchors)

    # Apply refinement deltas to positive anchors
    refined_anchors = utils.apply_box_deltas(
        positive_anchors,
        target_rpn_deltas[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)
    log("refined_anchors", refined_anchors, )
    # Display positive anchors before refinement (dotted) and
    # after refinement (solid).
    visualize.draw_boxes(resized_image, boxes=positive_anchors, refined_boxes=refined_anchors, ax=get_ax())
    plt.show()


# display_rpn_targets()

def display_rpn_prediction():
    # Run RPN sub-graph
    resized_image, image_meta, gt_class_ids, gt_bboxes, gt_masks = \
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    pillar = model.keras_model.get_layer("ROI").output  # node to start searching from

    # TF 1.4 and 1.9 introduce new versions of NMS. Search for all names to support TF 1.3~1.10
    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
    if nms_node is None:
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
    if nms_node is None:  # TF 1.9-1.10
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")

    rpn = model.run_graph([resized_image], [
        ("rpn_class", model.keras_model.get_layer("rpn_class").output),
        ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
        ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
        ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
        ("post_nms_anchor_ix", nms_node),
        ("proposals", model.keras_model.get_layer("ROI").output),
    ])
    ax = get_ax(2, 3)
    # Show top anchors by score (before refinement)
    limit = 100
    # np.flatten() 会把多维数组变成一维数组, 那么此处就默认 batch_size=1, 否则不能这样计算
    # 按从大到小排序
    sorted_anchor_ids = np.argsort(rpn['rpn_class'][:, :, 1].flatten())[::-1]
    visualize.draw_boxes(resized_image, boxes=model.anchors[sorted_anchor_ids[:limit]], ax=ax[0, 0])
    # Show top anchors with refinement. Then with clipping to image boundaries
    limit = 50
    pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], resized_image.shape[:2])
    refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], resized_image.shape[:2])
    visualize.draw_boxes(resized_image, boxes=pre_nms_anchors[:limit],
                         refined_boxes=refined_anchors[:limit], ax=ax[0, 1])
    refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], resized_image.shape[:2])
    visualize.draw_boxes(resized_image, refined_boxes=refined_anchors_clipped[:limit], ax=ax[0, 2])
    # Show refined anchors after non-max suppression
    ixs = rpn["post_nms_anchor_ix"][:limit]
    visualize.draw_boxes(resized_image, refined_boxes=refined_anchors_clipped[ixs], ax=ax[1, 0])
    # Show final proposals
    # These are the same as the previous step (refined anchors
    # after NMS) but with coordinates normalized to [0, 1] range.
    # Convert back to image coordinates for display
    h, w = resized_image.shape[:2]
    proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
    visualize.draw_boxes(resized_image, refined_boxes=proposals, ax=ax[1, 1])
    plt.show()


# display_rpn_prediction()
def display_mrcnn_prediction():
    resized_image, image_meta, gt_class_ids, gt_bboxes, gt_masks = \
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    # Get input and output to classifier and mask heads.
    mrcnn = model.run_graph([resized_image], [
        ("proposals", model.keras_model.get_layer("ROI").output),
        ("probs", model.keras_model.get_layer("mrcnn_class").output),
        ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
    ])
    ax = get_ax(1, 4)
    ################################## display detections ###############################################
    # Get detection class IDs. Trim zero padding.
    det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    padding_start_ix = np.where(det_class_ids == 0)[0][0]
    det_class_ids = det_class_ids[:padding_start_ix]
    detections = mrcnn['detections'][0, :padding_start_ix]
    log('trimmed_detection', detections)

    print("{} detections: {}".format(
        padding_start_ix, np.array(dataset.class_names)[det_class_ids]))

    captions = ["{} {:.3f}".format(dataset.class_names[int(class_id)], score) if class_id > 0 else ""
                for class_id, score in zip(detections[:, 4], detections[:, 5])]
    visualize.draw_boxes(resized_image.copy(),
                         refined_boxes=utils.denorm_boxes(detections[:, :4], resized_image.shape[:2]),
                         visibilities=[2] * len(detections),
                         captions=captions, title="Detections",
                         ax=ax[0])
    ################################### display proposals ##########################################
    # Proposals are in normalized coordinates. Scale them to image coordinates.
    h, w = resized_image.shape[:2]
    proposals = np.around(mrcnn["proposals"][0] * np.array([h, w, h, w])).astype(np.int32)

    # Class ID, score, and mask per proposal
    # mrcnn 的 shape 为 (batch_size, num_proposals=1000, num_classes)
    proposal_class_ids = np.argmax(mrcnn["probs"][0], axis=1)
    proposal_class_scores = mrcnn["probs"][0, np.arange(proposal_class_ids.shape[0]), proposal_class_ids]
    proposal_class_names = np.array(dataset.class_names)[proposal_class_ids]
    proposal_positive_ixs = np.where(proposal_class_ids > 0)[0]

    # How many ROIs vs empty rows?
    print("{} valid proposals out of {}".format(np.sum(np.any(proposals, axis=1)), proposals.shape[0]))
    print("{} positive ROIs".format(len(proposal_positive_ixs)))

    # Class counts
    print(list(zip(*np.unique(proposal_class_names, return_counts=True))))
    # Display a random sample of proposals.
    # Proposals classified as background are dotted, and
    # the rest show their class and confidence score.
    limit = 200
    ixs = np.random.randint(0, proposals.shape[0], limit)
    captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
                for c, s in zip(proposal_class_ids[ixs], proposal_class_scores[ixs])]
    visualize.draw_boxes(resized_image.copy(), boxes=proposals[ixs],
                         visibilities=np.where(proposal_class_ids[ixs] > 0, 2, 1),
                         captions=captions, title="Proposals Before Refinement",
                         ax=ax[1])
    #################################### apply bounding box refinement #############################
    # Class-specific bounding box shifts.
    # mrcnn['deltas'] 的 shape 为 (batch_size, num_proposals=1000, num_classes, 4)
    proposal_deltas = mrcnn["deltas"][0, np.arange(proposals.shape[0]), proposal_class_ids]
    log("proposals_deltas", proposal_deltas)

    # Apply bounding box transformations
    # Shape: (num_proposals=1000, (y1, x1, y2, x2)]
    # NOTE: delta 是不分 normalized coordinates 和 pixel coordinates 的
    refined_proposals = utils.apply_box_deltas(
        proposals, proposal_deltas * config.BBOX_STD_DEV).astype(np.int32)
    log("refined_proposals", refined_proposals)

    # Show positive proposals
    # ids = np.arange(proposals.shape[0])  # Display all
    limit = 5
    ids = np.random.randint(0, len(proposal_positive_ixs), limit)  # Display random sample
    captions = ["{} {:.3f}".format(dataset.class_names[class_id], score) if class_id > 0 else ""
                for class_id, score in
                zip(proposal_class_ids[proposal_positive_ixs][ids], proposal_class_scores[proposal_positive_ixs][ids])]
    visualize.draw_boxes(resized_image.copy(), boxes=proposals[proposal_positive_ixs][ids],
                         refined_boxes=refined_proposals[proposal_positive_ixs][ids],
                         visibilities=np.where(proposal_class_ids[proposal_positive_ixs][ids] > 0, 1, 0),
                         captions=captions, title="ROIs After Refinement",
                         ax=ax[2])
    #################################### more steps ################################################
    # Remove boxes classified as background
    keep_proposal_ixs = np.where(proposal_class_ids > 0)[0]
    print("Remove background proposals. Keep {}:\n{}".format(keep_proposal_ixs.shape[0], keep_proposal_ixs))
    # Remove low confidence detections
    keep_proposal_ixs = np.intersect1d(keep_proposal_ixs,
                                       np.where(proposal_class_scores >= config.DETECTION_MIN_CONFIDENCE)[0])
    print("Remove proposals below {} confidence. Keep {}:\n{}".format(
        config.DETECTION_MIN_CONFIDENCE, keep_proposal_ixs.shape[0], keep_proposal_ixs))
    # Apply per-class non-max suppression
    pre_nms_proposals = refined_proposals[keep_proposal_ixs]
    pre_nms_proposal_scores = proposal_class_scores[keep_proposal_ixs]
    pre_nms_proposal_class_ids = proposal_class_ids[keep_proposal_ixs]

    nms_keep_proposal_ixs = []
    for class_id in np.unique(pre_nms_proposal_class_ids):
        # Pick detections of this class
        ixs = np.where(pre_nms_proposal_class_ids == class_id)[0]
        # Apply NMS
        class_keep = utils.non_max_suppression(pre_nms_proposals[ixs],
                                               pre_nms_proposal_scores[ixs],
                                               config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep_proposal_ixs = keep_proposal_ixs[ixs[class_keep]]
        nms_keep_proposal_ixs = np.union1d(nms_keep_proposal_ixs, class_keep_proposal_ixs)
        print("{:12}: {} -> {}".format(dataset.class_names[class_id][:10], keep_proposal_ixs[ixs],
                                       class_keep_proposal_ixs))

    keep_proposal_ixs = np.intersect1d(keep_proposal_ixs, nms_keep_proposal_ixs).astype(np.int32)
    print("\nKeep after per-class NMS: {}\n{}".format(keep_proposal_ixs.shape[0], keep_proposal_ixs))
    #################################### Show final detections #####################################
    ixs = np.arange(len(keep_proposal_ixs))  # Display all
    # ixs = np.random.randint(0, len(keep), 10)  # Display random sample
    captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
                for c, s in
                zip(proposal_class_ids[keep_proposal_ixs][ixs], proposal_class_scores[keep_proposal_ixs][ixs])]
    visualize.draw_boxes(
        resized_image.copy(), boxes=proposals[keep_proposal_ixs][ixs],
        refined_boxes=refined_proposals[keep_proposal_ixs][ixs],
        visibilities=np.where(proposal_class_ids[keep_proposal_ixs][ixs] > 0, 1, 0),
        captions=captions, title="Detections after NMS",
        ax=ax[3])
    plt.show()


# display_mrcnn_prediction()

def display_mrcnn_mask_prediction():
    #################################### Mask Targets ##############################################
    # gt_masks 的 shape 为 (image_height, image_width, num_instances)
    resized_image, image_meta, gt_class_ids, gt_bboxes, gt_masks = \
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    display_images(np.transpose(gt_masks, [2, 0, 1]), cmap="Blues")
    # Get predictions of mask head
    mrcnn = model.run_graph([resized_image], [
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
    ])

    # Get detection class IDs. Trim zero padding.
    det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    padding_start_ix = np.where(det_class_ids == 0)[0][0]
    det_class_ids = det_class_ids[:padding_start_ix]

    print("{} detections: {}".format(padding_start_ix, np.array(dataset.class_names)[det_class_ids]))
    # Masks
    det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], resized_image.shape[:2])
    # mrcnn['masks'] 的 shape 为 (batch_size, num_instances, mask_height, mask_width, num_classes)
    det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c]
                                  for i, c in enumerate(det_class_ids)])
    det_masks = np.array([utils.unmold_mask(mask, det_boxes[i], resized_image.shape)
                          for i, mask in enumerate(det_mask_specific)])
    log("det_mask_specific", det_mask_specific)
    display_images(det_mask_specific[:4] * 255, cmap="Blues", interpolation="none")
    log("det_masks", det_masks)
    display_images(det_masks[:4] * 255, cmap="Blues", interpolation="none")


# display_mrcnn_mask_prediction()

def visualize_activations():
    # Get activations of a few sample layers
    resized_image, image_meta, gt_class_ids, gt_bboxes, gt_masks = \
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    activations = model.run_graph([resized_image], [
        # ("input_image", model.keras_model.get_layer("input_image").output),
        ("res2c_out", model.keras_model.get_layer("res2c_out").output),
        ("res3c_out", model.keras_model.get_layer("res3c_out").output),
        ("res4w_out", model.keras_model.get_layer("res4w_out").output),  # for resnet100
        ("rpn_bbox", model.keras_model.get_layer("rpn_bbox").output),
        ("roi", model.keras_model.get_layer("ROI").output),
    ])
    # Input image (normalized)
    # _ = plt.imshow(modellib.unmold_image(activations["input_image"][0], config))
    # Backbone feature map
    display_images(np.transpose(activations["res2c_out"][0, :, :, :4], [2, 0, 1]), cols=4)


visualize_activations()
