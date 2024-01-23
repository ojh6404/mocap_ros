#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import rospkg

MODEL_PATH = rospkg.RosPack().get_path("hand_object_detection_ros") + "/hand_object_detector"
FONT_PATH = MODEL_PATH + "/lib/model/utils/times_b.ttf"
CHECKPOINT_FILE = MODEL_PATH + "/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth"

sys.path.insert(0, MODEL_PATH)

import rospy
import torch
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import Rect, RectArray

from model.utils.config import cfg
from model.utils.blob import im_list_to_blob
from model.rpn.bbox_transform import clip_boxes
from model.rpn.bbox_transform import bbox_transform_inv
from model.roi_layers import nms

from model.utils.net_utils import vis_detections_filtered_objects
from model.utils.net_utils import vis_detections_filtered_objects_PIL

from model.faster_rcnn.resnet import resnet

np.random.seed(cfg.RNG_SEED)

# HAND_THRESHOLD = 0.5
# OBJECT_THRESHOLD = 0.5

# BAXTER
HAND_THRESHOLD = 0.9
OBJECT_THRESHOLD = 0.9

device = torch.device("cuda:0")
cfg.USE_GPU_NMS = True
cfg.CUDA = True

_pascal_classes = np.asarray(["__background__", "targetobject", "hand"])
fasterRCNN = resnet(_pascal_classes, 101, pretrained=False)
fasterRCNN.create_architecture()
checkpoint = torch.load(CHECKPOINT_FILE)

fasterRCNN.load_state_dict(checkpoint["model"])
fasterRCNN.to(device)
fasterRCNN.eval()

if "pooling_mode" in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint["pooling_mode"]

if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    cfg.TRAIN.BBOX_NORMALIZE_MEANS = torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS, dtype=torch.float, device=device)
    cfg.TRAIN.BBOX_NORMALIZE_STDS = torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS, dtype=torch.float, device=device)


# initilize the tensor holder here.
_im_data = torch.FloatTensor(1).to(device)
_im_info = torch.FloatTensor(1).to(device)
_num_boxes = torch.LongTensor(1).to(device)
_gt_boxes = torch.FloatTensor(1).to(device)
_box_info = torch.FloatTensor(1)


# demo.py
def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def bounding_box_regression(bbox_pred, boxes, scores, num_classes=len(_pascal_classes), class_agnostic=False):
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * cfg.TRAIN.BBOX_NORMALIZE_STDS + cfg.TRAIN.BBOX_NORMALIZE_MEANS
            if class_agnostic:
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(1, -1, 4 * num_classes)

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, _im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))
    return pred_boxes


def inference_step(im_blob, im_scales):
    im_data_pt = torch.as_tensor(im_blob, device=device).permute(0, 3, 1, 2)
    im_info_pt = torch.tensor([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=torch.float32, device=device)

    _im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
    _im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
    _gt_boxes.resize_(1, 1, 5).zero_()
    _num_boxes.resize_(1).zero_()
    _box_info.resize_(1, 1, 5).zero_()

    res = fasterRCNN(_im_data, _im_info, _gt_boxes, _num_boxes, _box_info)
    return res


def decompose_result(inference_result):
    (
        rois,
        cls_prob,
        bbox_pred,
        rpn_loss_cls,
        rpn_loss_box,
        RCNN_loss_cls,
        RCNN_loss_bbox,
        rois_label,
        loss_list,
    ) = inference_result

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    return scores, boxes, bbox_pred, loss_list


def get_detections(im, class_agnostic=False, thresh_hand=HAND_THRESHOLD, thresh_obj=OBJECT_THRESHOLD):
    im_blob, im_scales = _get_image_blob(im)
    res = inference_step(im_blob, im_scales)
    scores, boxes, bbox_pred, loss_list = decompose_result(res)

    # extact predicted params
    contact_vector = loss_list[0][0]  # hand contact state info
    offset_vector = loss_list[1][0].detach()  # offset vector (factored into a unit vector and a magnitude)
    lr_vector = loss_list[2][0].detach()  # hand side info (left/right)

    # get hand contact
    _, contact_indices = torch.max(contact_vector, 2)
    contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

    # get hand side
    lr = torch.sigmoid(lr_vector) > 0.5
    lr = lr.squeeze(0).float()

    pred_boxes = bounding_box_regression(bbox_pred, boxes, scores)
    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    obj_dets, hand_dets = None, None
    for j in range(1, len(_pascal_classes)):
        # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
        if _pascal_classes[j] == "hand":
            inds = torch.nonzero(scores[:, j] > thresh_hand).view(-1)
        elif _pascal_classes[j] == "targetobject":
            inds = torch.nonzero(scores[:, j] > thresh_obj).view(-1)

        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if class_agnostic:
                cls_boxes = pred_boxes[inds, :]
            else:
                cls_boxes = pred_boxes[inds][:, j * 4: (j + 1) * 4]

            cls_dets = torch.cat(
                (cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1
            )
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if _pascal_classes[j] == "targetobject":
                obj_dets = cls_dets.cpu().numpy()
            if _pascal_classes[j] == "hand":
                hand_dets = cls_dets.cpu().numpy()

    return obj_dets, hand_dets


def get_vis(im, obj_dets, hand_dets, thresh_hand=HAND_THRESHOLD, thresh_obj=OBJECT_THRESHOLD, use_PIL=True):
    vis_im = np.copy(im)
    if use_PIL:
        vis_im = vis_detections_filtered_objects_PIL(
            vis_im, obj_dets, hand_dets, thresh_hand, thresh_obj, font_path=FONT_PATH
        )
        vis_im = np.array(vis_im.convert("RGB"))
    else:
        # direct cv2 conversion is faster, but less pretty
        vis_im = vis_detections_filtered_objects(vis_im, obj_dets, hand_dets, thresh_hand)
        vis_im = cv2.cvtColor(vis_im, cv2.COLOR_BGR2RGB)

    return vis_im


class HandObjectDetectionNode(object):
    def __init__(self):
        self.sub = rospy.Subscriber("~input_image", Image, self.callback_image, queue_size=1, buff_size=2**24)
        self.pub_debug_image = rospy.Publisher("~debug_image", Image, queue_size=1)
        self.pub_object_boxes = rospy.Publisher("~objects", RectArray, queue_size=1)
        self.pub_hand_boxes = rospy.Publisher("~hands", RectArray, queue_size=1)
        self.bridge = CvBridge()

    def callback_image(self, msg):
        im = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        obj_dets, hand_dets = get_detections(im)
        obj_msg = RectArray(header=msg.header)
        hand_msg = RectArray(header=msg.header)
        stationary_hand_list = []
        if hand_dets is not None:
            hand_msg.rects = [self.get_rect(x) for x in hand_dets]
            stationary_hand_list = [self.get_rect(x) for x in hand_dets if self.get_state(x) == 4]
        if stationary_hand_list:
            # use hand boxes for all stationary objects, in order to
            # ensure interruption even when misinterpreting contact information
            obj_msg.rects = stationary_hand_list
        elif obj_dets is not None:
            obj_msg.rects = [self.get_rect(x) for x in obj_dets]

        self.pub_object_boxes.publish(obj_msg)
        self.pub_hand_boxes.publish(hand_msg)
        vis_im = get_vis(im, obj_dets, hand_dets)
        vis_msg = self.bridge.cv2_to_imgmsg(vis_im, encoding="rgb8")
        self.pub_debug_image.publish(vis_msg)

    def get_rect(self, detection):
        # [boxes(4), score(1), state(1), offset_vector(3), left/right(1)]
        left, top, right, bottom = detection[:4].astype(np.int32)
        return Rect(x=left, y=top, width=right - left, height=bottom - top)

    def get_state(self, detection):
        # [boxes(4), score(1), state(1), offset_vector(3), left/right(1)]
        # N:0,  S:1,  O:2,  P:3,  F:4
        state = detection[5].astype(np.int32)
        return state


if __name__ == "__main__":
    rospy.init_node("hand_object_detection_node")
    node = HandObjectDetectionNode()
    rospy.spin()
