#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import torch
import numpy as np
import cv2
from jsk_recognition_msgs.msg import Rect
from hand_object_detection_ros.msg import HandDetection, HandDetectionArray

# utils and constants
from utils import (
    PASCAL_CLASSES,
    HAND_OBJECT_MODEL_PATH,
    CHECKPOINT_FILE,
    FONT_PATH,
)

# hand object detector
sys.path.insert(0, HAND_OBJECT_MODEL_PATH)
from model.utils.config import cfg as hand_object_detector_cfg
from model.utils.blob import im_list_to_blob
from model.rpn.bbox_transform import clip_boxes
from model.rpn.bbox_transform import bbox_transform_inv
from model.roi_layers import nms
from model.utils.net_utils import vis_detections_filtered_objects, vis_detections_filtered_objects_PIL, filter_object
from model.faster_rcnn.resnet import resnet

np.random.seed(hand_object_detector_cfg.RNG_SEED)

class HandObjectDetectionModel(object):
    def __init__(
        self,
        device="cuda:0",
        hand_threshold=0.9,
        object_threshold=0.9,
        margin=10,
    ):
        self.device = device
        self.hand_threshold = hand_threshold
        self.object_threshold = object_threshold
        self.margin = margin
        self.init_model()

    def predict(self, im):
        # im : BGR image
        obj_dets, hand_dets = self.get_detections(
            im, hand_threshold=self.hand_threshold, object_threshold=self.object_threshold
        )

        # add margin to the bounding box
        if hand_dets is not None:
            for hand_det in hand_dets:
                hand_det[0] = max(0, hand_det[0] - self.margin)
                hand_det[1] = max(0, hand_det[1] - self.margin)
                hand_det[2] = min(im.shape[1], hand_det[2] + self.margin)
                hand_det[3] = min(im.shape[0], hand_det[3] + self.margin)

        detection_results = self.parse_result(obj_dets, hand_dets)
        vis_im = self.get_vis(
            im, obj_dets, hand_dets, hand_threshold=self.hand_threshold, object_threshold=self.object_threshold
        )
        return detection_results, vis_im


    def parse_result(self, obj_dets, hand_dets):
        hand_detections = HandDetectionArray()
        if hand_dets is not None and obj_dets is not None:
            img_obj_id = filter_object(obj_dets, hand_dets)
        if hand_dets is None:
            return hand_detections
        for i, hand_det in enumerate(hand_dets):
            hand_detection = HandDetection()
            hand_detection.hand = "left_hand" if hand_det[-1] < 0.5 else "right_hand"
            hand_detection.hand_rect = self.get_rect(hand_det)
            hand_detection.score = hand_det[4]
            hand_detection.state = (
                "N"
                if hand_det[5] == 0
                else "S" if hand_det[5] == 1 else "O" if hand_det[5] == 2 else "P" if hand_det[5] == 3 else "F"
            )
            if hand_detection.state != "N" and obj_dets is not None:
                hand_detection.object_rect = self.get_rect(obj_dets[img_obj_id[i]])
            hand_detections.detections.append(hand_detection)
        return hand_detections

    def get_rect(self, detection):
        # [boxes(4), score(1), state(1), offset_vector(3), left/right(1)]
        left, top, right, bottom = detection[:4].astype(np.int32)
        return Rect(x=left, y=top, width=right - left, height=bottom - top)

    def get_state(self, detection):
        # [boxes(4), score(1), state(1), offset_vector(3), left/right(1)]
        # N:0,  S:1,  O:2,  P:3,  F:4
        state = detection[5].astype(np.int32)
        return state

    def init_model(self):
        hand_object_detector_cfg.USE_GPU_NMS = True if self.device.startswith("cuda") else False
        hand_object_detector_cfg.CUDA = True if self.device.startswith("cuda") else False

        self.fasterRCNN = resnet(PASCAL_CLASSES, 101, pretrained=False)
        self.fasterRCNN.create_architecture()
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=self.device)

        self.fasterRCNN.load_state_dict(checkpoint["model"])
        self.fasterRCNN.to(self.device)
        self.fasterRCNN.eval()

        if "pooling_mode" in checkpoint.keys():
            hand_object_detector_cfg.POOLING_MODE = checkpoint["pooling_mode"]

        if hand_object_detector_cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            hand_object_detector_cfg.TRAIN.BBOX_NORMALIZE_MEANS = torch.tensor(
                hand_object_detector_cfg.TRAIN.BBOX_NORMALIZE_MEANS, dtype=torch.float, device=self.device
            )
            hand_object_detector_cfg.TRAIN.BBOX_NORMALIZE_STDS = torch.tensor(
                hand_object_detector_cfg.TRAIN.BBOX_NORMALIZE_STDS, dtype=torch.float, device=self.device
            )

        # initilize the tensor holder here.
        self._im_data = torch.FloatTensor(1).to(self.device)
        self._im_info = torch.FloatTensor(1).to(self.device)
        self._num_boxes = torch.LongTensor(1).to(self.device)
        self._gt_boxes = torch.FloatTensor(1).to(self.device)
        self._box_info = torch.FloatTensor(1)

    @torch.no_grad()
    def inference_step(self, im_blob, im_scales):
        im_data_pt = torch.as_tensor(im_blob, device=self.device).permute(0, 3, 1, 2)
        im_info_pt = torch.tensor(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=torch.float32, device=self.device
        )

        self._im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        self._im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        self._gt_boxes.resize_(1, 1, 5).zero_()
        self._num_boxes.resize_(1).zero_()
        self._box_info.resize_(1, 1, 5).zero_()

        res = self.fasterRCNN(self._im_data, self._im_info, self._gt_boxes, self._num_boxes, self._box_info)
        return res

    def get_detections(self, im, class_agnostic=False, hand_threshold=0.9, object_threshold=0.9):
        im_blob, im_scales = self._get_image_blob(im)
        res = self.inference_step(im_blob, im_scales)
        scores, boxes, bbox_pred, loss_list = self.decompose_result(res)

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

        pred_boxes = self.bounding_box_regression(bbox_pred, boxes, scores)
        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        obj_dets, hand_dets = None, None
        for j in range(1, len(PASCAL_CLASSES)):
            # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
            if PASCAL_CLASSES[j] == "hand":
                inds = torch.nonzero(scores[:, j] > hand_threshold).view(-1)
            elif PASCAL_CLASSES[j] == "targetobject":
                inds = torch.nonzero(scores[:, j] > object_threshold).view(-1)

            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4 : (j + 1) * 4]

                cls_dets = torch.cat(
                    (
                        cls_boxes,
                        cls_scores.unsqueeze(1),
                        contact_indices[inds],
                        offset_vector.squeeze(0)[inds],
                        lr[inds],
                    ),
                    1,
                )
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], hand_object_detector_cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if PASCAL_CLASSES[j] == "targetobject":
                    obj_dets = cls_dets.cpu().numpy()
                if PASCAL_CLASSES[j] == "hand":
                    hand_dets = cls_dets.cpu().numpy()

        return obj_dets, hand_dets

    def _get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
        im (ndarray): a color image in BGR order
        Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= hand_object_detector_cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in hand_object_detector_cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > hand_object_detector_cfg.TEST.MAX_SIZE:
                im_scale = float(hand_object_detector_cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def bounding_box_regression(self, bbox_pred, boxes, scores, num_classes=len(PASCAL_CLASSES), class_agnostic=False):
        if hand_object_detector_cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if hand_object_detector_cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                box_deltas = (
                    box_deltas.view(-1, 4) * hand_object_detector_cfg.TRAIN.BBOX_NORMALIZE_STDS
                    + hand_object_detector_cfg.TRAIN.BBOX_NORMALIZE_MEANS
                )
                if class_agnostic:
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(1, -1, 4 * num_classes)

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, self._im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        return pred_boxes

    def decompose_result(self, inference_result):
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

    def get_vis(self, im, obj_dets, hand_dets, hand_threshold=0.9, object_threshold=0.9, use_PIL=True):
        vis_im = np.copy(im)
        if use_PIL:
            vis_im = vis_detections_filtered_objects_PIL(
                vis_im, obj_dets, hand_dets, hand_threshold, object_threshold, font_path=FONT_PATH
            )
            vis_im = np.array(vis_im.convert("RGB"))  # type: ignore
        else:
            # direct cv2 conversion is faster, but less pretty
            vis_im = vis_detections_filtered_objects(vis_im, obj_dets, hand_dets, hand_threshold)
            vis_im = cv2.cvtColor(vis_im, cv2.COLOR_BGR2RGB)
        return vis_im
