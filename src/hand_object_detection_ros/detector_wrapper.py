#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import sys
import torch
import numpy as np
import cv2
import supervision as sv
from jsk_recognition_msgs.msg import Rect
from hand_object_detection_ros.msg import MocapDetection, MocapDetectionArray

# utils and constants
from hand_object_detection_ros.utils import (
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


BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


class DetectionModelFactory:
    @staticmethod
    def from_config(model: str, model_config: dict):
        if model == "hand_object_detector":
            return HandObjectDetectorModel(**model_config)
        elif model == "mediapipe_hand":
            return MediapipeHandModel(**model_config)
        elif model == "yolo":
            return YoloModel(**model_config)
        else:
            raise ValueError(f"Unknown model: {model}")


class DetectionModelBase(ABC):
    @abstractmethod
    def predict(self, detection_results, im, vis_im):
        pass


class HandObjectDetectorModel(DetectionModelBase):
    def __init__(
        self,
        threshold: float = 0.9,
        object_threshold: float = 0.9,
        margin: int = 10,
        device: str = "cuda:0",
    ):
        self.device = device
        self.threshold = threshold
        self.object_threshold = object_threshold
        self.margin = margin

        # init model
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

    def predict(self, im):
        # im : BGR image
        obj_dets, hand_dets = self.get_detections(
            im, hand_threshold=self.threshold, object_threshold=self.object_threshold
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
            im, obj_dets, hand_dets, hand_threshold=self.threshold, object_threshold=self.object_threshold
        )
        return detection_results, vis_im

    def parse_result(self, obj_dets, hand_dets):
        hand_detections = MocapDetectionArray()
        if hand_dets is not None and obj_dets is not None:
            img_obj_id = filter_object(obj_dets, hand_dets)
        if hand_dets is None:
            return hand_detections
        for i, hand_det in enumerate(hand_dets):
            hand_detection = MocapDetection()
            hand_detection.label = "left_hand" if hand_det[-1] < 0.5 else "right_hand"
            hand_detection.rect = self.get_rect(hand_det)
            hand_detection.score = hand_det[4]
            # hand_detection.state = (
            #     "N"
            #     if hand_det[5] == 0
            #     else "S" if hand_det[5] == 1 else "O" if hand_det[5] == 2 else "P" if hand_det[5] == 3 else "F"
            # )
            # if hand_detection.state != "N" and obj_dets is not None:
            #     hand_detection.object_rect = self.get_rect(obj_dets[img_obj_id[i]])
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


class MediapipeHandModel(DetectionModelBase):
    def __init__(
        self,
        threshold: float = 0.9,
        margin: int = 10,
        device: str = "cuda:0",
    ):
        self.device = device
        self.threshold = threshold
        self.margin = margin

        # init model
        import mediapipe as mp

        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            min_detection_confidence=self.threshold,
            min_tracking_confidence=0.5,
            max_num_hands=2,
        )

    def predict(self, im):
        hand_detections = MocapDetectionArray()
        results = self.mp_hands.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_detection = MocapDetection()

                # flip image
                hand_detection.label = "left_hand" if handedness.classification[0].label == "Right" else "right_hand"
                hand_detection.score = handedness.classification[0].score
                # hand_detection.state = "N"
                image_width, image_height = im.shape[1], im.shape[0]

                landmark_array = np.empty((0, 2), int)

                for _, landmark in enumerate(hand_landmarks.landmark):
                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)

                    landmark_point = [np.array((landmark_x, landmark_y))]

                    landmark_array = np.append(landmark_array, landmark_point, axis=0)

                x, y, w, h = cv2.boundingRect(landmark_array)

                # add margin to the bounding box
                x1 = max(0, x - self.margin)
                y1 = max(0, y - self.margin)
                x2 = min(im.shape[1], x + w + self.margin)
                y2 = min(im.shape[0], y + h + self.margin)
                w = x2 - x1
                h = y2 - y1

                hand_detection.rect = Rect(x=x1, y=y1, width=w, height=h)
                hand_detections.detections.append(hand_detection)

                # draw hand bbox
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # put text left of right
                cv2.putText(im, hand_detection.label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        return hand_detections, im


class YoloModel(DetectionModelBase):
    def __init__(
        self,
        threshold: float = 0.5,
        margin: int = 10,
        device: str = "cuda:0",
    ):
        self.threshold = threshold
        self.margin = margin
        self.device = device

        # init model
        from ultralytics import YOLO

        self.detector = YOLO("yolov9e-seg.pt")

    def predict(self, im):
        # im : BGR image
        results = self.detector(im, save=False, conf=self.threshold, iou=0.5, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]  # just detect person
        xyxys = [xyxy for xyxy in detections.xyxy]
        scores = detections.confidence.tolist()
        body_detections = MocapDetectionArray()
        for i, detection in enumerate(detections):
            body_detection = MocapDetection()
            body_detection.label = "person"
            body_detection.rect = Rect(
                x=int(xyxys[i][0]),
                y=int(xyxys[i][1]),
                width=int(xyxys[i][2] - xyxys[i][0]),
                height=int(xyxys[i][3] - xyxys[i][1]),
            )
            body_detection.score = scores[i]
            body_detections.detections.append(body_detection)

        # visualize
        labels = [self.detector.names[i] for i in detections.class_id]
        labels_with_scores = [f"{label} {score:.2f}" for label, score in zip(labels, scores)]
        visualization = im.copy()
        visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
        visualization = BOX_ANNOTATOR.annotate(scene=visualization, detections=detections)
        visualization = LABEL_ANNOTATOR.annotate(scene=visualization, detections=detections, labels=labels_with_scores)
        return body_detections, visualization
