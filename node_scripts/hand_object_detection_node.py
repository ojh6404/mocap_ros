#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import rospy
import torch
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point
from jsk_recognition_msgs.msg import Rect, Segment, HumanSkeleton
from hand_object_detection_ros.msg import HandDetection, HandDetectionArray

# utils and constants
from utils import (
    rotation_matrix_to_quaternion,
    draw_axis,
    load_hamer,
    FINGER_JOINTS_CONNECTION,
    PALM_JOINTS,
    WEIGHTS,
    PASCAL_CLASSES,
    HAND_OBJECT_MODEL_PATH,
    FRANKMOCAP_PATH,
    FRANKMOCAP_CHECKPOINT,
    SMPL_DIR,
    CHECKPOINT_FILE,
    FONT_PATH,
    HAMER_CHECKPOINT_PATH,
    HAMER_CONFIG_PATH,
    HamerRenderer,
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

# frankmocap
sys.path.insert(0, FRANKMOCAP_PATH)
import mocap_utils.demo_utils as demo_utils
from handmocap.hand_mocap_api import HandMocap as FrankMocap

# hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full

np.random.seed(hand_object_detector_cfg.RNG_SEED)


class HandObjectDetectionNode(object):
    def __init__(self):
        self.device = rospy.get_param("~device", "cuda:0")
        self.hand_threshold = rospy.get_param("~hand_threshold", 0.9)
        self.object_threshold = rospy.get_param("~object_threshold", 0.9)
        self.with_handmocap = rospy.get_param("~with_handmocap", True)
        if self.with_handmocap:
            self.mocap_model = rospy.get_param("~mocap_model", "frankmocap") # frankmocap, hamer
            if self.mocap_model == "hamer":
                self.rescale_factor = rospy.get_param("~rescale_factor", 2.0)
            self.visualize = rospy.get_param("~visualize", True)
            if self.visualize:
                self.render_type = rospy.get_param("~render_type", "opengl")  # pytorch3d, opendr, opengl
        self.init_model()
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber("~input_image", Image, self.callback_image, queue_size=1, buff_size=2**24)
        self.pub_debug_image = rospy.Publisher("~debug_image", Image, queue_size=1)
        self.pub_hand_detections = rospy.Publisher("~hand_detections", HandDetectionArray, queue_size=1)

    def callback_image(self, msg):
        im = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        obj_dets, hand_dets = self.get_detections(
            im, hand_threshold=self.hand_threshold, object_threshold=self.object_threshold
        )
        detection_results = self.parse_result(obj_dets, hand_dets)
        detection_results.header = msg.header

        vis_im = self.get_vis(
            im, obj_dets, hand_dets, hand_threshold=self.hand_threshold, object_threshold=self.object_threshold
        )

        if self.with_handmocap:
            if self.mocap_model == "frankmocap":
                hand_bbox_list = []
                hand_bbox_dict = {"left_hand": None, "right_hand": None}
                if detection_results.detections:
                    for detection in detection_results.detections:
                        hand_bbox_dict[detection.hand] = np.array(  # type: ignore
                            [
                                detection.hand_rect.x,
                                detection.hand_rect.y,
                                detection.hand_rect.width,
                                detection.hand_rect.height,
                            ],
                            dtype=np.float32,
                        )
                    hand_bbox_list.append(hand_bbox_dict)
                    # Hand Pose Regression
                    pred_output_list = self.hand_mocap.regress(im, hand_bbox_list, add_margin=True)
                    pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

                    if self.visualize:
                        # visualize
                        vis_im = self.renderer.visualize(
                            vis_im, pred_mesh_list=pred_mesh_list, hand_bbox_list=hand_bbox_list
                        )

                    for hand in pred_output_list[0]:  # TODO: handle multiple hands
                        if pred_output_list[0][hand] is not None:

                            joint_coords = pred_output_list[0][hand]["pred_joints_img"]
                            hand_origin = np.sum(joint_coords[PALM_JOINTS] * WEIGHTS[:, None], axis=0)
                            hand_orientation = pred_output_list[0][hand]["pred_hand_pose"][0, :3].astype(
                                np.float32
                            )  # angle-axis representation

                            joint_3d_coords = pred_output_list[0][hand]["pred_joints_smpl"]  # (21, 3)
                            hand_skeleton = HumanSkeleton(header=msg.header)
                            hand_skeleton.bone_names = []
                            hand_skeleton.bones = []
                            for i, (start, end) in enumerate(FINGER_JOINTS_CONNECTION):
                                bone = Segment()
                                bone.start_point = Point(
                                    x=joint_3d_coords[start][0], y=joint_3d_coords[start][1], z=joint_3d_coords[start][2]
                                )
                                bone.end_point = Point(
                                    x=joint_3d_coords[end][0], y=joint_3d_coords[end][1], z=joint_3d_coords[end][2]
                                )
                                hand_skeleton.bones.append(bone)
                                hand_skeleton.bone_names.append(f"bone_{i}")

                            hand_pose = Pose()
                            hand_pose.position.x = hand_origin[0]
                            hand_pose.position.y = hand_origin[1]
                            hand_pose.position.z = 0  # cause it's working in 2D

                            for detection in detection_results.detections:
                                if detection.hand == hand:
                                    detection.pose = hand_pose
                                    detection.skeleton = hand_skeleton

                            rotation, _ = cv2.Rodrigues(hand_orientation)
                            quat = rotation_matrix_to_quaternion(rotation)  # [w, x, y, z]
                            if hand == "right_hand":
                                x_axis = np.array([0, 0, 1])
                                y_axis = np.array([0, -1, 0])
                                z_axis = np.array([-1, 0, 0])
                                rotated_result = R.from_rotvec(np.pi * np.array([1, 0, 0])) * R.from_quat(quat)
                                quat = rotated_result.as_quat()  # [w, x, y, z]
                            else:
                                x_axis = np.array([0, 0, 1])
                                y_axis = np.array([0, 1, 0])
                                z_axis = np.array([1, 0, 0])
                            hand_pose.orientation.x = quat[1]
                            hand_pose.orientation.y = quat[2]
                            hand_pose.orientation.z = quat[3]
                            hand_pose.orientation.w = quat[0]
                            x_axis_rotated = rotation @ x_axis
                            y_axis_rotated = rotation @ y_axis
                            z_axis_rotated = rotation @ z_axis

                            # visualize hand orientation
                            vis_im = draw_axis(vis_im, hand_origin, x_axis_rotated, (0, 0, 255))  # x: red
                            vis_im = draw_axis(vis_im, hand_origin, y_axis_rotated, (0, 255, 0))  # y: green
                            vis_im = draw_axis(vis_im, hand_origin, z_axis_rotated, (255, 0, 0))  # z: blue
            elif self.mocap_model == "hamer":
                if detection_results.detections:
                    boxes = np.array([[detection.hand_rect.x, detection.hand_rect.y, detection.hand_rect.x + detection.hand_rect.width, detection.hand_rect.y + detection.hand_rect.height] for detection in detection_results.detections])
                    right = np.array([1 if detection.hand == "right_hand" else 0 for detection in detection_results.detections])

                    # TODO clean it and fix this not to use datasetloader
                    dataset = ViTDetDataset(self.hamer_cfg, im, boxes, right, rescale_factor=self.rescale_factor)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
                    for batch in dataloader:
                        batch = recursive_to(batch, self.device) # to device
                        with torch.no_grad():
                            out = self.hand_mocap(batch)
                            # out keys
                            # pred_cam, pred_mano_params, pred_cam_t, focal_length, pred_keypoints_3d, pred_keypoints_2d, pred_vertices

                    if self.visualize:
                        all_verts = []
                        all_cam_t = []
                        all_right = []

                        multiplier = (2*batch['right']-1)
                        pred_cam = out['pred_cam']
                        pred_cam[:,1] = multiplier*pred_cam[:,1]
                        box_center = batch["box_center"].float()
                        box_size = batch["box_size"].float()
                        img_size = batch["img_size"].float()
                        multiplier = (2*batch['right']-1)
                        scaled_focal_length = self.hamer_cfg.EXTRA.FOCAL_LENGTH / self.hamer_cfg.MODEL.IMAGE_SIZE * img_size.max()
                        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                        # Render the result
                        batch_size = batch['img'].shape[0]
                        for n in range(batch_size):
                            # Add all verts and cams to list
                            verts = out['pred_vertices'][n].detach().cpu().numpy()
                            is_right = batch['right'][n].cpu().numpy()
                            verts[:,0] = (2*is_right-1)*verts[:,0] # Flip x-axis
                            cam_t = pred_cam_t_full[n]
                            all_verts.append(verts)
                            all_cam_t.append(cam_t)
                            all_right.append(is_right)


                        # Render front view
                        if len(all_verts) > 0:
                            # misc_args = dict(
                            #     focal_length=scaled_focal_length,
                            # )

                            rgba = self.renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, is_right=all_right)
                            rgb = rgba[..., :3].astype(np.float32)
                            alpha = rgba[..., 3].astype(np.float32) / 255.0

                            # Overlay image
                            input_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # BGR to RGB
                            vis_im = (alpha[..., None] * rgb + (1 - alpha[..., None]) * input_im).astype(np.uint8)

                    # 2D keypoints
                    box_center = batch["box_center"].detach().cpu().numpy() # [N, 2]
                    box_size = batch["box_size"].detach().cpu().numpy() # [N,]
                    pred_keypoints_2d = out['pred_keypoints_2d'].detach().cpu().numpy() # [N, 21, 2]
                    pred_keypoints_2d[:, :, 0] = (2*right[:, None]-1) * pred_keypoints_2d[:, :, 0] # flip x-axis for left hand
                    pred_keypoints_2d = pred_keypoints_2d * box_size[:, None, None] + box_center[:, None, :]

                    # 3D keypoints
                    pred_keypoints_3d = out['pred_keypoints_3d'].detach().cpu().numpy() # [N, 21, 3]
                    pred_keypoints_3d[:, :, 0] = (2*right[:, None]-1) * pred_keypoints_3d[:, :, 0]

                    # hand pose
                    hand_origin = np.sum(pred_keypoints_2d[:, PALM_JOINTS] * WEIGHTS[:, None], axis=1) # [N, 2]
                    hand_origin = np.concatenate([hand_origin, np.zeros((hand_origin.shape[0], 1))], axis=1) # [N, 3]
                    global_orient = out['pred_mano_params']['global_orient'].squeeze(1).detach().cpu().numpy() # [N, 3, 3]

                    for i, hand_id in enumerate(right): # for each hand
                        assert detection_results.detections[i].hand == "right_hand" if hand_id == 1 else "left_hand"
                        hand_pose = Pose()
                        hand_pose.position.x = hand_origin[i][0]
                        hand_pose.position.y = hand_origin[i][1]
                        hand_pose.position.z = 0  # cause it's working in 2D
                        rotation = global_orient[i]
                        if hand_id == 0:
                            rotation[1::3] *= -1
                            rotation[2::3] *= -1

                        quat = rotation_matrix_to_quaternion(rotation) # [w, x, y, z]
                        if right[i] == 1:
                            x_axis = np.array([0, 0, 1])
                            y_axis = np.array([0, -1, 0])
                            z_axis = np.array([-1, 0, 0])
                            rotated_result = R.from_rotvec(np.pi * np.array([1, 0, 0])) * R.from_quat(quat) # rotate 180 degree around x-axis
                            quat = rotated_result.as_quat()  # [w, x, y, z]
                        else:
                            x_axis = np.array([0, 0, -1])
                            y_axis = np.array([0, -1, 0])
                            z_axis = np.array([1, 0, 0])
                            rotated_result = R.from_rotvec(np.pi * np.array([0, 0, 1])) * R.from_quat(quat) # rotate 180 degree around x-axis
                            quat = rotated_result.as_quat()  # [w, x, y, z]
                        hand_pose.orientation.x = quat[1]
                        hand_pose.orientation.y = quat[2]
                        hand_pose.orientation.z = quat[3]
                        hand_pose.orientation.w = quat[0]
                        x_axis_rotated = rotation @ x_axis
                        y_axis_rotated = rotation @ y_axis
                        z_axis_rotated = rotation @ z_axis
                        # visualize hand orientation
                        vis_im = draw_axis(vis_im, hand_origin[i], x_axis_rotated, (0, 0, 255))  # x: red
                        vis_im = draw_axis(vis_im, hand_origin[i], y_axis_rotated, (0, 255, 0))  # y: green
                        vis_im = draw_axis(vis_im, hand_origin[i], z_axis_rotated, (255, 0, 0))  # z: blue

                        # hand skeleton
                        hand_skeleton = HumanSkeleton(header=msg.header)
                        hand_skeleton.bone_names = []
                        hand_skeleton.bones = []
                        for j, (start, end) in enumerate(FINGER_JOINTS_CONNECTION):
                            bone = Segment()
                            bone.start_point = Point(
                                x=pred_keypoints_3d[i][start][0], y=pred_keypoints_3d[i][start][1], z=pred_keypoints_3d[i][start][2]
                            )
                            bone.end_point = Point(
                                x=pred_keypoints_3d[i][end][0], y=pred_keypoints_3d[i][end][1], z=pred_keypoints_3d[i][end][2]
                            )
                            hand_skeleton.bones.append(bone)
                            hand_skeleton.bone_names.append(f"bone_{j}")

                        detection_results.detections[i].pose = hand_pose
                        detection_results.detections[i].skeleton = hand_skeleton
            else:
                raise ValueError("Invalid mocap model")
        vis_msg = self.bridge.cv2_to_imgmsg(vis_im.astype(np.uint8), encoding="rgb8")
        vis_msg.header = msg.header
        self.pub_hand_detections.publish(detection_results)
        self.pub_debug_image.publish(vis_msg)

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

        if self.with_handmocap:
            if self.mocap_model == "frankmocap":
                self.hand_mocap = FrankMocap(FRANKMOCAP_CHECKPOINT, SMPL_DIR, device=self.device)
                if self.visualize:
                    if self.render_type in ["pytorch3d", "opendr"]:
                        from renderer.screen_free_visualizer import Visualizer
                    else:
                        from renderer.visualizer import Visualizer
                    self.renderer = Visualizer(self.render_type)
            elif self.mocap_model == "hamer":
                self.hand_mocap, self.hamer_cfg = load_hamer(HAMER_CHECKPOINT_PATH, HAMER_CONFIG_PATH)
                self.hand_mocap.to(self.device)
                self.hand_mocap.eval()
                if self.visualize:
                    self.renderer = HamerRenderer(faces=self.hand_mocap.mano.faces, cfg=self.hamer_cfg, width=640, height=480)
            else:
                raise ValueError("Invalid mocap model")

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
                box_deltas = box_deltas.view(-1, 4) * hand_object_detector_cfg.TRAIN.BBOX_NORMALIZE_STDS + hand_object_detector_cfg.TRAIN.BBOX_NORMALIZE_MEANS
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


if __name__ == "__main__":
    rospy.init_node("hand_object_detection_node")
    node = HandObjectDetectionNode()
    rospy.loginfo("Hand Object Detection node started")
    rospy.spin()
