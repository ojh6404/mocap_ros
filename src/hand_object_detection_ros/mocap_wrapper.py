#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import sys
import torch
import numpy as np
import cv2
from pyvirtualdisplay import Display
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, Point
from jsk_recognition_msgs.msg import Segment, HumanSkeleton

# utils and constants
from hand_object_detection_ros.utils import (
    rotation_matrix_to_quaternion,
    draw_axis,
    load_hamer,
    load_hmr2,
    recursive_to,
    cam_crop_to_full,
    MANO_JOINTS_CONNECTION,
    MANO_CONNECTION_NAMES,
    SMPL_JOINTS_CONNECTION,
    SMPL_CONNECTION_NAMES,
    SPIN_JOINTS_CONNECTION,
    SPIN_CONNECTION_NAMES,
    FRANKMOCAP_PATH,
    FRANKMOCAP_CHECKPOINT,
    SMPL_DIR,
    HAMER_CHECKPOINT_PATH,
    HAMER_CONFIG_PATH,
    Renderer,
)

# frankmocap hand
sys.path.insert(0, FRANKMOCAP_PATH)
import mocap_utils.demo_utils as demo_utils
from handmocap.hand_mocap_api import HandMocap as FrankMocapHand

# hamer
from hamer.datasets.vitdet_dataset import ViTDetDataset as HamerViTDetDataset

# 4DHuman
from hmr2.models import DEFAULT_CHECKPOINT
from hmr2.datasets.vitdet_dataset import ViTDetDataset as HMR2ViTDetDataset


class MocapModelFactory:
    @staticmethod
    def from_config(model: str, model_config: dict):
        if model == "frankmocap_hand":
            return FrankMocapHandModel(**model_config)
        elif model == "hamer":
            return HamerModel(**model_config)
        elif model == "4d-human":
            return HMR2Model(**model_config)
        else:
            raise ValueError(f"Invalid mocap model: {model_config['model']}")


class MocapModelBase(ABC):
    @abstractmethod
    def predict(self, detection_results, im, vis_im):
        pass


class FrankMocapHandModel(MocapModelBase):
    def __init__(
        self,
        img_size: tuple = (640, 480),
        render_type: str = "opengl",
        visualize: bool = True,
        device: str = "cuda:0",
    ):
        self.display = Display(visible=0, size=img_size)
        self.display.start()
        self.img_size = img_size
        self.visualize = visualize
        self.device = device

        # init model
        self.mocap = FrankMocapHand(FRANKMOCAP_CHECKPOINT, SMPL_DIR, device=self.device)
        if self.visualize:
            if self.render_type in ["pytorch3d", "opendr"]:
                from renderer.screen_free_visualizer import Visualizer
            elif self.render_type == "opengl":
                from renderer.visualizer import Visualizer
            else:
                raise ValueError("Invalid render type")
            self.renderer = Visualizer(self.render_type)

    def predict(self, detection_results, im, vis_im):
        hand_bbox_list = []
        hand_bbox_dict = {"left_hand": None, "right_hand": None}
        if detection_results.detections:
            for detection in detection_results.detections:
                hand_bbox_dict[detection.label] = np.array(  # type: ignore
                    [
                        detection.rect.x,
                        detection.rect.y,
                        detection.rect.width,
                        detection.rect.height,
                    ],
                    dtype=np.float32,
                )
            hand_bbox_list.append(hand_bbox_dict)
            # Hand Pose Regression
            pred_output_list = self.mocap.regress(im, hand_bbox_list, add_margin=True)
            pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

            if self.visualize:
                # visualize
                vis_im = self.renderer.visualize(vis_im, pred_mesh_list=pred_mesh_list, hand_bbox_list=hand_bbox_list)

            for hand in pred_output_list[0]:  # TODO: handle multiple hands
                if pred_output_list[0][hand] is not None:
                    joint_coords = pred_output_list[0][hand]["pred_joints_img"]
                    hand_origin = np.sum(joint_coords[PALM_JOINTS] * WEIGHTS[:, None], axis=0)
                    hand_orientation = pred_output_list[0][hand]["pred_hand_pose"][0, :3].astype(
                        np.float32
                    )  # angle-axis representation

                    joint_3d_coords = pred_output_list[0][hand]["pred_joints_smpl"]  # (21, 3)
                    hand_skeleton = HumanSkeleton()
                    hand_skeleton.bone_names = []
                    hand_skeleton.bones = []
                    for i, (start, end) in enumerate(MANO_JOINTS_CONNECTION):
                        bone = Segment()
                        bone.start_point = Point(
                            x=joint_3d_coords[start][0], y=joint_3d_coords[start][1], z=joint_3d_coords[start][2]
                        )
                        bone.end_point = Point(
                            x=joint_3d_coords[end][0], y=joint_3d_coords[end][1], z=joint_3d_coords[end][2]
                        )
                        hand_skeleton.bones.append(bone)
                        hand_skeleton.bone_names.append(MANO_CONNECTION_NAMES[i])

                    hand_pose = Pose()
                    hand_pose.position.x = hand_origin[0]
                    hand_pose.position.y = hand_origin[1]
                    hand_pose.position.z = 0  # cause it's working in 2D

                    for detection in detection_results.detections:
                        if detection.label == hand:
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

        # return detection_results, pose_array, vis_im
        return detection_results, vis_im

    def __del__(self):
        self.display.stop()


class HamerModel(MocapModelBase):
    def __init__(
        self,
        focal_length: float = 525.0,
        rescale_factor: float = 2.0,
        img_size: tuple = (640, 480),
        visualize: bool = True,
        device: str = "cuda:0",
    ):
        self.display = Display(visible=0, size=img_size)
        self.display.start()
        self.focal_length = focal_length
        self.rescale_factor = rescale_factor
        self.img_size = img_size
        self.visualize = visualize
        self.device = device

        # init model
        self.mocap, self.model_cfg = load_hamer(
            HAMER_CHECKPOINT_PATH,
            HAMER_CONFIG_PATH,
            img_size=self.img_size,
            focal_length=self.focal_length,
        )
        self.mocap.to(self.device)
        self.mocap.eval()
        if self.visualize:
            self.renderer = Renderer(
                faces=self.mocap.mano.faces,
                cfg=self.model_cfg,
                width=self.img_size[0],
                height=self.img_size[1],
            )

    def predict(self, detection_results, im, vis_im):
        # im : BGR image
        if detection_results.detections:
            boxes = np.array(
                [
                    [
                        detection.rect.x,
                        detection.rect.y,
                        detection.rect.x + detection.rect.width,
                        detection.rect.y + detection.rect.height,
                    ]
                    for detection in detection_results.detections
                ]
            )  # x1, y1, x2, y2
            right = np.array(
                [1 if detection.label == "right_hand" else 0 for detection in detection_results.detections]
            )

            # TODO clean it and fix this not to use datasetloader
            dataset = HamerViTDetDataset(self.model_cfg, im, boxes, right, rescale_factor=self.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
            for batch in dataloader:
                batch = recursive_to(batch, self.device)  # to device
                with torch.no_grad():
                    out = self.mocap(batch)
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] *= 2 * batch["right"] - 1
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = (
                cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            )

            # 2D keypoints
            box_center = batch["box_center"].detach().cpu().numpy()  # [N, 2]
            box_size = batch["box_size"].detach().cpu().numpy()  # [N,]
            pred_keypoints_2d = out["pred_keypoints_2d"].detach().cpu().numpy()  # [N, 21, 2]
            pred_keypoints_2d[:, :, 0] = (2 * right[:, None] - 1) * pred_keypoints_2d[
                :, :, 0
            ]  # flip x-axis for left hand
            pred_keypoints_2d = pred_keypoints_2d * box_size[:, None, None] + box_center[:, None, :]

            # 3D keypoints
            pred_keypoints_3d = out["pred_keypoints_3d"].detach().cpu().numpy()  # [N, 21, 3]
            pred_keypoints_3d[:, :, 0] = (2 * right[:, None] - 1) * pred_keypoints_3d[:, :, 0]
            pred_keypoints_3d += pred_cam_t_full[:, None, :]

            # hand pose
            hand_origin = np.mean(pred_keypoints_2d, axis=1)  # [N, 2]
            hand_origin = np.concatenate([hand_origin, np.zeros((hand_origin.shape[0], 1))], axis=1)  # [N, 3]
            global_orient = out["pred_mano_params"]["global_orient"].squeeze(1).detach().cpu().numpy()  # [N, 3, 3]
            quats = []

            for i, hand_id in enumerate(right):  # for each hand
                assert (
                    detection_results.detections[i].label == "right_hand" if hand_id == 1 else "left_hand"
                ), "Hand ID and hand detection mismatch"
                hand_pose = Pose()
                hand_pose.position.x = pred_keypoints_3d[i][0][0]  # wrist
                hand_pose.position.y = pred_keypoints_3d[i][0][1]
                hand_pose.position.z = pred_keypoints_3d[i][0][2]
                rotation = global_orient[i]
                if hand_id == 0:
                    rotation[1::3] *= -1
                    rotation[2::3] *= -1

                quat = rotation_matrix_to_quaternion(rotation)  # [w, x, y, z]
                if right[i] == 1:
                    x_axis = np.array([0, 0, 1])
                    y_axis = np.array([0, -1, 0])
                    z_axis = np.array([-1, 0, 0])
                    rotated_result = R.from_rotvec(np.pi * np.array([1, 0, 0])) * R.from_quat(
                        quat
                    )  # rotate 180 degree around x-axis
                    quat = rotated_result.as_quat()  # [w, x, y, z]
                else:
                    x_axis = np.array([0, 0, -1])
                    y_axis = np.array([0, -1, 0])
                    z_axis = np.array([1, 0, 0])
                    rotated_result = R.from_rotvec(np.pi * np.array([0, 0, 1])) * R.from_quat(
                        quat
                    )  # rotate 180 degree around x-axis
                    quat = rotated_result.as_quat()  # [w, x, y, z]
                quats.append(quat)
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
                hand_skeleton = HumanSkeleton()
                hand_skeleton.bone_names = []
                hand_skeleton.bones = []
                for j, (start, end) in enumerate(MANO_JOINTS_CONNECTION):
                    bone = Segment()
                    bone.start_point = Point(
                        x=pred_keypoints_3d[i][start][0],
                        y=pred_keypoints_3d[i][start][1],
                        z=pred_keypoints_3d[i][start][2],
                    )
                    bone.end_point = Point(
                        x=pred_keypoints_3d[i][end][0], y=pred_keypoints_3d[i][end][1], z=pred_keypoints_3d[i][end][2]
                    )
                    hand_skeleton.bones.append(bone)
                    hand_skeleton.bone_names.append(MANO_CONNECTION_NAMES[j])

                detection_results.detections[i].pose = hand_pose
                detection_results.detections[i].skeleton = hand_skeleton

            if self.visualize:
                all_verts = []
                all_cam_t = []
                all_right = []

                # Render the result
                batch_size = batch["img"].shape[0]
                for n in range(batch_size):
                    # Add all verts and cams to list
                    verts = out["pred_vertices"][n].detach().cpu().numpy()
                    is_right = batch["right"][n].cpu().numpy()
                    verts[:, 0] = (2 * is_right - 1) * verts[:, 0]  # Flip x-axis
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)

                # Render front view
                if len(all_verts) > 0:
                    rgba, _ = self.renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, is_right=all_right)
                    rgb = rgba[..., :3].astype(np.float32)
                    alpha = rgba[..., 3].astype(np.float32) / 255.0
                    vis_im = (
                        alpha[..., None] * rgb + (1 - alpha[..., None]) * cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    ).astype(np.uint8)

        # return detection_results, pose_array, vis_im
        return detection_results, vis_im

    def __del__(self):
        self.display.stop()


class HMR2Model(MocapModelBase):
    def __init__(
        self,
        focal_length: float = 525.0,
        rescale_factor: float = 2.0,
        img_size: tuple = (640, 480),
        visualize: bool = True,
        device: str = "cuda:0",
    ):
        self.display = Display(visible=0, size=img_size)
        self.display.start()
        self.focal_length = focal_length
        self.rescale_factor = rescale_factor
        self.img_size = img_size
        self.visualize = visualize
        self.device = device

        # init model
        self.mocap, self.model_cfg = load_hmr2(
            DEFAULT_CHECKPOINT,
            img_size=self.img_size,
            focal_length=self.focal_length,
        )
        self.mocap.to(self.device)
        self.mocap.eval()
        if self.visualize:
            self.renderer = Renderer(
                faces=self.mocap.smpl.faces,
                cfg=self.model_cfg,
                width=self.img_size[0],
                height=self.img_size[1],
            )

    def predict(self, detection_results, im, vis_im):
        # im : BGR image
        if detection_results.detections:
            boxes = np.array(
                [
                    [
                        detection.rect.x,
                        detection.rect.y,
                        detection.rect.x + detection.rect.width,
                        detection.rect.y + detection.rect.height,
                    ]
                    for detection in detection_results.detections
                ]
            )  # x1, y1, x2, y2

            # TODO clean it and fix this not to use datasetloader
            dataset = HMR2ViTDetDataset(self.model_cfg, im, boxes)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
            for batch in dataloader:
                batch = recursive_to(batch, self.device)  # to device
                with torch.no_grad():
                    out = self.mocap(batch)
            pred_cam = out["pred_cam"]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = (
                cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            )

            # this model uses 44 keypoints, but we use only 25 keypoints which corresponds to OpenPose keypoints
            # 2D keypoints
            box_center = batch["box_center"].detach().cpu().numpy()  # [N, 2]
            box_size = batch["box_size"].detach().cpu().numpy()  # [N,]
            pred_keypoints_2d = out["pred_keypoints_2d"].detach().cpu().numpy()  # [N, 44, 2]
            pred_keypoints_2d = pred_keypoints_2d * box_size[:, None, None] + box_center[:, None, :]
            pred_keypoints_2d = pred_keypoints_2d[:, :25, :]  # use only 25 keypoints

            # 3D keypoints
            pred_keypoints_3d = out["pred_keypoints_3d"].detach().cpu().numpy()  # [N, 44, 3]
            pred_keypoints_3d += pred_cam_t_full[:, None, :]
            pred_keypoints_3d = pred_keypoints_3d[:, :25, :]  # use only 25 keypoints

            # body pose
            body_origin = np.mean(pred_keypoints_2d, axis=1)  # [N, 2]
            body_origin = np.concatenate([body_origin, np.zeros((body_origin.shape[0], 1))], axis=1)  # [N, 3]
            global_orient = out["pred_smpl_params"]["global_orient"].squeeze(1).detach().cpu().numpy()  # [N, 3, 3]
            quats = []

            for i in range(len(detection_results.detections)):  # for each body
                body_pose = Pose()
                body_pose.position.x = pred_keypoints_3d[i][0][0]  # wrist
                body_pose.position.y = pred_keypoints_3d[i][0][1]
                body_pose.position.z = pred_keypoints_3d[i][0][2]
                rotation = global_orient[i]

                quat = rotation_matrix_to_quaternion(rotation)  # [w, x, y, z]
                x_axis = np.array([0, 1, 0])
                y_axis = np.array([1, 0, 0])
                z_axis = np.array([0, 0, 1])
                rotated_result = R.from_rotvec(np.pi * np.array([1, 0, 0])) * R.from_quat(
                    quat
                )  # rotate 180 degree around x-axis
                quat = rotated_result.as_quat()  # [w, x, y, z]
                quats.append(quat)
                body_pose.orientation.x = quat[1]
                body_pose.orientation.y = quat[2]
                body_pose.orientation.z = quat[3]
                body_pose.orientation.w = quat[0]
                x_axis_rotated = rotation @ x_axis
                y_axis_rotated = rotation @ y_axis
                z_axis_rotated = rotation @ z_axis
                # visualize hand orientation
                vis_im = draw_axis(vis_im, body_origin[i], x_axis_rotated, (0, 0, 255))  # x: red
                vis_im = draw_axis(vis_im, body_origin[i], y_axis_rotated, (0, 255, 0))  # y: green
                vis_im = draw_axis(vis_im, body_origin[i], z_axis_rotated, (255, 0, 0))  # z: blue

                # body skeleton
                body_skeleton = HumanSkeleton()
                body_skeleton.bone_names = []
                body_skeleton.bones = []
                for j, (start, end) in enumerate(SPIN_JOINTS_CONNECTION):
                    bone = Segment()
                    bone.start_point = Point(
                        x=pred_keypoints_3d[i][start][0],
                        y=pred_keypoints_3d[i][start][1],
                        z=pred_keypoints_3d[i][start][2],
                    )
                    bone.end_point = Point(
                        x=pred_keypoints_3d[i][end][0], y=pred_keypoints_3d[i][end][1], z=pred_keypoints_3d[i][end][2]
                    )

                    body_skeleton.bones.append(bone)
                    body_skeleton.bone_names.append(SPIN_CONNECTION_NAMES[j])

                detection_results.detections[i].pose = body_pose
                detection_results.detections[i].skeleton = body_skeleton

            if self.visualize:
                all_verts = []
                all_cam_t = []

                # Render the result
                batch_size = batch["img"].shape[0]
                for n in range(batch_size):
                    # Add all verts and cams to list
                    verts = out["pred_vertices"][n].detach().cpu().numpy()
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)

                # Render front view
                if len(all_verts) > 0:
                    rgba, _ = self.renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t)
                    rgb = rgba[..., :3].astype(np.float32)
                    alpha = rgba[..., 3].astype(np.float32) / 255.0
                    vis_im = (
                        alpha[..., None] * rgb + (1 - alpha[..., None]) * cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    ).astype(np.uint8)

                # Draw 2D keypoints
                for i, keypoints in enumerate(pred_keypoints_2d):
                    for j, keypoint in enumerate(keypoints):
                        cv2.circle(vis_im, (int(keypoint[0]), int(keypoint[1])), 5, (0, 255, 0), -1)

        # return detection_results, pose_array, vis_im
        return detection_results, vis_im

    def __del__(self):
        self.display.stop()
