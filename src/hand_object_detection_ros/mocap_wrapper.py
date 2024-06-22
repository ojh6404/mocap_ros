#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import torch
import numpy as np
import cv2
from pyvirtualdisplay import Display
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, Point, PoseArray
from jsk_recognition_msgs.msg import Segment, HumanSkeleton

# utils and constants
from hand_object_detection_ros.utils import (
    rotation_matrix_to_quaternion,
    draw_axis,
    load_hamer,
    FINGER_JOINTS_CONNECTION,
    CONNECTION_NAMES,
    PALM_JOINTS,
    WEIGHTS,
    FRANKMOCAP_PATH,
    FRANKMOCAP_CHECKPOINT,
    SMPL_DIR,
    HAMER_CHECKPOINT_PATH,
    HAMER_CONFIG_PATH,
    HamerRenderer,
)

# frankmocap
sys.path.insert(0, FRANKMOCAP_PATH)
import mocap_utils.demo_utils as demo_utils
from handmocap.hand_mocap_api import HandMocap as FrankMocap

# hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full

class MocapModel(object):
    def __init__(
            self,
            model_config: dict,
            img_size: tuple=(640, 480),
            visualize: bool=True,
            device: str="cuda:0",
    ):
        self.display = Display(visible=0, size=img_size)
        self.display.start()
        self.model_config = model_config
        self.mocap_model = model_config["mocap_model"]
        if self.mocap_model == "frankmocap":
            self.render_type = model_config["render_type"]
        elif self.mocap_model == "hamer":
            self.focal_length = model_config["focal_length"]
            self.rescale_factor = model_config["rescale_factor"]
        else:
            raise ValueError("Invalid mocap model")
        self.img_size = img_size
        self.visualize = visualize
        self.device = device
        self.init_model()

    def init_model(self):
        if self.mocap_model == "frankmocap":
            self.hand_mocap = FrankMocap(FRANKMOCAP_CHECKPOINT, SMPL_DIR, device=self.device)
            if self.visualize:
                if self.render_type in ["pytorch3d", "opendr"]:
                    from renderer.screen_free_visualizer import Visualizer
                elif self.render_type == "opengl":
                    from renderer.visualizer import Visualizer
                else:
                    raise ValueError("Invalid render type")
                self.renderer = Visualizer(self.render_type)
        elif self.mocap_model == "hamer":
            self.hand_mocap, self.hamer_cfg = load_hamer(
                HAMER_CHECKPOINT_PATH,
                HAMER_CONFIG_PATH,
                img_size=self.img_size,
                focal_length=self.focal_length,
            )
            self.hand_mocap.to(self.device)
            self.hand_mocap.eval()
            if self.visualize:
                self.renderer = HamerRenderer(
                    faces=self.hand_mocap.mano.faces, cfg=self.hamer_cfg, width=self.img_size[0], height=self.img_size[1],
                )
        else:
            raise ValueError("Invalid mocap model")

    def predict(self, detection_results, im, vis_im):
        # im : BGR image
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
                        hand_skeleton = HumanSkeleton()
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
                            hand_skeleton.bone_names.append(CONNECTION_NAMES[i])

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
                pred_cam = out['pred_cam']
                pred_cam[:,1] *= (2*batch['right']-1)
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = self.hamer_cfg.EXTRA.FOCAL_LENGTH / self.hamer_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                # 2D keypoints
                box_center = batch["box_center"].detach().cpu().numpy() # [N, 2]
                box_size = batch["box_size"].detach().cpu().numpy() # [N,]
                pred_keypoints_2d = out['pred_keypoints_2d'].detach().cpu().numpy() # [N, 21, 2]
                pred_keypoints_2d[:, :, 0] = (2*right[:, None]-1) * pred_keypoints_2d[:, :, 0] # flip x-axis for left hand
                pred_keypoints_2d = pred_keypoints_2d * box_size[:, None, None] + box_center[:, None, :]

                # 3D keypoints
                pred_keypoints_3d = out['pred_keypoints_3d'].detach().cpu().numpy() # [N, 21, 3]
                pred_keypoints_3d[:, :, 0] = (2*right[:, None]-1) * pred_keypoints_3d[:, :, 0]
                pred_keypoints_3d += pred_cam_t_full[:, None, :]

                # hand pose
                hand_origin = np.mean(pred_keypoints_2d, axis=1) # [N, 2]
                hand_origin = np.concatenate([hand_origin, np.zeros((hand_origin.shape[0], 1))], axis=1) # [N, 3]
                global_orient = out['pred_mano_params']['global_orient'].squeeze(1).detach().cpu().numpy() # [N, 3, 3]
                quats = []

                for i, hand_id in enumerate(right): # for each hand
                    assert detection_results.detections[i].hand == "right_hand" if hand_id == 1 else "left_hand", "Hand ID and hand detection mismatch"
                    hand_pose = Pose()
                    hand_pose.position.x = pred_keypoints_3d[i][0][0] # wrist
                    hand_pose.position.y = pred_keypoints_3d[i][0][1]
                    hand_pose.position.z = pred_keypoints_3d[i][0][2]
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
                    for j, (start, end) in enumerate(FINGER_JOINTS_CONNECTION):
                        bone = Segment()
                        bone.start_point = Point(
                            x=pred_keypoints_3d[i][start][0], y=pred_keypoints_3d[i][start][1], z=pred_keypoints_3d[i][start][2]
                        )
                        bone.end_point = Point(
                            x=pred_keypoints_3d[i][end][0], y=pred_keypoints_3d[i][end][1], z=pred_keypoints_3d[i][end][2]
                        )
                        hand_skeleton.bones.append(bone)
                        hand_skeleton.bone_names.append(CONNECTION_NAMES[j])

                    detection_results.detections[i].pose = hand_pose
                    detection_results.detections[i].skeleton = hand_skeleton

                if self.visualize:
                    all_verts = []
                    all_cam_t = []
                    all_right = []

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
                        rgba, _ = self.renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, is_right=all_right)
                        rgb = rgba[..., :3].astype(np.float32)
                        alpha = rgba[..., 3].astype(np.float32) / 255.0
                        vis_im = (alpha[..., None] * rgb + (1 - alpha[..., None]) * cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).astype(np.uint8)

                # # project 3d keypoints to 2d and draw
                # for i, keypoints in enumerate(pred_keypoints_2d):
                #     for j, keypoint in enumerate(keypoints):
                #         point_x, point_y = self.camera_model.project3dToPixel(pred_keypoints_3d[i][j])
                #         cv2.circle(vis_im, (int(point_x), int(point_y)), 5, (0, 255, 0), -1)

                pose_array = PoseArray()
                for i, keypoints in enumerate(pred_keypoints_3d):
                    for j, keypoint in enumerate(keypoints):
                        pose = Pose()
                        pose.position.x = keypoint[0]
                        pose.position.y = keypoint[1]
                        pose.position.z = keypoint[2]
                        pose.orientation.x = quats[i][1]
                        pose.orientation.y = quats[i][2]
                        pose.orientation.z = quats[i][3]
                        pose.orientation.w = quats[i][0]
                        pose_array.poses.append(pose)
        else:
            raise ValueError("Invalid mocap model")

        # return detection_results, pose_array, vis_im
        return detection_results, vis_im

    def __del__(self):
        self.display.stop()
