#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import rospkg
import rospy
import torch
import torch.nn as nn
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point
from jsk_recognition_msgs.msg import Segment, HumanSkeleton
from mocap_ros.msg import HandDetection, HandDetectionArray

ACR_PATH = rospkg.RosPack().get_path("mocap_ros") + "/Arbitrary-Hands-3D-Reconstruction"
sys.path.insert(0, ACR_PATH)

import acr.config as config
from acr.config import args, parse_args, ConfigContext
from acr.utils import *
from acr.utils import justify_detection_state, reorganize_results, img_preprocess
from acr.visualization import Visualizer

if args().model_precision == "fp16":
    from torch.cuda.amp import autocast

from acr.model import ACR as ACR_v1
from acr.mano_wrapper import MANOWrapper

from utils import rotation_matrix_to_quaternion, draw_axis, MANO_JOINTS_CONNECTION


class ACRHandMocapNode(object):
    def __init__(self):
        self.device = rospy.get_param("~device", "cuda:0")
        self.hand_bbox_size_thresh = rospy.get_param("~hand_bbox_size_thresh", 50)
        self.renderer = rospy.get_param("~renderer", "pytorch3d")  # pyrender
        self.detection_thresh = rospy.get_param("~detection_thresh", 0.5)
        self.temporal_optimization = rospy.get_param("~temporal_optimization", True)
        self.smooth_coeff = rospy.get_param("~smooth_coeff", 4.0)
        self.visualization = rospy.get_param("~visualization", True)

        args().renderer = self.renderer
        args().centermap_conf_thresh = self.detection_thresh
        args().temporal_optimization = self.temporal_optimization
        args().smooth_coeff = self.smooth_coeff

        self.init_model()
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber("~input_image", Image, self.callback_image, queue_size=1, buff_size=2**24)
        self.pub_debug_image = rospy.Publisher("~debug_image", Image, queue_size=1)
        self.pub_hand_detections = rospy.Publisher("~hand_detections", HandDetectionArray, queue_size=1)

    @torch.no_grad()
    def callback_image(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        outputs = self.single_image_forward(img)
        vis_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if outputs is not None and outputs["detection_flag"]:  # hand detected
            outputs, results = self.process_results(outputs)
            # outputs : ['l_params_maps', 'r_params_maps', 'l_center_map', 'r_center_map', 'l_prior_maps', 'r_prior_maps', 'segms', 'l_params_pred', 'r_params_pred', 'detection_flag', 'params_pred', 'l_centers_pred', 'r_centers_pred', 'l_centers_conf', 'r_centers_conf', 'left_hand_num', 'right_hand_num', 'reorganize_idx', 'detection_flag_cache', 'output_hand_type', 'params_dict', 'meta_data', 'verts', 'j3d', 'verts_camed', 'pj2d', 'cam_trans', 'pj2d_org']
            # print("j3d", one_hand_result['j3d']) # shape: (21, 3)
            # print("pj2d_org", one_hand_result['pj2d_org']) # shape: (21, 2)

            # visualization: render mesh to image
            if self.visualization:
                show_items_list = ["mesh"]  # ['org_img', 'mesh', 'pj2d', 'centermap']
                results_dict, _ = self.visualizer.visulize_result_live(
                    outputs,
                    img,
                    outputs["meta_data"],
                    show_items=show_items_list,
                    vis_cfg={"settings": ["put_org"]},
                    save2html=False,
                )
                vis_img = results_dict["mesh_rendering_orgimgs"]["figs"][0]

            hand_detections = HandDetectionArray()
            hand_detections.header = msg.header
            for result in results:  # for each hand
                pj2d_org = result["pj2d_org"]
                hand_orientation = result["poses"][:3].astype(np.float32)  # angle-axis representation
                hand_origin = np.array([pj2d_org[0, 0], pj2d_org[0, 1], 0])

                # hand bbox
                hand_left_x = np.min(pj2d_org[:, 0])
                hand_right_x = np.max(pj2d_org[:, 0])
                hand_top_y = np.min(pj2d_org[:, 1])
                hand_bottom_y = np.max(pj2d_org[:, 1])

                hand_bbox_size = np.sqrt((hand_right_x - hand_left_x) * (hand_bottom_y - hand_top_y))
                if hand_bbox_size < self.hand_bbox_size_thresh:  # filter mis-detected hand
                    continue

                hand_pose = Pose()
                hand_pose.position.x = hand_origin[0]
                hand_pose.position.y = hand_origin[1]
                hand_pose.position.z = 0  # cause it's working in 2D

                j3d = result["j3d"]
                hand_skeleton = HumanSkeleton(header=msg.header)
                hand_skeleton.bone_names = []
                hand_skeleton.bones = []
                for i, (start, end) in enumerate(MANO_JOINTS_CONNECTION):
                    bone = Segment()
                    bone.start_point = Point(x=j3d[start][0], y=j3d[start][1], z=j3d[start][2])
                    bone.end_point = Point(x=j3d[end][0], y=j3d[end][1], z=j3d[end][2])
                    hand_skeleton.bones.append(bone)
                    hand_skeleton.bone_names.append(f"bone_{i}")

                rotation, _ = cv2.Rodrigues(hand_orientation)
                quat = rotation_matrix_to_quaternion(rotation)  # [w, x, y, z]
                # aligning x-axis to finger direction
                if result["hand_type"] == 1:  # right hand
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
                vis_img = draw_axis(vis_img, hand_origin, x_axis_rotated, (0, 0, 255))  # x: red
                vis_img = draw_axis(vis_img, hand_origin, y_axis_rotated, (0, 255, 0))  # y: green
                vis_img = draw_axis(vis_img, hand_origin, z_axis_rotated, (255, 0, 0))  # z: blue

                hand_detection = HandDetection()
                hand_detection.hand = "left_hand" if result["hand_type"] == 0 else "right_hand"
                hand_detection.state = "N"  # TODO dummy
                hand_detection.score = 1.0  # TODO dummy
                hand_detection.pose = hand_pose
                hand_detection.skeleton = hand_skeleton
                hand_detections.detections.append(hand_detection)
            if len(hand_detections.detections) > 0:
                self.pub_hand_detections.publish(hand_detections)
        else:  # not detected
            pass
        vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="rgb8")
        vis_msg.header = msg.header
        self.pub_debug_image.publish(vis_msg)

    def init_model(self):
        init_args = ["--demo_mode", "webcam", "-t"]
        with ConfigContext(parse_args(init_args)) as args_set:
            print("Loading the configurations from {}".format(args_set.configs_yml))
            self.demo_cfg = {"mode": "parsing", "calc_loss": False}
            self.project_dir = config.project_dir
            self._initialize_(vars(args() if args_set is None else args_set))

        self.visualizer = Visualizer(resolution=(self.render_size, self.render_size), renderer_type=self.renderer)
        self._build_model_()

    def _initialize_(self, config_dict):
        # configs
        hparams_dict = {}
        for i, j in config_dict.items():
            setattr(self, i, j)
            hparams_dict[i] = j

        # optimizations parameters
        if self.temporal_optimization:
            self.filter_dict = {}
            self.filter_dict[0] = create_OneEuroFilter(args().smooth_coeff)
            self.filter_dict[1] = create_OneEuroFilter(args().smooth_coeff)

        return hparams_dict

    def _build_model_(self):
        model = ACR_v1().eval()
        model = load_model(
            os.path.join(self.project_dir, self.model_path), model, prefix="module.", drop_prefix="", fix_loaded=False
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.mano_regression = MANOWrapper().to(self.device)

    @torch.no_grad()
    def process_results(self, outputs):
        # temporal optimization
        if self.temporal_optimization:
            out_hand = []  # [0],[1],[0,1]
            for idx, i in enumerate(outputs["detection_flag_cache"]):
                if i:
                    out_hand.append(idx)  # idx is also hand type, 0 for left, 1 for right
                else:
                    out_hand.append(-1)
            assert len(outputs["params_dict"]["poses"]) == 2
            for sid, tid in enumerate(out_hand):
                if tid == -1:
                    continue
                outputs["params_dict"]["poses"][sid], outputs["params_dict"]["betas"][sid] = smooth_results(
                    self.filter_dict[tid], outputs["params_dict"]["poses"][sid], outputs["params_dict"]["betas"][sid]
                )
        outputs = self.mano_regression(outputs, outputs["meta_data"])
        reorganize_idx = outputs["reorganize_idx"].cpu().numpy()
        new_results = reorganize_results(outputs, outputs["meta_data"]["imgpath"], reorganize_idx)
        return outputs, new_results["0"]

    @torch.no_grad()
    def single_image_forward(self, bgr_frame):
        meta_data = img_preprocess(bgr_frame, "0", input_size=args().input_size, single_img_input=True)
        ds_org, imgpath_org = get_remove_keys(meta_data, keys=["data_set", "imgpath"])
        meta_data["batch_ids"] = torch.arange(len(meta_data["image"]))
        if self.model_precision == "fp16":
            with autocast():
                outputs = self.model(meta_data, **self.demo_cfg)
        else:
            outputs = self.model(meta_data, **self.demo_cfg)
        outputs["detection_flag"], outputs["reorganize_idx"] = justify_detection_state(
            outputs["detection_flag"], outputs["reorganize_idx"]
        )
        meta_data.update({"imgpath": imgpath_org, "data_set": ds_org})
        outputs["meta_data"]["imgpath"] = ["0"]
        return outputs


if __name__ == "__main__":
    rospy.init_node("acr_hand_mocap_node")
    node = ACRHandMocapNode()
    rospy.spin()
