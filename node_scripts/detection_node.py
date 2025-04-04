#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Pose, Quaternion
from jsk_recognition_msgs.msg import Rect, HumanSkeleton, Segment
from mocap_ros.msg import Detection, DetectionArray, Mocap, MocapArray

from motion_capture.detector import DetectionModelFactory
from motion_capture.mocap import MocapModelFactory
from motion_capture.utils import (
    MANO_KEYPOINT_NAMES,
    MANO_JOINTS_CONNECTION,
    MANO_CONNECTION_NAMES,
    SPIN_KEYPOINT_NAMES,
    SPIN_JOINTS_CONNECTION,
    SPIN_CONNECTION_NAMES,
)


class DetectionNode(object):
    def __init__(self):
        self.device = rospy.get_param("~device", "cuda:0")
        self.publish_tf = rospy.get_param("~publish_tf", True)
        self.camera_info = rospy.wait_for_message("~camera_info", CameraInfo)
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(self.camera_info)
        self.img_size = (self.camera_info.width, self.camera_info.height)

        if self.publish_tf:
            self.tf_broadcaster = tf.TransformBroadcaster()

        # Detector
        self.detector = rospy.get_param(
            "~detector_model", "hand_object_detector"
        )  # hand_object_detector, mediapipe_hand
        if self.detector == "hand_object_detector":
            self.detector_config = {
                "threshold": rospy.get_param("~threshold", 0.9),
                "object_threshold": rospy.get_param("~object_threshold", 0.9),
                "margin": rospy.get_param("~margin", 10),
                "device": self.device,
            }
        elif self.detector == "mediapipe_hand":
            self.detector_config = {
                "threshold": rospy.get_param("~threshold", 0.9),
                "margin": rospy.get_param("~margin", 10),
                "device": self.device,
            }
        elif self.detector == "yolo":
            self.detector_config = {
                "margin": rospy.get_param("~margin", 10),
                "threshold": rospy.get_param("~threshold", 0.9),
                "device": self.device,
            }
        else:
            raise ValueError(f"Invalid detector model: {self.detector}")

        # Mocap
        self.with_mocap = rospy.get_param("~with_mocap", True)
        if self.with_mocap:
            self.mocap = rospy.get_param("~mocap_model", "wilor")  # hamer, wilor, 4d-human
            if self.mocap == "hamer" or self.mocap == "wilor":
                self.mocap_config = {
                    "focal_length": self.camera_model.fx(),
                    "rescale_factor": rospy.get_param("~rescale_factor", 2.0),
                    "img_size": self.img_size,
                    "visualize": rospy.get_param("~visualize", True),
                    "device": self.device,
                }
                self.keypoint_names = MANO_KEYPOINT_NAMES
                self.connection_names = MANO_CONNECTION_NAMES
                self.joint_connections = MANO_JOINTS_CONNECTION
            elif self.mocap == "4d-human":
                self.mocap_config = {
                    "focal_length": self.camera_model.fx(),
                    "rescale_factor": rospy.get_param("~rescale_factor", 2.0),
                    "img_size": self.img_size,
                    "visualize": rospy.get_param("~visualize", True),
                    "device": self.device,
                }
                self.keypoint_names = SPIN_KEYPOINT_NAMES
                self.connection_names = SPIN_CONNECTION_NAMES
                self.joint_connections = SPIN_JOINTS_CONNECTION
            else:
                raise ValueError(f"Invalid mocap model: {self.mocap}")

        # Initialize models and ROS
        self.bridge = CvBridge()
        self.image = None
        self.frame_id = None
        self.init_model()
        self.sub = rospy.Subscriber("~input_image", Image, self.callback_image, queue_size=1, buff_size=2**24)
        self.pub_debug_image = rospy.Publisher("~debug_image", Image, queue_size=1)
        self.pub_detections = rospy.Publisher("~detections", DetectionArray, queue_size=1)
        if self.with_mocap:
            self.pub_mocaps = rospy.Publisher("~mocaps", MocapArray, queue_size=1)

        while not rospy.is_shutdown():
            if self.image is not None and self.frame_id is not None:
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = self.frame_id
                detections, visualization = self.detection_model.predict(self.image)

                # to DetectionArray msg
                detection_array = DetectionArray(header=header)
                detection_array.detections = [
                    Detection(
                        label=detection.label,
                        score=detection.score,
                        rect=Rect(
                            x=int(detection.rect[0]),
                            y=int(detection.rect[1]),
                            width=int(detection.rect[2] - detection.rect[0]),
                            height=int(detection.rect[3] - detection.rect[1]),
                        ),
                    )
                    for detection in detections
                ]
                self.pub_detections.publish(detection_array)

                if self.with_mocap:
                    mocaps, visualization = self.mocap_model.predict(
                        img=self.image, detections=detections, vis_img=visualization
                    )

                    # to MocapArray msg
                    mocap_array = MocapArray(header=header)
                    mocap_array.mocaps = [
                        Mocap(detection=detection_array.detections[i]) for i in range(len(detection_array.detections))
                    ]

                    for i in range(len(mocaps)):
                        skeleton = HumanSkeleton()
                        skeleton.bone_names = []
                        skeleton.bones = []
                        for j, (start, end) in enumerate(self.joint_connections):
                            bone = Segment()
                            bone.start_point = Point(
                                x=mocaps[i].keypoints[start][0],
                                y=mocaps[i].keypoints[start][1],
                                z=mocaps[i].keypoints[start][2],
                            )
                            bone.end_point = Point(
                                x=mocaps[i].keypoints[end][0],
                                y=mocaps[i].keypoints[end][1],
                                z=mocaps[i].keypoints[end][2],
                            )
                            skeleton.bones.append(bone)
                            skeleton.bone_names.append(self.connection_names[j])

                        quat = R.from_matrix(mocaps[i].orientation).as_quat(scalar_first=False)
                        mocap_array.mocaps[i].pose = Pose(
                            position=Point(
                                x=mocaps[i].position[0],
                                y=mocaps[i].position[1],
                                z=mocaps[i].position[2],
                            ),
                            orientation=Quaternion(
                                x=quat[0],
                                y=quat[1],
                                z=quat[2],
                                w=quat[3],
                            ),
                        )
                        mocap_array.mocaps[i].skeleton = skeleton
                    self.pub_mocaps.publish(mocap_array)

                # Publish visualization
                vis_msg = self.bridge.cv2_to_imgmsg(visualization.astype(np.uint8), encoding="bgr8")
                vis_msg.header = header
                self.pub_debug_image.publish(vis_msg)

                # Publish tf
                if self.publish_tf and self.with_mocap:
                    for mocap in mocap_array.mocaps:
                        try:
                            # publish pose in the camera frame
                            self.tf_broadcaster.sendTransform(
                                (
                                    mocap.pose.position.x,
                                    mocap.pose.position.y,
                                    mocap.pose.position.z,
                                ),
                                (
                                    mocap.pose.orientation.x,
                                    mocap.pose.orientation.y,
                                    mocap.pose.orientation.z,
                                    mocap.pose.orientation.w,
                                ),
                                rospy.Time.now(),
                                mocap.detection.label + "/" + self.keypoint_names[0],
                                header.frame_id,
                            )
                            for bone_name in mocap.skeleton.bone_names:
                                parent_name = bone_name.split("->")[0]
                                child_name = bone_name.split("->")[1]
                                bone_idx = mocap.skeleton.bone_names.index(bone_name)

                                parent_point = mocap.skeleton.bones[bone_idx].start_point
                                child_point = mocap.skeleton.bones[bone_idx].end_point
                                parent_to_child = R.from_quat(
                                    [
                                        mocap.pose.orientation.x,
                                        mocap.pose.orientation.y,
                                        mocap.pose.orientation.z,
                                        mocap.pose.orientation.w,
                                    ]
                                ).inv().as_matrix() @ np.array(
                                    [
                                        child_point.x - parent_point.x,
                                        child_point.y - parent_point.y,
                                        child_point.z - parent_point.z,
                                    ]
                                )  # cause the bone is in the camera frame

                                # broadcast bone pose in tree structure
                                self.tf_broadcaster.sendTransform(
                                    (parent_to_child[0], parent_to_child[1], parent_to_child[2]),
                                    (0, 0, 0, 1),
                                    rospy.Time.now(),
                                    mocap.detection.label + "/" + child_name,
                                    mocap.detection.label + "/" + parent_name,
                                )
                        except (
                            tf.LookupException,
                            tf.ConnectivityException,
                            tf.ExtrapolationException,
                        ) as e:
                            rospy.logerr(e)

    def init_model(self):
        self.detection_model = DetectionModelFactory.from_config(
            model=self.detector,
            model_config=self.detector_config,
        )

        if self.with_mocap:
            self.mocap_model = MocapModelFactory.from_config(
                model=self.mocap,
                model_config=self.mocap_config,
            )

    def callback_image(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.frame_id = msg.header.frame_id


if __name__ == "__main__":
    rospy.init_node("detection_node")
    node = DetectionNode()
    rospy.loginfo("Detection node started")
    rospy.spin()
