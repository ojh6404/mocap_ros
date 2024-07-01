#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped
from hand_object_detection_ros.msg import MocapDetectionArray

from hand_object_detection_ros.detector_wrapper import DetectionModelFactory
from hand_object_detection_ros.mocap_wrapper import MocapModelFactory
from hand_object_detection_ros.utils import axes_to_quaternion, SPIN_KEYPOINT_NAMES, MANO_KEYPOINT_NAMES


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
            self.mocap = rospy.get_param("~mocap_model", "hamer")  # frankmocap_hand, hamer, 4d-human
            if self.mocap == "frankmocap_hand":
                self.mocap_config = {
                    "render_type": rospy.get_param("~render_type", "opengl"),  # pytorch3d, opendr, opengl
                    "img_size": self.img_size,
                    "visualize": rospy.get_param("~visualize", True),
                    "device": self.device,
                }
                self.keypoint_names = MANO_KEYPOINT_NAMES
            elif self.mocap == "hamer":
                self.mocap_config = {
                    "focal_length": self.camera_model.fx(),
                    "rescale_factor": rospy.get_param("~rescale_factor", 2.0),
                    "img_size": self.img_size,
                    "visualize": rospy.get_param("~visualize", True),
                    "device": self.device,
                }
                self.keypoint_names = MANO_KEYPOINT_NAMES
            elif self.mocap == "4d-human":
                self.mocap_config = {
                    "focal_length": self.camera_model.fx(),
                    "rescale_factor": rospy.get_param("~rescale_factor", 2.0),
                    "img_size": self.img_size,
                    "visualize": rospy.get_param("~visualize", True),
                    "device": self.device,
                }
                self.keypoint_names = SPIN_KEYPOINT_NAMES
            else:
                raise ValueError(f"Invalid mocap model: {self.mocap}")

        # Initialize models and ROS
        self.bridge = CvBridge()
        self.init_model()
        self.sub = rospy.Subscriber("~input_image", Image, self.callback_image, queue_size=1, buff_size=2**24)
        self.pub_debug_image = rospy.Publisher("~debug_image", Image, queue_size=1)
        self.pub_detections = rospy.Publisher("~detections", MocapDetectionArray, queue_size=1)

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
        im = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        detections, vis_im = self.detection_model.predict(im)
        detections.header = msg.header

        if self.with_mocap:
            detections, vis_im = self.mocap_model.predict(detections, im, vis_im)



        vis_msg = self.bridge.cv2_to_imgmsg(vis_im.astype(np.uint8), encoding="rgb8")
        vis_msg.header = msg.header
        self.pub_debug_image.publish(vis_msg)
        self.pub_detections.publish(detections)

        if self.publish_tf:
            for i, detection in enumerate(detections.detections):
                try:
                    # publish pose in the camera frame
                    self.tf_broadcaster.sendTransform(
                        (detection.pose.position.x, detection.pose.position.y, detection.pose.position.z),
                        (
                            detection.pose.orientation.x,
                            detection.pose.orientation.y,
                            detection.pose.orientation.z,
                            detection.pose.orientation.w,
                        ),
                        rospy.Time.now(),
                        detection.label + "_" + str(i) + "/pose"
                        msg.header.frame_id,
                    )
                    for i, bone in enumerate(detection.skeleton.bones):
                        # Broadcast keypoints in the camera frame
                        self.tf_broadcaster.sendTransform(
                            (bone.end_point.x, bone.end_point.y, bone.end_point.z),
                            (
                                detection.pose.orientation.x,
                                detection.pose.orientation.y,
                                detection.pose.orientation.z,
                                detection.pose.orientation.w,
                            ),
                            rospy.Time.now(),
                            detection.label + "_" + str(i) + "/" + self.keypoint_names[i],
                            msg.header.frame_id,
                        )
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                    rospy.logerr(e)


if __name__ == "__main__":
    rospy.init_node("detection_node")
    node = DetectionNode()
    rospy.loginfo("Detection node started")
    rospy.spin()
