#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import Image, CameraInfo
from hand_object_detection_ros.msg import MocapDetectionArray

from hand_object_detection_ros.detector_wrapper import DetectionModelFactory
from hand_object_detection_ros.mocap_wrapper import MocapModelFactory


class DetectionNode(object):
    def __init__(self):
        self.device = rospy.get_param("~device", "cuda:0")
        self.camera_info = rospy.wait_for_message("~camera_info", CameraInfo)
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(self.camera_info)
        self.img_size = (self.camera_info.width, self.camera_info.height)

        # Detector
        self.detector = rospy.get_param("~detector_model", "hand_object_detector") # hand_object_detector, mediapipe_hand
        if self.detector == "hand_object_detector":
            self.detector_config = {
                'threshold': rospy.get_param("~threshold", 0.9),
                'object_threshold': rospy.get_param("~object_threshold", 0.9),
                'margin': rospy.get_param("~margin", 10),
                'device': self.device,
            }
        elif self.detector == "mediapipe_hand":
            self.detector_config = {
                'threshold': rospy.get_param("~threshold", 0.9),
                'margin': rospy.get_param("~margin", 10),
                'device': self.device,
            }
        elif self.detector == "yolo":
            self.detector_config = {
                'margin': rospy.get_param("~margin", 10),
                'threshold': rospy.get_param("~threshold", 0.9),
                'device': self.device,
            }
        else:
            raise ValueError(f"Invalid detector model: {self.detector}")

        # Mocap
        self.with_mocap = rospy.get_param("~with_mocap", True)
        if self.with_mocap:
            self.mocap = rospy.get_param("~mocap_model", "hamer") # frankmocap_hand, hamer, 4d-human
            if self.mocap == "frankmocap_hand":
                self.mocap_config = {
                    "render_type": rospy.get_param("~render_type", "opengl"),  # pytorch3d, opendr, opengl
                    "img_size": self.img_size,
                    "visualize": rospy.get_param("~visualize", True),
                    "device": self.device,
                }
            elif self.mocap == "hamer":
                self.mocap_config = {
                    "focal_length": self.camera_model.fx(),
                    "rescale_factor": rospy.get_param("~rescale_factor", 2.0),
                    "img_size": self.img_size,
                    "visualize": rospy.get_param("~visualize", True),
                    "device": self.device,
                }
            elif self.mocap == "4d-human":
                self.mocap_config = {
                    "focal_length": self.camera_model.fx(),
                    "rescale_factor": rospy.get_param("~rescale_factor", 2.0),
                    "img_size": self.img_size,
                    "visualize": rospy.get_param("~visualize", True),
                    "device": self.device,
                }
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
        detection_results, vis_im = self.detection_model.predict(im)
        detection_results.header = msg.header

        if self.with_mocap:
            detection_results, vis_im = self.mocap_model.predict(detection_results, im, vis_im)

        vis_msg = self.bridge.cv2_to_imgmsg(vis_im.astype(np.uint8), encoding="rgb8")
        vis_msg.header = msg.header
        self.pub_debug_image.publish(vis_msg)
        self.pub_detections.publish(detection_results)



if __name__ == "__main__":
    rospy.init_node("detection_node")
    node = DetectionNode()
    rospy.loginfo("Detection node started")
    rospy.spin()
