#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import Image, CameraInfo
from hand_object_detection_ros.msg import HandDetectionArray

from hand_object_detection_ros.detector_wrapper import HandObjectDetectionModel
from hand_object_detection_ros.mocap_wrapper import MocapModel


class HandObjectDetectionNode(object):
    def __init__(self):
        self.device = rospy.get_param("~device", "cuda:0")
        self.hand_threshold = rospy.get_param("~hand_threshold", 0.9)
        self.object_threshold = rospy.get_param("~object_threshold", 0.9)
        self.margin = rospy.get_param("~margin", 10)
        self.camera_info = rospy.wait_for_message("~camera_info", CameraInfo)
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(self.camera_info)
        self.img_size = (self.camera_info.width, self.camera_info.height)

        self.detector_config = dict()
        self.detector_config["detector_model"] = rospy.get_param("~detector_model", "hand_object_detector")

        self.with_handmocap = rospy.get_param("~with_handmocap", True)
        if self.with_handmocap:
            self.mocap_model = rospy.get_param("~mocap_model", "frankmocap") # frankmocap, hamer
            self.mocap_config = dict()
            self.mocap_config["mocap_model"] = self.mocap_model
            if self.mocap_model == "frankmocap":
                self.render_type = rospy.get_param("~render_type", "opengl")  # pytorch3d, opendr, opengl
                self.mocap_config["render_type"] = self.render_type
            elif self.mocap_model == "hamer":
                self.rescale_factor = rospy.get_param("~rescale_factor", 2.0)
                self.mocap_config["focal_length"] = self.camera_model.fx()
                self.mocap_config["rescale_factor"] = self.rescale_factor
            self.visualize = rospy.get_param("~visualize", True)
        self.bridge = CvBridge()
        self.init_model()
        self.sub = rospy.Subscriber("~input_image", Image, self.callback_image, queue_size=1, buff_size=2**24)
        self.pub_debug_image = rospy.Publisher("~debug_image", Image, queue_size=1)
        self.pub_hand_detections = rospy.Publisher("~hand_detections", HandDetectionArray, queue_size=1)

    def init_model(self):
        self.hand_object_detection_model = HandObjectDetectionModel(
            detector_config=self.detector_config,
            device=self.device,
            hand_threshold=self.hand_threshold,
            object_threshold=self.object_threshold,
            margin=self.margin
        )


        if self.with_handmocap:
            self.hand_mocap = MocapModel(
                model_config=self.mocap_config,
                img_size=self.img_size,
                visualize=self.visualize,
                device=self.device,
            )

    def callback_image(self, msg):
        im = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        detection_results, vis_im = self.hand_object_detection_model.predict(im)
        detection_results.header = msg.header

        if self.with_handmocap:
            detection_results, vis_im = self.hand_mocap.predict(detection_results, im, vis_im)

        vis_msg = self.bridge.cv2_to_imgmsg(vis_im.astype(np.uint8), encoding="rgb8")
        vis_msg.header = msg.header
        self.pub_debug_image.publish(vis_msg)
        self.pub_hand_detections.publish(detection_results)



if __name__ == "__main__":
    rospy.init_node("hand_object_detection_node")
    node = HandObjectDetectionNode()
    rospy.loginfo("Hand Object Detection node started")
    rospy.spin()
