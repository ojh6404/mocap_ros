#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import message_filters as mf
import tf
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray
from image_geometry import PinholeCameraModel
from hand_object_detection_ros.msg import HandDetection, HandDetectionArray
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class Hand3DNode(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.base_frame = rospy.get_param('~base_frame', 'base_footprint') # Target frame
        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()

        # Subscribe to the camera info and depth image topics
        self.info_sub = mf.Subscriber('~camera_info', CameraInfo, buff_size=2**24)
        self.depth_sub = mf.Subscriber('~input_depth', Image, buff_size=2**24)
        self.hand_sub = mf.Subscriber('~input_detections', HandDetectionArray, buff_size=2**24)
        self.ts = mf.ApproximateTimeSynchronizer([self.info_sub, self.depth_sub, self.hand_sub], queue_size=1, slop=0.15)
        self.ts.registerCallback(self.callback)

        # Publisher for Pose
        self.pose_array_pub = rospy.Publisher("~/hand_pose", PoseArray, queue_size=10)


    def callback(self, cam_info_data, depth_data, hand_data):
        # Extract the camera frame from the image message
        camera_frame = depth_data.header.frame_id
        camera_model = PinholeCameraModel()
        camera_model.fromCameraInfo(cam_info_data)

        try:
            # Convert the depth image to a Numpy array
            cv_image = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")
        except CvBridgeError as e:
            rospy.logerr(e)

        for detection in hand_data.detections:
            # Get the depth value at the 2D point
            point_2d = [detection.pose.position.x, detection.pose.position.y]
            depth = cv_image[int(point_2d[1]), int(point_2d[0])]

            # Calculate 3D coordinates in the camera frame
            x_cam = (point_2d[0] - camera_model.cx()) * depth / camera_model.fx()
            y_cam = (point_2d[1] - camera_model.cy()) * depth / camera_model.fy()
            z_cam = depth

            # Transform the point to the base_footprint frame
            try:
                (trans, rot) = self.tf_listener.lookupTransform(self.base_frame, camera_frame, rospy.Time(0))
                transformed_point = tf.transformations.quaternion_matrix(rot)
                transformed_point[0][3] = trans[0]
                transformed_point[1][3] = trans[1]
                transformed_point[2][3] = trans[2]

                point_base = np.dot(transformed_point, np.array([x_cam, y_cam, z_cam, 1]))

                # Create PoseArray message and publish it
                pose_msg = Pose()
                pose_msg.position.x = point_base[0]
                pose_msg.position.y = point_base[1]
                pose_msg.position.z = point_base[2]
                pose_msg.orientation.x = detection.pose.orientation.x
                pose_msg.orientation.y = detection.pose.orientation.y
                pose_msg.orientation.z = detection.pose.orientation.z
                pose_msg.orientation.w = detection.pose.orientation.w

                pose_array_msg = PoseArray()
                pose_array_msg.header.frame_id = self.base_frame
                pose_array_msg.header.stamp = rospy.Time.now()
                pose_array_msg.poses.append(pose_msg)

                self.pose_array_pub.publish(pose_array_msg)

                # Broadcast the transform
                self.tf_broadcaster.sendTransform((pose_msg.position.x, pose_msg.position.y, pose_msg.position.z),
                                                (detection.pose.orientation.x, detection.pose.orientation.y, detection.pose.orientation.z, detection.pose.orientation.w),
                                                rospy.Time.now(),
                                                detection.hand,
                                                self.base_frame)

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr(e)

if __name__ == '__main__':
    rospy.init_node('hand_3d_node')
    node = Hand3DNode()
    rospy.spin()
