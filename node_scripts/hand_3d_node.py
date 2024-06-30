#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import message_filters as mf
import tf
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped
from image_geometry import PinholeCameraModel
from hand_object_detection_ros.msg import MocapDetectionArray
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from hand_object_detection_ros.utils import MANO_KEYPOINT_NAMES, axes_to_quaternion


class Hand3DNode(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.scale = rospy.get_param("~scale", 0.001)
        self.slop = rospy.get_param("~slop", 0.15)

        # Subscribe to the camera info and depth image topics
        self.info_sub = mf.Subscriber("~camera_info", CameraInfo, buff_size=2**24)
        self.depth_sub = mf.Subscriber("~input_depth", Image, buff_size=2**24)
        self.hand_sub = mf.Subscriber("~input_detections", MocapDetectionArray, buff_size=2**24)
        self.ts = mf.ApproximateTimeSynchronizer(
            [self.info_sub, self.depth_sub, self.hand_sub], queue_size=1, slop=self.slop
        )
        self.ts.registerCallback(self.callback)


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
            point_3d = (
                detection.pose.position.x,
                detection.pose.position.y,
                detection.pose.position.z,
            )
            point_2d = camera_model.project3dToPixel(point_3d)
            # clip
            point_2d = (
                min(max(point_2d[0], 0), cv_image.shape[1] - 1),
                min(max(point_2d[1], 0), cv_image.shape[0] - 1),
            )
            depth = cv_image[int(point_2d[1]), int(point_2d[0])]
            if np.isnan(depth) or (depth == 0.0):
                continue

            # Calculate 3D coordinates in the camera frame
            x_cam = (point_2d[0] - camera_model.cx()) * depth / camera_model.fx()
            y_cam = (point_2d[1] - camera_model.cy()) * depth / camera_model.fy()
            z_cam = depth

            try:
                # Create PoseArray message and send transform of predicted hand pose without depth calibration
                # publish wrist pose first
                self.tf_broadcaster.sendTransform(
                    (detection.pose.position.x, detection.pose.position.y, detection.pose.position.z),
                    (detection.pose.orientation.x, detection.pose.orientation.y, detection.pose.orientation.z, detection.pose.orientation.w),
                    rospy.Time.now(),
                    detection.label + "_"+ MANO_KEYPOINT_NAMES[0],
                    camera_frame,
                )
                for i, bone in enumerate(detection.skeleton.bones):
                    # Broadcast keypoints in the camera frame
                    self.tf_broadcaster.sendTransform(
                        (bone.end_point.x, bone.end_point.y, bone.end_point.z),
                        (detection.pose.orientation.x, detection.pose.orientation.y, detection.pose.orientation.z, detection.pose.orientation.w),
                        rospy.Time.now(),
                        detection.label + "_" + MANO_KEYPOINT_NAMES[i+1],
                        camera_frame,
                    )

                # Calibrate wrist pose using real 3D coordinates of wrist
                pose_msg = Pose() # wrist pose
                pose_msg.position.x = x_cam * self.scale
                pose_msg.position.y = y_cam * self.scale
                pose_msg.position.z = z_cam * self.scale
                pose_msg.orientation = detection.pose.orientation

                # Broadcast hand pose in the camera frame
                self.tf_broadcaster.sendTransform(
                    (pose_msg.position.x, pose_msg.position.y, pose_msg.position.z),
                    (pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w),
                    rospy.Time.now(),
                    "calibrated/" + detection.label + "_"+ MANO_KEYPOINT_NAMES[0],
                    camera_frame,
                )

                # calibrate skeleton keypoints with wrist pose
                for i, bone in enumerate(detection.skeleton.bones):
                    bone.start_point.x += pose_msg.position.x - detection.pose.position.x
                    bone.start_point.y += pose_msg.position.y - detection.pose.position.y
                    bone.start_point.z += pose_msg.position.z - detection.pose.position.z
                    bone.end_point.x += pose_msg.position.x - detection.pose.position.x
                    bone.end_point.y += pose_msg.position.y - detection.pose.position.y
                    bone.end_point.z += pose_msg.position.z - detection.pose.position.z
                    self.tf_broadcaster.sendTransform(
                        (bone.end_point.x, bone.end_point.y, bone.end_point.z),
                        (pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w),
                        rospy.Time.now(),
                        "calibrated/" + detection.label + "_" + MANO_KEYPOINT_NAMES[i+1],
                        camera_frame,
                    )

                # Broadcast calibrated hand pose in the camera frame
                # avg position of thumb and index finger keypoints is the gripper tool frame
                # x-axis is avg vector of thumb2->thumb3 and index2->index3
                # y-axis is direction from thumb to index finger when left hand and from index to thumb when right hand
                # z-axis is perpendicular to x and y axis with right hand rule
                gripper_pose_msg = PoseStamped()
                bone_names = detection.skeleton.bone_names
                thumb_tip_bone = detection.skeleton.bones[bone_names.index("thumb2->thumb3")]
                index_tip_bone = detection.skeleton.bones[bone_names.index("index2->index3")]

                # position of gripper tool frame
                gripper_pose_msg.header.stamp = rospy.Time.now()
                gripper_pose_msg.header.frame_id = camera_frame
                gripper_pose_msg.pose.position.x = (thumb_tip_bone.end_point.x + index_tip_bone.end_point.x) / 2
                gripper_pose_msg.pose.position.y = (thumb_tip_bone.end_point.y + index_tip_bone.end_point.y) / 2
                gripper_pose_msg.pose.position.z = (thumb_tip_bone.end_point.z + index_tip_bone.end_point.z) / 2

                thumb2to3 = np.array([thumb_tip_bone.end_point.x - thumb_tip_bone.start_point.x, thumb_tip_bone.end_point.y - thumb_tip_bone.start_point.y, thumb_tip_bone.end_point.z - thumb_tip_bone.start_point.z])
                index2to3 = np.array([index_tip_bone.end_point.x - index_tip_bone.start_point.x, index_tip_bone.end_point.y - index_tip_bone.start_point.y, index_tip_bone.end_point.z - index_tip_bone.start_point.z])
                x_axis = (thumb2to3 + index2to3) / 2
                norm_x = np.linalg.norm(x_axis)
                x_axis = x_axis / norm_x

                if detection.label == "right_hand":
                    y_axis = np.array([thumb_tip_bone.end_point.x - index_tip_bone.end_point.x, thumb_tip_bone.end_point.y - index_tip_bone.end_point.y, thumb_tip_bone.end_point.z - index_tip_bone.end_point.z])
                    norm_y = np.linalg.norm(y_axis)
                    y_axis = y_axis / norm_y
                else:
                    y_axis = np.array([index_tip_bone.end_point.x - thumb_tip_bone.end_point.x, index_tip_bone.end_point.y - thumb_tip_bone.end_point.y, index_tip_bone.end_point.z - thumb_tip_bone.end_point.z])
                    norm_y = np.linalg.norm(y_axis)
                    y_axis = y_axis / norm_y

                z_axis = np.cross(x_axis, y_axis)
                norm_z = np.linalg.norm(z_axis)
                z_axis = z_axis / norm_z

                gripper_orientation = axes_to_quaternion(x_axis, y_axis, z_axis)
                gripper_pose_msg.pose.orientation.x = gripper_orientation[1]
                gripper_pose_msg.pose.orientation.y = gripper_orientation[2]
                gripper_pose_msg.pose.orientation.z = gripper_orientation[3]
                gripper_pose_msg.pose.orientation.w = gripper_orientation[0]

                # broadcast gripper tool frame
                self.tf_broadcaster.sendTransform(
                    (gripper_pose_msg.pose.position.x, gripper_pose_msg.pose.position.y, gripper_pose_msg.pose.position.z),
                    (gripper_pose_msg.pose.orientation.x, gripper_pose_msg.pose.orientation.y, gripper_pose_msg.pose.orientation.z, gripper_pose_msg.pose.orientation.w),
                    rospy.Time.now(),
                    "calibrated/" + detection.label + "_gripper_tool_frame",
                    camera_frame,
                )

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr(e)


if __name__ == "__main__":
    rospy.init_node("hand_3d_node")
    node = Hand3DNode()
    rospy.loginfo("Hand 3D node started")
    rospy.spin()
