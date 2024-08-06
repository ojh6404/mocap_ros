#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script is used to process rosbag files and save the processed results without roscore.
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import argparse
from tqdm import tqdm
import os

import rospy
import rosbag
import message_filters
import tf2_ros
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, TransformStamped, Point, Quaternion
from jsk_recognition_msgs.msg import Rect, Segment, HumanSkeleton
from mocap_ros.msg import Detection, DetectionArray, Mocap, MocapArray

from motion_capture.detector import DetectionModelFactory
from motion_capture.mocap import MocapModelFactory
from motion_capture.utils.utils import (
    MANO_KEYPOINT_NAMES,
    MANO_JOINTS_CONNECTION,
    MANO_CONNECTION_NAMES,
    axes_to_quaternion,
)


bridge = CvBridge()


def main(args):
    # some constants
    device = "cuda:0"
    hand_threshold = 0.9
    object_threshold = 0.9
    margin = 10
    img_size = (640, 480)
    focal_length = args.focal_length
    tmpbag = "/tmp/tmp.bag"

    # Detector config
    detector = args.detector_model  # hand_object_detector, mediapipe_hand
    if detector == "hand_object_detector":
        detector_config = {
            "threshold": hand_threshold,
            "object_threshold": object_threshold,
            "margin": margin,
            "device": device,
        }
    elif detector == "mediapipe_hand":
        detector_config = {
            "threshold": hand_threshold,
            "margin": margin,
            "device": device,
        }
    else:
        raise ValueError(f"Invalid detector model: {detector}")

    # Mocap config
    mocap = args.mocap_model
    if mocap == "frankmocap_hand":
        mocap_config = {
            "render_type": "opengl",  # pytorch3d, opendr, opengl
            "img_size": img_size,
            "visualize": True,
            "device": device,
        }
        keypoint_names = MANO_KEYPOINT_NAMES
    elif mocap == "hamer":
        mocap_config = {
            "focal_length": focal_length,
            "rescale_factor": 2.0,
            "img_size": img_size,
            "visualize": True,
            "device": device,
        }
        keypoint_names = MANO_KEYPOINT_NAMES
    else:
        raise ValueError(f"Invalid mocap model: {mocap}")

    # Initialize models
    detection_model = DetectionModelFactory.from_config(
        model=detector,
        model_config=detector_config,
    )
    mocap_model = MocapModelFactory.from_config(
        model=mocap,
        model_config=mocap_config,
    )

    # ================== Process rosbag ================== #
    with rosbag.Bag(tmpbag, "w") as outbag:
        topics = [
            "/kinect_head/rgb/image_rect_color/compressed",
            "/kinect_head/rgb/camera_info",
        ]
        msgs = [CompressedImage, CameraInfo]
        if args.calibrate:
            topics += [
                "/kinect_head/depth_registered/image_raw",
            ]
            msgs += [Image]

        subscribers = {topic: message_filters.Subscriber(topic, msg) for topic, msg in zip(topics, msgs)}
        ts = message_filters.ApproximateTimeSynchronizer(
            subscribers.values(),
            queue_size=1,
            slop=0.1,
            allow_headerless=False,
        )

        def callback(*msgs):
            img_msg = msgs[0]
            cam_info_msg = msgs[1]
            if args.calibrate:
                depth_msg = msgs[2]
                depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            image = (
                bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
                if "Compressed" in img_msg._type
                else bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            )
            camera_model = PinholeCameraModel()
            camera_model.fromCameraInfo(cam_info_msg)
            camera_frame = img_msg.header.frame_id

            # Detect Hands
            detections, vis_im = detection_model.predict(image)

            # to DetectionArray msg
            detection_array = DetectionArray(header=msg.header)
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
            vis_im_msg = bridge.cv2_to_compressed_imgmsg(vis_im, dst_format="jpeg")
            vis_im_msg.header = msg.header
            outbag.write("/mocap/detection_image/compressed", vis_im_msg, msg.header.stamp)

            # Process Mocap
            mocaps, vis_im = mocap_model.predict(detections, image, vis_im)
            # to MocapArray msg
            mocap_array = MocapArray(header=msg.header)
            mocap_array.mocaps = [
                Mocap(detection=detection_array.detections[i]) for i in range(len(detection_array.detections))
            ]

            for i in range(len(mocaps)):
                skeleton = HumanSkeleton()
                skeleton.bone_names = []
                skeleton.bones = []
                for j, (start, end) in enumerate(MANO_JOINTS_CONNECTION):
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
                    skeleton.bone_names.append(MANO_CONNECTION_NAMES[j])

                mocap_array.mocaps[i].pose = Pose(
                    position=Point(
                        x=mocaps[i].position[0],
                        y=mocaps[i].position[1],
                        z=mocaps[i].position[2],
                    ),
                    orientation=Quaternion(
                        x=mocaps[i].orientation[0],
                        y=mocaps[i].orientation[1],
                        z=mocaps[i].orientation[2],
                        w=mocaps[i].orientation[3],
                    ),
                )
                mocap_array.mocaps[i].skeleton = skeleton

            vis_im_msg = bridge.cv2_to_compressed_imgmsg(vis_im, dst_format="jpeg")
            vis_im_msg.header = msg.header

            if args.calibrate:
                for mocap in mocap_array.mocaps:
                    pred_point_3d = (
                        mocap.pose.position.x,
                        mocap.pose.position.y,
                        mocap.pose.position.z,
                    )  # predicted 3d wrist point in camera frame
                    pred_point_2d = camera_model.project3dToPixel(pred_point_3d)
                    # Clip
                    pred_point_2d = (
                        min(max(pred_point_2d[0], 0), depth.shape[1] - 1),
                        min(max(pred_point_2d[1], 0), depth.shape[0] - 1),
                    )
                    depth_z = depth[int(pred_point_2d[1]), int(pred_point_2d[0])]
                    if np.isnan(depth_z) or (depth_z == 0.0):
                        continue

                    # Calculate real 3D point in camera frame using depth
                    x_cam = (pred_point_2d[0] - camera_model.cx()) * depth_z / camera_model.fx() * args.scale
                    y_cam = (pred_point_2d[1] - camera_model.cy()) * depth_z / camera_model.fy() * args.scale
                    z_cam = depth_z * args.scale

                    # Calibrate skeleton keypoints with wrist pose
                    mocap.pose.position.x = x_cam
                    mocap.pose.position.y = y_cam
                    mocap.pose.position.z = z_cam
                    for bone in mocap.skeleton.bones:
                        bone.start_point.x += x_cam - pred_point_3d[0]
                        bone.start_point.y += y_cam - pred_point_3d[1]
                        bone.start_point.z += z_cam - pred_point_3d[2]
                        bone.end_point.x += x_cam - pred_point_3d[0]
                        bone.end_point.y += y_cam - pred_point_3d[1]
                        bone.end_point.z += z_cam - pred_point_3d[2]

            # Write detections and visualization image
            outbag.write("/mocap/hand_mocaps", mocap_array, msg.header.stamp)
            outbag.write("/mocap/mocap_image/compressed", vis_im_msg, msg.header.stamp)

            pose_array_msg = PoseArray()
            pose_array_msg.header.frame_id = camera_frame
            pose_array_msg.header.stamp = img_msg.header.stamp

            # Calibrate PoseArray message
            keypoints_msg = PoseArray()
            keypoints_msg.header.frame_id = camera_frame
            keypoints_msg.header.stamp = img_msg.header.stamp

            for mocap in mocap_array.mocaps:
                # Write hand pose in camera frame
                try:
                    tf_msg = TFMessage()
                    tf_transform = TransformStamped()
                    tf_transform.header.stamp = t
                    tf_transform.header.frame_id = camera_frame
                    tf_transform.child_frame_id = mocap.detection.label + "/" + keypoint_names[0]
                    tf_transform.transform.translation.x = mocap.pose.position.x
                    tf_transform.transform.translation.y = mocap.pose.position.y
                    tf_transform.transform.translation.z = mocap.pose.position.z
                    tf_transform.transform.rotation.x = mocap.pose.orientation.x
                    tf_transform.transform.rotation.y = mocap.pose.orientation.y
                    tf_transform.transform.rotation.z = mocap.pose.orientation.z
                    tf_transform.transform.rotation.w = mocap.pose.orientation.w
                    tf_msg.transforms.append(tf_transform)
                    outbag.write("/tf", tf_msg, t)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    rospy.logerr(e)
                    continue

                # Write keypoints in each bone frame
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
                    try:
                        tf_transform = TransformStamped()
                        tf_transform.header.stamp = t
                        tf_transform.header.frame_id = mocap.detection.label + "/" + parent_name
                        tf_transform.child_frame_id = mocap.detection.label + "/" + child_name
                        tf_transform.transform.translation.x = parent_to_child[0]
                        tf_transform.transform.translation.y = parent_to_child[1]
                        tf_transform.transform.translation.z = parent_to_child[2]
                        tf_transform.transform.rotation.x = 0
                        tf_transform.transform.rotation.y = 0
                        tf_transform.transform.rotation.z = 0
                        tf_transform.transform.rotation.w = 1
                        tf_msg.transforms.append(tf_transform)
                        outbag.write("/tf", tf_msg, t)
                    except (
                        tf2_ros.LookupException,
                        tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException,
                    ) as e:
                        rospy.logerr(e)
                        continue

        ts.registerCallback(callback)

        print("Processing rosbags...")
        bag_reader = rosbag.Bag(args.rosbag, skip_index=True)

        # message filter
        for _, (topic, msg, t) in tqdm(enumerate(bag_reader.read_messages(topics=topics))):
            subscriber = subscribers[topic]
            if subscriber:
                subscriber.signalMessage(msg)

        # write original messages
        print("Writing original messages...")
        with rosbag.Bag(args.rosbag, "r") as input_bag:
            for topic, msg, t in tqdm(input_bag.read_messages()):
                outbag.write(topic, msg, t)

    # Caching TF messages
    tf_buffer = tf2_ros.Buffer(rospy.Duration(1000000.0))
    with rosbag.Bag(tmpbag, "r") as input_bag:
        for topic, msg, t in input_bag.read_messages(topics=["/tf", "/tf_static"]):
            for msg_tf in msg.transforms:
                if topic == "/tf_static":
                    tf_buffer.set_transform_static(msg_tf, "default_authority")
                else:
                    tf_buffer.set_transform(msg_tf, "default_authority")

    # Write keypoints and grasp pose in target frame
    with rosbag.Bag(args.output, "w") as outbag:
        with rosbag.Bag(tmpbag, "r") as input_bag:
            for topic, msg, t in tqdm(input_bag.read_messages()):
                outbag.write(topic, msg, t)
                if topic == "/mocap/hand_mocaps":
                    for mocap in msg.mocaps:
                        try:
                            # Write hand keypoints in target frame
                            pose_array_msg = PoseArray()
                            pose_array_msg.header.stamp = t
                            pose_array_msg.header.frame_id = args.target_frame
                            for keypoint in MANO_KEYPOINT_NAMES:
                                transform = tf_buffer.lookup_transform(
                                    args.target_frame, mocap.detection.label + "/" + keypoint, t
                                )
                                pose_msg = Pose()
                                pose_msg.position.x = transform.transform.translation.x
                                pose_msg.position.y = transform.transform.translation.y
                                pose_msg.position.z = transform.transform.translation.z
                                pose_msg.orientation.x = transform.transform.rotation.x
                                pose_msg.orientation.y = transform.transform.rotation.y
                                pose_msg.orientation.z = transform.transform.rotation.z
                                pose_msg.orientation.w = transform.transform.rotation.w
                                pose_array_msg.poses.append(pose_msg)
                            outbag.write("/mocap/" + mocap.detection.label + "/keypoints", pose_array_msg, t)

                            # create new grasp pose which is the average of thumb and index finger keypoints
                            # avg position of thumb and index finger keypoints is the grasp pose in target frame
                            # x-axis is avg vector of thumb2->thumb3 and index2->index3
                            # y-axis is direction from thumb to index finger when left hand and from index to thumb when right hand
                            # z-axis is perpendicular to x and y axis with right hand rule

                            thumb2_idx = MANO_KEYPOINT_NAMES.index("thumb2")
                            thumb3_idx = MANO_KEYPOINT_NAMES.index("thumb3")
                            index2_idx = MANO_KEYPOINT_NAMES.index("index2")
                            index3_idx = MANO_KEYPOINT_NAMES.index("index3")

                            grasp_pose_msg = PoseStamped()
                            thumb2_keypoint = pose_array_msg.poses[thumb2_idx]
                            thumb3_keypoint = pose_array_msg.poses[thumb3_idx]
                            index2_keypoint = pose_array_msg.poses[index2_idx]
                            index3_keypoint = pose_array_msg.poses[index3_idx]

                            # calculate axes
                            thumb2to3 = np.array(
                                [
                                    thumb3_keypoint.position.x - thumb2_keypoint.position.x,
                                    thumb3_keypoint.position.y - thumb2_keypoint.position.y,
                                    thumb3_keypoint.position.z - thumb2_keypoint.position.z,
                                ]
                            )
                            index2to3 = np.array(
                                [
                                    index3_keypoint.position.x - index2_keypoint.position.x,
                                    index3_keypoint.position.y - index2_keypoint.position.y,
                                    index3_keypoint.position.z - index2_keypoint.position.z,
                                ]
                            )
                            x_axis = (thumb2to3 + index2to3) / 2
                            x_axis = x_axis / np.linalg.norm(x_axis)

                            if mocap.detection.label == "right_hand":
                                y_axis = np.array(
                                    [
                                        thumb3_keypoint.position.x - index3_keypoint.position.x,
                                        thumb3_keypoint.position.y - index3_keypoint.position.y,
                                        thumb3_keypoint.position.z - index3_keypoint.position.z,
                                    ]
                                )
                            else:
                                y_axis = np.array(
                                    [
                                        index3_keypoint.position.x - thumb3_keypoint.position.x,
                                        index3_keypoint.position.y - thumb3_keypoint.position.y,
                                        index3_keypoint.position.z - thumb3_keypoint.position.z,
                                    ]
                                )
                            norm_y = np.linalg.norm(y_axis)
                            y_axis = y_axis / norm_y

                            z_axis = np.cross(x_axis, y_axis)
                            z_axis = z_axis / np.linalg.norm(z_axis)

                            # calculate new grasp pose
                            grasp_orientation = axes_to_quaternion(x_axis, y_axis, z_axis)

                            grasp_pose_msg.header.stamp = t
                            grasp_pose_msg.header.frame_id = args.target_frame
                            # average of thumb3 and index3 finger keypoints
                            grasp_pose_msg.pose.position.x = (
                                thumb3_keypoint.position.x + index3_keypoint.position.x
                            ) / 2
                            grasp_pose_msg.pose.position.y = (
                                thumb3_keypoint.position.y + index3_keypoint.position.y
                            ) / 2
                            grasp_pose_msg.pose.position.z = (
                                thumb3_keypoint.position.z + index3_keypoint.position.z
                            ) / 2
                            grasp_pose_msg.pose.orientation.x = grasp_orientation[1]
                            grasp_pose_msg.pose.orientation.y = grasp_orientation[2]
                            grasp_pose_msg.pose.orientation.z = grasp_orientation[3]
                            grasp_pose_msg.pose.orientation.w = grasp_orientation[0]
                            outbag.write("/mocap/" + mocap.detection.label + "/grasp_pose", grasp_pose_msg, t)

                        except (
                            tf2_ros.LookupException,
                            tf2_ros.ConnectivityException,
                            tf2_ros.ExtrapolationException,
                        ) as e:
                            rospy.logerr(e)
                            continue

    # Remove temporary bag file
    os.remove(tmpbag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process rosbag file with hand object detection and mocap model")
    parser.add_argument("-bag", "--rosbag", type=str, default=None, help="rosbag file to process")
    parser.add_argument("-o", "--output", type=str, default="output.bag", help="output rosbag file or mp4 file")
    parser.add_argument("-d", "--detector_model", type=str, default="hand_object_detector", help="detector model")
    parser.add_argument("-m", "--mocap_model", type=str, default="hamer", help="mocap model")
    parser.add_argument("-f", "--focal_length", type=float, default=525.0, help="focal length of the camera")
    parser.add_argument("-c", "--calibrate", action="store_true", help="calibrate depth with real depth")
    parser.add_argument("-s", "--scale", type=float, default=0.001, help="scale factor for depth calibration")
    parser.add_argument(
        "-sf", "--source_frame", type=str, default="head_mount_kinect_rgb_optical_frame", help="source frame"
    )
    parser.add_argument("-tf", "--target_frame", type=str, default="base_footprint", help="target frame")

    args = parser.parse_args()
    rospy.init_node("rosbag_processor")
    main(args)
