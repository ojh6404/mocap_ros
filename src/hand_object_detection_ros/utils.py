import rospkg
import torch
import numpy as np
import cv2
import pyrender
import trimesh
from scipy.spatial.transform import Rotation as R

# hand object detector and frankmocap paths
FRANKMOCAP_PATH = rospkg.RosPack().get_path("hand_object_detection_ros") + "/frankmocap"
HAND_OBJECT_MODEL_PATH = FRANKMOCAP_PATH + "/detectors/hand_object_detector/lib"
FONT_PATH = HAND_OBJECT_MODEL_PATH + "/model/utils/times_b.ttf"
CHECKPOINT_FILE = FRANKMOCAP_PATH + "/extra_data/hand_module/hand_detector/faster_rcnn_1_8_132028.pth"
FRANKMOCAP_CHECKPOINT = FRANKMOCAP_PATH + "/extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
SMPL_DIR = FRANKMOCAP_PATH + "/extra_data/smpl/"
PASCAL_CLASSES = np.asarray(["__background__", "targetobject", "hand"])  # for hand object detector

# hamer paths
HAMER_ROOT = rospkg.RosPack().get_path("hand_object_detection_ros") + "/hamer"
HAMER_CHECKPOINT_PATH = HAMER_ROOT + "/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
HAMER_CONFIG_PATH = HAMER_ROOT + "/_DATA/hamer_ckpts/model_config.yaml"
HAND_COLOR = (0.65098039, 0.74117647, 0.85882353)

# hand constants
MANO_KEYPOINT_NAMES = [
    "wrist",
    "thumb0",
    "thumb1",
    "thumb2",
    "thumb3",
    "index0",
    "index1",
    "index2",
    "index3",
    "middle0",
    "middle1",
    "middle2",
    "middle3",
    "ring0",
    "ring1",
    "ring2",
    "ring3",
    "pinky0",
    "pinky1",
    "pinky2",
    "pinky3",
]
MANO_JOINTS_CONNECTION = [
    (0, 1),  # wrist -> thumb0
    (1, 2),  # thumb0 -> thumb1
    (2, 3),  # thumb1 -> thumb2
    (3, 4),  # thumb2 -> thumb3
    (0, 5),  # wrist -> index0
    (5, 6),  # index0 -> index1
    (6, 7),  # index1 -> index2
    (7, 8),  # index2 -> index3
    (0, 9),  # wrist -> middle0
    (9, 10),  # middle0 -> middle1
    (10, 11),  # middle1 -> middle2
    (11, 12),  # middle2 -> middle3
    (0, 13),  # wrist -> ring0
    (13, 14),  # ring0 -> ring1
    (14, 15),  # ring1 -> ring2
    (15, 16),  # ring2 -> ring3
    (0, 17),  # wrist -> pinky0
    (17, 18),  # pinky0 -> pinky1
    (18, 19),  # pinky1 -> pinky2
    (19, 20),  # pinky2 -> pinky3
]
MANO_CONNECTION_NAMES = [
    "wrist->thumb0",
    "thumb0->thumb1",
    "thumb1->thumb2",
    "thumb2->thumb3",
    "wrist->index0",
    "index0->index1",
    "index1->index2",
    "index2->index3",
    "wrist->middle0",
    "middle0->middle1",
    "middle1->middle2",
    "middle2->middle3",
    "wrist->ring0",
    "ring0->ring1",
    "ring1->ring2",
    "ring2->ring3",
    "wrist->pinky0",
    "pinky0->pinky1",
    "pinky1->pinky2",
    "pinky2->pinky3",
]

# SMPL constants
SMPL_KEYPOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]
SMPL_JOINTS_CONNECTION = [
    (0, 1),  # pelvis -> left_hip
    (0, 2),  # pelvis -> right_hip
    (0, 3),  # pelvis -> spine1
    (1, 4),  # left_hip -> left_knee
    (2, 5),  # right_hip -> right_knee
    (3, 6),  # spine1 -> spine2
    (4, 7),  # left_knee -> left_ankle
    (5, 8),  # right_knee -> right_ankle
    (6, 9),  # spine2 -> spine3
    (7, 10),  # left_ankle -> left_foot
    (8, 11),  # right_ankle -> right_foot
    (9, 12),  # spine3 -> neck
    (12, 13),  # neck -> left_collar
    (12, 14),  # neck -> right_collar
    (13, 15),  # left_collar -> head
    (14, 16),  # right_collar -> head
    (15, 17),  # head -> left_shoulder
    (16, 18),  # head -> right_shoulder
    (17, 19),  # left_shoulder -> left_elbow
    (18, 20),  # right_shoulder -> right_elbow
    (19, 21),  # left_elbow -> left_wrist
    (20, 22),  # right_elbow -> right_wrist
    (21, 23),  # left_wrist -> left_hand
    (22, 24),  # right_wrist -> right_hand
]
SMPL_CONNECTION_NAMES = [
    "pelvis->left_hip",
    "pelvis->right_hip",
    "pelvis->spine1",
    "left_hip->left_knee",
    "right_hip->right_knee",
    "spine1->spine2",
    "left_knee->left_ankle",
    "right_knee->right_ankle",
    "spine2->spine3",
    "left_ankle->left_foot",
    "right_ankle->right_foot",
    "spine3->neck",
    "neck->left_collar",
    "neck->right_collar",
    "left_collar->head",
    "right_collar->head",
    "head->left_shoulder",
    "head->right_shoulder",
    "left_shoulder->left_elbow",
    "right_shoulder->right_elbow",
    "left_elbow->left_wrist",
    "right_elbow->right_wrist",
    "left_wrist->left_hand",
    "right_wrist->right_hand",
]

# for 4D-Humans, See https://github.com/shubham-goel/4D-Humans/issues/45
SPIN_KEYPOINT_NAMES = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
    # 24 Ground Truth joints (superset of joints from different datasets)
    # 'Right Ankle',
    # 'Right Knee',
    # 'Right Hip',
    # 'Left Hip',
    # 'Left Knee',
    # 'Left Ankle',
    # 'Right Wrist',
    # 'Right Elbow',
    # 'Right Shoulder',
    # 'Left Shoulder',
    # 'Left Elbow',
    # 'Left Wrist',
    # 'Neck (LSP)',
    # 'Top of Head (LSP)',
    # 'Pelvis (MPII)',
    # 'Thorax (MPII)',
    # 'Spine (H36M)',
    # 'Jaw (H36M)',
    # 'Head (H36M)',
]
SPIN_JOINTS_CONNECTION = [
    (0, 1),  # Nose -> Neck
    (1, 2),  # Neck -> RShoulder
    (2, 3),  # RShoulder -> RElbow
    (3, 4),  # RElbow -> RWrist
    (1, 5),  # Neck -> LShoulder
    (5, 6),  # LShoulder -> LElbow
    (6, 7),  # LElbow -> LWrist
    (1, 8),  # Neck -> MidHip
    (8, 9),  # MidHip -> RHip
    (9, 10),  # RHip -> RKnee
    (10, 11),  # RKnee -> RAnkle
    (8, 12),  # MidHip -> LHip
    (12, 13),  # LHip -> LKnee
    (13, 14),  # LKnee -> LAnkle
    (0, 15),  # Nose -> REye
    (0, 16),  # Nose -> LEye
    (15, 17),  # REye -> REar
    (16, 18),  # LEye -> LEar
    (14, 19),  # LAnkle -> LBigToe
    (19, 20),  # LBigToe -> LSmallToe
    (20, 21),  # LSmallToe -> LHeel
    (11, 22),  # RAnkle -> RBigToe
    (22, 23),  # RBigToe -> RSmallToe
    (23, 24),  # RSmallToe -> RHeel
]
SPIN_CONNECTION_NAMES = [
    "Nose->Neck",
    "Neck->RShoulder",
    "RShoulder->RElbow",
    "RElbow->RWrist",
    "Neck->LShoulder",
    "LShoulder->LElbow",
    "LElbow->LWrist",
    "Neck->MidHip",
    "MidHip->RHip",
    "RHip->RKnee",
    "RKnee->RAnkle",
    "MidHip->LHip",
    "LHip->LKnee",
    "LKnee->LAnkle",
    "Nose->REye",
    "Nose->LEye",
    "REye->REar",
    "LEye->LEar",
    "LAnkle->LBigToe",
    "LBigToe->LSmallToe",
    "LSmallToe->LHeel",
    "RAnkle->RBigToe",
    "RBigToe->RSmallToe",
    "RSmallToe->RHeel",
]


def axes_to_quaternion(x_axis, y_axis, z_axis):
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    r = R.from_matrix(rotation_matrix)
    quaternion = r.as_quat()
    quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
    return quaternion


def rotation_matrix_to_quaternion(R):
    """Convert a rotation matrix to a quaternion.
    Args:
        R (np.ndarray): A 3x3 rotation matrix.
    Returns:
        np.ndarray: A 4-element unit quaternion.
    """
    q = np.empty((4,), dtype=np.float32)
    q[0] = np.sqrt(np.maximum(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    q[1] = np.sqrt(np.maximum(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    q[2] = np.sqrt(np.maximum(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    q[3] = np.sqrt(np.maximum(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    q[1] *= np.sign(q[1] * (R[2, 1] - R[1, 2]))
    q[2] *= np.sign(q[2] * (R[0, 2] - R[2, 0]))
    q[3] *= np.sign(q[3] * (R[1, 0] - R[0, 1]))
    return q


def draw_axis(img, origin, axis, color, scale=20):
    point = origin + scale * axis
    img = cv2.line(img, (int(origin[0]), int(origin[1])), (int(point[0]), int(point[1])), color, 2)
    return img


def load_hamer(checkpoint_path, config_path, img_size, focal_length):
    from hamer.configs import get_config
    from hamer.models import HAMER

    model_cfg = get_config(config_path)
    model_cfg.defrost()
    model_cfg.MANO.MODEL_PATH = HAMER_ROOT + "/_DATA/data/mano"
    model_cfg.MANO.MEAN_PARAMS = HAMER_ROOT + "/_DATA/data/mano_mean_params.npz"
    model_cfg.EXTRA.FOCAL_LENGTH = int(focal_length * model_cfg.MODEL.IMAGE_SIZE / max(img_size))
    model_cfg.freeze()

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
        model_cfg.defrost()
        assert (
            model_cfg.MODEL.IMAGE_SIZE == 256
        ), f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    # Update config to be compatible with demo
    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        model_cfg.freeze()

    model = HAMER.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg


def load_hmr2(checkpoint_path, img_size, focal_length):
    from pathlib import Path
    from hmr2.configs import get_config, CACHE_DIR_4DHUMANS
    from hmr2.models import check_smpl_exists, download_models
    from hmr2.models.hmr2 import HMR2

    download_models(CACHE_DIR_4DHUMANS)

    model_cfg = str(Path(checkpoint_path).parent.parent / "model_config.yaml")
    model_cfg = get_config(model_cfg, update_cachedir=True)
    model_cfg.defrost()
    model_cfg.EXTRA.FOCAL_LENGTH = int(focal_length * model_cfg.MODEL.IMAGE_SIZE / max(img_size))
    model_cfg.freeze()

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
        model_cfg.defrost()
        assert (
            model_cfg.MODEL.IMAGE_SIZE == 256
        ), f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    # Ensure SMPL model exists
    check_smpl_exists()

    model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg


class Renderer(object):
    def __init__(self, faces, cfg, width=640, height=480):
        super(Renderer, self).__init__()
        self.width = width
        self.height = height

        self.focal_length = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(width, height)
        self.camera_center = [self.width / 2.0, self.height / 2.0]
        self.camera_pose = np.eye(4)
        self.camera = pyrender.IntrinsicsCamera(
            fx=self.focal_length, fy=self.focal_length, cx=self.camera_center[0], cy=self.camera_center[1]
        )

        self.lights = self.create_raymond_lights()

        self.faces = faces
        self.faces_left = self.faces[:, [0, 2, 1]]

        self.renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    def vertices_to_trimesh(
        self, vertices, camera_translation, mesh_base_color=HAND_COLOR, rot_axis=[1, 0, 0], rot_angle=0, is_right=1
    ):
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(
                vertices.copy() + camera_translation, self.faces_left.copy(), vertex_colors=vertex_colors
            )

        rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba_multiple(
        self,
        vertices,
        cam_t,
        rot_axis=[1, 0, 0],
        rot_angle=0,
        is_right=None,
    ):
        # Create pyrender scene
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        # Add meshes to the scene
        if is_right is None:
            is_right = [1 for _ in range(len(vertices))]
        mesh_list = [
            pyrender.Mesh.from_trimesh(
                self.vertices_to_trimesh(vvv, ttt.copy(), rot_axis=rot_axis, rot_angle=rot_angle, is_right=sss)
            )
            for vvv, ttt, sss in zip(vertices, cam_t, is_right)
        ]
        for i, mesh in enumerate(mesh_list):
            scene.add(mesh, f"mesh_{i}")

        # Create camera node and add it to pyRender scene
        scene.add(self.camera, pose=self.camera_pose)

        # Add lights to the scene
        for node in self.lights:
            scene.add_node(node)

        rgba, depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        return rgba, depth

    def create_raymond_lights(self):
        """
        Return raymond light nodes for the scene.
        """
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), matrix=matrix))

        return nodes

def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    # Convert cam_bbox to full image
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

def recursive_to(x, target: torch.device):
    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(target)
    elif isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    else:
        return x
