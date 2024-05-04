import rospkg
import numpy as np
import cv2

# hand object detector and frankmocap paths
FRANKMOCAP_PATH = rospkg.RosPack().get_path("hand_object_detection_ros") + "/frankmocap"
HAND_OBJECT_MODEL_PATH = FRANKMOCAP_PATH + "/detectors/hand_object_detector"
FONT_PATH = HAND_OBJECT_MODEL_PATH + "/lib/model/utils/times_b.ttf"
CHECKPOINT_FILE = FRANKMOCAP_PATH + "/extra_data/hand_module/hand_detector/faster_rcnn_1_8_132028.pth"
FRANKMOCAP_CHECKPOINT = FRANKMOCAP_PATH + "/extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
SMPL_DIR = FRANKMOCAP_PATH + "/extra_data/smpl/"
PASCAL_CLASSES = np.asarray(["__background__", "targetobject", "hand"])  # for hand object detector

# hamer paths
HAMER_ROOT = rospkg.RosPack().get_path("hand_object_detection_ros") + "/hamer"
HAMER_CHECKPOINT_PATH = HAMER_ROOT + "/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
HAMER_CONFIG_PATH = HAMER_ROOT + "/_DATA/hamer_ckpts/model_config.yaml"
HAND_COLOR = (0.65098039,  0.74117647,  0.85882353)

# hand constants
FINGER_JOINTS_CONNECTION = [
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

# for calculating hand origin
PALM_JOINTS = [0, 2, 5, 9, 13, 17]
WEIGHTS = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])


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


def load_hamer(checkpoint_path, config_path):
    from hamer.configs import get_config
    from hamer.models import HAMER

    model_cfg = get_config(config_path)
    model_cfg.defrost()
    model_cfg.MANO.MODEL_PATH = HAMER_ROOT + "/_DATA/data/mano"
    model_cfg.MANO.MEAN_PARAMS = HAMER_ROOT + "/_DATA/data/mano_mean_params.npz"
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
