import os
import sys
import time
import json

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import ailia
    HAS_AILIA = True
except ImportError:
    HAS_AILIA = False

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import imread  # noqa
import webcamera_utils  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'bevformer_tiny.onnx'
MODEL_PATH = 'bevformer_tiny.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/bevformer/'

CAMERA_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]
IMAGE_PATH = [f'demo_{name}.jpg' for name in CAMERA_NAMES]
SAVE_IMAGE_PATH = 'output.png'

# Model input dimensions (must match the ONNX model)
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 800
NUM_CAMS = 6

# BEV configuration
BEV_H = 50
BEV_W = 50
POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
NUM_QUERIES = 300
NUM_CLASSES = 10

# ImageNet normalization (BEVFormer-tiny uses RGB with ImageNet mean/std)
IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

THRESHOLD = 0.3
MAX_DETECTIONS = 20

# nuScenes detection classes
NUSCENES_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

# Colors for each class (BGR for OpenCV, RGB for matplotlib)
CLASS_COLORS_BGR = [
    (0, 255, 0),    # car - green
    (0, 165, 255),  # truck - orange
    (0, 255, 255),  # construction_vehicle - yellow
    (255, 0, 0),    # bus - blue
    (128, 0, 128),  # trailer - purple
    (128, 128, 128),  # barrier - gray
    (0, 0, 255),    # motorcycle - red
    (255, 255, 0),  # bicycle - cyan
    (255, 0, 255),  # pedestrian - magenta
    (0, 128, 255),  # traffic_cone - light orange
]

CLASS_COLORS_RGB = [
    tuple(c[::-1]) for c in CLASS_COLORS_BGR
]

# Approximate nuScenes camera parameters for 3D bbox projection
# Ego frame: x=right, y=forward, z=up
# Camera frame: x=right, y=down, z=forward
CAMERA_INTRINSIC = np.array([
    [1266.4, 0, 816.3],
    [0, 1266.4, 491.5],
    [0, 0, 1],
], dtype=np.float64)

# Camera yaw angles (degrees from forward/+y, CCW positive)
CAMERA_YAWS_DEG = [0, 55, -55, 180, 110, -110]

# Camera positions in ego frame [x_right, y_forward, z_up]
CAMERA_POSITIONS = [
    [0.0, 1.7, 1.5],    # CAM_FRONT
    [0.5, 1.7, 1.5],    # CAM_FRONT_LEFT
    [-0.5, 1.7, 1.5],   # CAM_FRONT_RIGHT
    [0.0, 0.0, 1.5],    # CAM_BACK
    [0.5, 1.0, 1.5],    # CAM_BACK_LEFT
    [-0.5, 1.0, 1.5],   # CAM_BACK_RIGHT
]

# Original nuScenes image size (for projection)
NUSCENES_IMG_W = 1600
NUSCENES_IMG_H = 900


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'BEVFormer: Bird\'s-Eye-View 3D Object Detection',
    IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='Detection confidence threshold.'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
parser.add_argument(
    '--onnx', action='store_true',
    help='Use ONNX Runtime instead of ailia SDK.'
)
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================

def preprocess_image(img):
    """Preprocess a single camera image for BEVFormer-tiny.

    Args:
        img: BGR image (H, W, 3)

    Returns:
        preprocessed image (3, IMAGE_HEIGHT, IMAGE_WIDTH) float32
    """
    # Resize to model input size
    img_resized = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Normalize with ImageNet mean/std
    img_norm = (img_rgb - IMG_MEAN) / IMG_STD

    # HWC -> CHW
    img_chw = img_norm.transpose(2, 0, 1).astype(np.float32)

    return img_chw


def prepare_multi_cam_input(imgs):
    """Prepare 6-camera input from a list of images.

    Args:
        imgs: list of BGR images (H, W, 3), length 6

    Returns:
        images: (1, 6, 3, H, W) float32
    """
    cam_imgs = [preprocess_image(img) for img in imgs]
    multi_cam = np.stack(cam_imgs, axis=0)

    # Add batch dimension: (1, 6, 3, H, W)
    multi_cam = np.expand_dims(multi_cam, axis=0)

    return multi_cam


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def decode_bbox_3d(bbox_pred):
    """Decode 3D bounding box prediction.

    Args:
        bbox_pred: (code_size,) raw prediction
            [cx, cy, cz, w, l, h, sin, cos, vx, vy]

    Returns:
        dict with 3D bbox parameters
    """
    cx, cy, cz = bbox_pred[0], bbox_pred[1], bbox_pred[2]
    w, l, h = bbox_pred[3], bbox_pred[4], bbox_pred[5]
    sin_yaw, cos_yaw = bbox_pred[6], bbox_pred[7]
    vx, vy = bbox_pred[8], bbox_pred[9]

    # Convert sin/cos to yaw angle
    yaw = np.arctan2(sin_yaw, cos_yaw)

    # Map normalized center to point cloud range
    pc = POINT_CLOUD_RANGE
    cx = cx * (pc[3] - pc[0]) + pc[0]
    cy = cy * (pc[4] - pc[1]) + pc[1]
    cz = cz * (pc[5] - pc[2]) + pc[2]

    # Exponentiate dimensions
    w = np.exp(np.clip(w, -5, 5))
    l = np.exp(np.clip(l, -5, 5))
    h = np.exp(np.clip(h, -5, 5))

    return {
        'center': [float(cx), float(cy), float(cz)],
        'size': [float(w), float(l), float(h)],
        'yaw': float(yaw),
        'velocity': [float(vx), float(vy)],
    }


def post_process(cls_scores, bbox_preds, threshold):
    """Post-process model outputs to get detections.

    Args:
        cls_scores: (B, num_queries, num_classes)
        bbox_preds: (B, num_queries, code_size)
        threshold: confidence threshold

    Returns:
        list of detection dicts
    """
    cls_scores = cls_scores[0]  # (num_queries, num_classes)
    bbox_preds = bbox_preds[0]  # (num_queries, code_size)

    # Apply sigmoid to classification scores
    scores = sigmoid(cls_scores)

    max_scores = np.max(scores, axis=1)
    max_classes = np.argmax(scores, axis=1)

    # Filter by threshold
    mask = max_scores > threshold
    indices = np.where(mask)[0]

    detections = []
    for idx in indices:
        score = float(max_scores[idx])
        cls_id = int(max_classes[idx])
        bbox = decode_bbox_3d(bbox_preds[idx])

        detections.append({
            'score': score,
            'class_id': cls_id,
            'class_name': NUSCENES_CLASSES[cls_id],
            'bbox_3d': bbox,
        })

    detections.sort(key=lambda x: x['score'], reverse=True)
    detections = detections[:MAX_DETECTIONS]
    return detections


def get_ego_to_cam_rotation(yaw_deg):
    """Get rotation matrix from ego frame to camera frame.

    Args:
        yaw_deg: camera yaw in degrees (from +y/forward, CCW positive)

    Returns:
        3x3 rotation matrix
    """
    theta = np.radians(yaw_deg)
    c, s = np.cos(theta), np.sin(theta)
    # R = R_ego2cam_front @ R_z(-theta)
    return np.array([
        [c, s, 0],
        [0, 0, -1],
        [-s, c, 0],
    ])


def project_3d_to_camera(points_3d, cam_idx):
    """Project 3D points (ego frame) to camera image coordinates.

    Args:
        points_3d: (N, 3) points in ego frame
        cam_idx: camera index (0-5)

    Returns:
        pts_2d: (N, 2) image coordinates
        valid: (N,) boolean mask for points in front of camera and in image
    """
    R = get_ego_to_cam_rotation(CAMERA_YAWS_DEG[cam_idx])
    t = np.array(CAMERA_POSITIONS[cam_idx])

    # Transform to camera frame
    pts_cam = (R @ (points_3d - t).T).T  # (N, 3)

    # Check if points are in front of camera
    depth = pts_cam[:, 2]
    in_front = depth > 0.1

    # Project to image
    pts_2d = np.zeros((len(points_3d), 2))
    if np.any(in_front):
        proj = CAMERA_INTRINSIC @ pts_cam[in_front].T  # (3, M)
        pts_2d[in_front, 0] = proj[0] / proj[2]
        pts_2d[in_front, 1] = proj[1] / proj[2]

    # Check if projected points are within image bounds
    in_image = (
        in_front &
        (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < NUSCENES_IMG_W) &
        (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < NUSCENES_IMG_H)
    )

    return pts_2d, in_image


# 3D box edge pairs (indices into 8 corners)
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
]


def get_3d_box_corners(center, size, yaw):
    """Compute 8 corners of a 3D bounding box."""
    w, l, h = size
    x, y, z = center

    corners = np.array([
        [-l / 2, -w / 2, -h / 2],
        [l / 2, -w / 2, -h / 2],
        [l / 2, w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
        [-l / 2, -w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [l / 2, w / 2, h / 2],
        [-l / 2, w / 2, h / 2],
    ])

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1],
    ])

    corners = corners @ rot.T + np.array([x, y, z])
    return corners


def draw_3d_bbox_on_camera(ax, corners_3d, cam_idx, color, score):
    """Draw projected 3D bounding box on a camera image axis."""
    pts_2d, valid = project_3d_to_camera(corners_3d, cam_idx)

    for i, j in BOX_EDGES:
        if valid[i] and valid[j]:
            ax.plot(
                [pts_2d[i, 0], pts_2d[j, 0]],
                [pts_2d[i, 1], pts_2d[j, 1]],
                color=color, linewidth=1.5, alpha=0.8)

    # Draw label at the top-front center if visible
    top_center_idx = [4, 5, 6, 7]  # top face corners
    visible_top = [k for k in top_center_idx if valid[k]]
    if visible_top:
        label_x = np.mean([pts_2d[k, 0] for k in visible_top])
        label_y = np.min([pts_2d[k, 1] for k in visible_top]) - 5
        ax.text(label_x, label_y, f'{score:.2f}',
                fontsize=5, color=color, ha='center',
                bbox=dict(boxstyle='round,pad=0.1',
                          facecolor='black', alpha=0.5))


def draw_bev_detections(detections, imgs=None):
    """Draw detection results in Bird's Eye View with camera projections.

    Args:
        detections: list of detection dicts
        imgs: list of 6 BGR camera images, or None
    """
    pc = POINT_CLOUD_RANGE

    if imgs is not None:
        # Layout: 3 rows of camera images on left, BEV on right
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 2])

        # Camera layout:
        # Row 0: FRONT_LEFT, FRONT, FRONT_RIGHT
        # Row 1: (empty),    BEV,   (empty)
        # Row 2: BACK_LEFT,  BACK,  BACK_RIGHT
        cam_grid_positions = [
            (0, 1),  # CAM_FRONT
            (0, 0),  # CAM_FRONT_LEFT
            (0, 2),  # CAM_FRONT_RIGHT
            (2, 1),  # CAM_BACK
            (2, 0),  # CAM_BACK_LEFT
            (2, 2),  # CAM_BACK_RIGHT
        ]
        cam_axes = []
        for i, (row, col) in enumerate(cam_grid_positions):
            ax = fig.add_subplot(gs[row, col])
            img_rgb = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(CAMERA_NAMES[i], fontsize=8)
            ax.set_xlim([0, imgs[i].shape[1]])
            ax.set_ylim([imgs[i].shape[0], 0])
            ax.axis('off')
            cam_axes.append((i, ax))

        # Draw 3D bboxes on camera images
        for det in detections:
            bbox = det['bbox_3d']
            corners = get_3d_box_corners(
                bbox['center'], bbox['size'], bbox['yaw'])
            cls_id = det['class_id']
            color = np.array(CLASS_COLORS_RGB[cls_id]) / 255.0
            for cam_idx, ax in cam_axes:
                draw_3d_bbox_on_camera(
                    ax, corners, cam_idx, color, det['score'])

        ax_bev = fig.add_subplot(gs[:, 3])
    else:
        fig, ax_bev = plt.subplots(1, 1, figsize=(10, 10))

    ax_bev.set_xlim([pc[0], pc[3]])
    ax_bev.set_ylim([pc[1], pc[4]])
    ax_bev.set_aspect('equal')
    ax_bev.set_xlabel('X (m)')
    ax_bev.set_ylabel('Y (m)')
    ax_bev.set_title('Bird\'s Eye View Detections')
    ax_bev.grid(True, alpha=0.3)

    # Ego vehicle
    ax_bev.plot(0, 0, 'k^', markersize=12, label='Ego vehicle')

    for det in detections:
        bbox = det['bbox_3d']
        center = bbox['center']
        size = bbox['size']
        yaw = bbox['yaw']
        cls_id = det['class_id']
        score = det['score']
        cls_name = det['class_name']
        color = np.array(CLASS_COLORS_RGB[cls_id]) / 255.0

        corners = get_3d_box_corners(center, size, yaw)
        bev_corners = corners[:4, :2]

        polygon = plt.Polygon(
            bev_corners, fill=False, edgecolor=color, linewidth=2)
        ax_bev.add_patch(polygon)

        # Heading arrow
        front_center = (bev_corners[1] + bev_corners[2]) / 2
        rear_center = (bev_corners[0] + bev_corners[3]) / 2
        box_center = (front_center + rear_center) / 2
        ax_bev.annotate(
            '', xy=front_center, xytext=box_center,
            arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        ax_bev.text(
            center[0], center[1] + size[0] / 2 + 1.5,
            f'{score:.2f}',
            fontsize=6, color=color, ha='center')

    # Distance circles
    for r in [10, 20, 30, 40, 50]:
        circle = plt.Circle(
            (0, 0), r, fill=False, color='gray',
            linestyle='--', alpha=0.2)
        ax_bev.add_patch(circle)

    # Legend
    seen_classes = set()
    legend_elements = []
    for det in detections:
        cls_id = det['class_id']
        if cls_id not in seen_classes:
            seen_classes.add(cls_id)
            color = np.array(CLASS_COLORS_RGB[cls_id]) / 255.0
            legend_elements.append(
                plt.Line2D([0], [0], color=color, lw=2,
                           label=NUSCENES_CLASSES[cls_id]))
    if legend_elements:
        ax_bev.legend(handles=legend_elements, loc='upper right', fontsize=8)

    fig.tight_layout()
    return fig


# ======================
# Main functions
# ======================

def predict_onnx(session, imgs):
    """Run inference with ONNX Runtime."""
    images = prepare_multi_cam_input(imgs)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: images})
    cls_scores, bbox_preds = outputs[0], outputs[1]
    detections = post_process(cls_scores, bbox_preds, args.threshold)
    return detections


def predict_ailia(net, imgs):
    """Run inference with ailia SDK."""
    images = prepare_multi_cam_input(imgs)
    output = net.predict([images])
    cls_scores, bbox_preds = output[0], output[1]
    detections = post_process(cls_scores, bbox_preds, args.threshold)
    return detections


def load_camera_images(input_paths):
    """Load 6 camera images from input paths.

    If 6 paths are given, load each as a separate camera view.
    If 1 path is given, replicate to all 6 cameras.

    Returns:
        list of 6 BGR images
    """
    if len(input_paths) == NUM_CAMS:
        imgs = []
        for path in input_paths:
            img = imread(path)
            if img is None:
                logger.error(f'Could not read image: {path}')
                return None
            imgs.append(img)
        return imgs
    elif len(input_paths) == 1:
        img = imread(input_paths[0])
        if img is None:
            logger.error(f'Could not read image: {input_paths[0]}')
            return None
        logger.warning(
            'Only 1 image provided, replicating to all 6 cameras. '
            'For proper results, provide 6 surround-view camera images.')
        return [img] * NUM_CAMS
    else:
        logger.error(
            f'Expected 1 or {NUM_CAMS} input images, got {len(input_paths)}. '
            f'Provide 6 images in order: {", ".join(CAMERA_NAMES)}')
        return None


def recognize_from_image(predictor, predict_fn):
    """Run inference on image(s)."""
    imgs = load_camera_images(args.input)
    if imgs is None:
        return

    for i, path in enumerate(args.input[:NUM_CAMS]):
        logger.info(f'{CAMERA_NAMES[i]}: {path}')

    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            detections = predict_fn(predictor, imgs)
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)

            logger.info(
                f'\tailia processing time {estimation_time} ms')
            if i != 0:
                total_time += estimation_time

        logger.info(
            f'\taverage time '
            f'{total_time / (args.benchmark_count - 1)} ms')
    else:
        detections = predict_fn(predictor, imgs)

    logger.info(f'Detected {len(detections)} objects')
    for det in detections:
        logger.info(
            f'  {det["class_name"]}: score={det["score"]:.3f}, '
            f'center=({det["bbox_3d"]["center"][0]:.1f}, '
            f'{det["bbox_3d"]["center"][1]:.1f}, '
            f'{det["bbox_3d"]["center"][2]:.1f})')

    fig = draw_bev_detections(detections, imgs)

    save_path = get_savepath(args.savepath, args.input[0], ext='.png')
    fig.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    logger.info(f'saved at : {save_path}')

    if args.write_json:
        json_file = '%s.json' % save_path.rsplit('.', 1)[0]
        with open(json_file, 'w') as f:
            json.dump(detections, f, indent=2)
        logger.info(f'JSON saved at : {json_file}')

    logger.info('Script finished successfully.')


def recognize_from_video(predictor, predict_fn):
    """Run inference on video (uses single camera replicated to 6 views)."""
    video_file = args.video if args.video else args.input[0]
    capture = webcamera_utils.get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        imgs = [frame] * NUM_CAMS
        detections = predict_fn(predictor, imgs)

        fig = draw_bev_detections(detections, imgs)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        vis_img = buf.reshape(h, w, 3)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        plt.close(fig)

        cv2.imshow('frame', vis_img)
        frame_shown = True

        if writer is not None:
            writer.write(vis_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    use_onnx = args.onnx or not HAS_AILIA

    if use_onnx:
        import onnxruntime as ort
        logger.info('Using ONNX Runtime')

        check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

        session = ort.InferenceSession(
            WEIGHT_PATH, providers=['CPUExecutionProvider'])

        predict_fn = predict_onnx
        predictor = session
    else:
        check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

        env_id = args.env_id

        memory_mode = ailia.get_memory_mode(
            reduce_constant=True, ignore_input_with_initializer=True,
            reduce_interstage=True, reuse_interstage=False)
        net = ailia.Net(
            MODEL_PATH, WEIGHT_PATH,
            env_id=env_id, memory_mode=memory_mode)

        predict_fn = predict_ailia
        predictor = net

    if args.video is not None:
        recognize_from_video(predictor, predict_fn)
    else:
        recognize_from_image(predictor, predict_fn)


if __name__ == '__main__':
    main()
