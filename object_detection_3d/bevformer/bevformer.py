import os
import sys
import time
import json

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from image_utils import imread  # noqa
import webcamera_utils  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_BACKBONE_PATH = 'bevformer_tiny_backbone.onnx'
MODEL_BACKBONE_PATH = 'bevformer_tiny_backbone.onnx.prototxt'
WEIGHT_HEAD_PATH = 'bevformer_tiny_head.onnx'
MODEL_HEAD_PATH = 'bevformer_tiny_head.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/bevformer/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

# BEVFormer-tiny configuration
IMAGE_HEIGHT = 928
IMAGE_WIDTH = 1600
BEV_H = 50
BEV_W = 50
POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
NUM_QUERIES = 900
NUM_CLASSES = 10

# ImageNet normalization (BEVFormer-tiny uses RGB with ImageNet mean/std)
IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

THRESHOLD = 0.3

# nuScenes detection classes
NUSCENES_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

# Colors for each class (BGR for visualization)
CLASS_COLORS = [
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

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'BEVFormer: Bird\'s-Eye-View 3D Object Detection', IMAGE_PATH, SAVE_IMAGE_PATH
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
    '--multi_view', nargs='+', default=None,
    help='Paths to 6 camera images: front, front_right, front_left, '
         'back, back_left, back_right'
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
        preprocessed image (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    """
    h, w = img.shape[:2]

    # Resize to model input size
    img_resized = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Normalize with ImageNet mean/std
    img_norm = (img_rgb - IMG_MEAN) / IMG_STD

    # HWC -> CHW
    img_chw = img_norm.transpose(2, 0, 1)

    # Add batch dimension
    img_batch = np.expand_dims(img_chw, axis=0).astype(np.float32)

    return img_batch


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
    pc_range = POINT_CLOUD_RANGE
    cx = cx * (pc_range[3] - pc_range[0]) + pc_range[0]
    cy = cy * (pc_range[4] - pc_range[1]) + pc_range[1]
    cz = cz * (pc_range[5] - pc_range[2]) + pc_range[2]

    # Exponentiate dimensions
    w = np.exp(w)
    l = np.exp(l)
    h = np.exp(h)

    return {
        'center': [float(cx), float(cy), float(cz)],
        'size': [float(w), float(l), float(h)],
        'yaw': float(yaw),
        'velocity': [float(vx), float(vy)],
    }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def post_process(cls_scores, bbox_preds, threshold):
    """Post-process model outputs to get detections.

    Args:
        cls_scores: (B, num_queries, num_classes)
        bbox_preds: (B, num_queries, code_size)
        threshold: confidence threshold

    Returns:
        list of detection dicts
    """
    # Take first batch
    cls_scores = cls_scores[0]  # (num_queries, num_classes)
    bbox_preds = bbox_preds[0]  # (num_queries, code_size)

    # Apply sigmoid to classification scores
    scores = sigmoid(cls_scores)

    # Get max score and class for each query
    max_scores = np.max(scores, axis=1)  # (num_queries,)
    max_classes = np.argmax(scores, axis=1)  # (num_queries,)

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

    # Sort by score (descending)
    detections.sort(key=lambda x: x['score'], reverse=True)

    return detections


def get_3d_box_corners(center, size, yaw):
    """Compute 8 corner points of a 3D bounding box in BEV.

    Args:
        center: [x, y, z]
        size: [w, l, h]
        yaw: rotation around z-axis (radians)

    Returns:
        corners: (8, 3) array of corner coordinates
    """
    w, l, h = size
    x, y, z = center

    # 8 corners in local coordinate (centered at origin)
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

    # Rotation matrix around z-axis
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1],
    ])

    # Rotate and translate
    corners = corners @ rot.T + np.array([x, y, z])

    return corners


def draw_bev_detections(detections, img=None):
    """Draw detection results in Bird's Eye View.

    Args:
        detections: list of detection dicts
        img: optional camera image to show alongside BEV

    Returns:
        matplotlib figure
    """
    pc_range = POINT_CLOUD_RANGE

    if img is not None:
        fig, (ax_img, ax_bev) = plt.subplots(1, 2, figsize=(20, 8))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax_img.imshow(img_rgb)
        ax_img.set_title('Camera Image')
        ax_img.axis('off')
    else:
        fig, ax_bev = plt.subplots(1, 1, figsize=(10, 10))

    # Draw BEV grid
    ax_bev.set_xlim([pc_range[0], pc_range[3]])
    ax_bev.set_ylim([pc_range[1], pc_range[4]])
    ax_bev.set_aspect('equal')
    ax_bev.set_xlabel('X (m)')
    ax_bev.set_ylabel('Y (m)')
    ax_bev.set_title('Bird\'s Eye View Detections')
    ax_bev.grid(True, alpha=0.3)

    # Draw ego vehicle position
    ax_bev.plot(0, 0, 'k^', markersize=12, label='Ego vehicle')

    # Draw detection boxes
    for det in detections:
        bbox = det['bbox_3d']
        center = bbox['center']
        size = bbox['size']
        yaw = bbox['yaw']
        cls_id = det['class_id']
        score = det['score']
        cls_name = det['class_name']

        corners = get_3d_box_corners(center, size, yaw)

        # Project to BEV (x-y plane, take bottom 4 corners)
        bev_corners = corners[:4, :2]
        polygon = plt.Polygon(
            bev_corners,
            fill=False,
            edgecolor=np.array(CLASS_COLORS[cls_id][::-1]) / 255.0,
            linewidth=2,
        )
        ax_bev.add_patch(polygon)

        # Draw heading direction
        front_center = (bev_corners[1] + bev_corners[2]) / 2
        rear_center = (bev_corners[0] + bev_corners[3]) / 2
        box_center = (front_center + rear_center) / 2
        ax_bev.annotate(
            '',
            xy=front_center,
            xytext=box_center,
            arrowprops=dict(
                arrowstyle='->',
                color=np.array(CLASS_COLORS[cls_id][::-1]) / 255.0,
                lw=1.5),
        )

        # Label
        ax_bev.text(
            center[0], center[1] + size[0] / 2 + 1.0,
            f'{cls_name} {score:.2f}',
            fontsize=7,
            color=np.array(CLASS_COLORS[cls_id][::-1]) / 255.0,
            ha='center',
        )

    # Draw distance circles
    for r in [10, 20, 30, 40, 50]:
        circle = plt.Circle(
            (0, 0), r, fill=False, color='gray',
            linestyle='--', alpha=0.2)
        ax_bev.add_patch(circle)

    # Add legend
    legend_elements = []
    seen_classes = set()
    for det in detections:
        cls_id = det['class_id']
        if cls_id not in seen_classes:
            seen_classes.add(cls_id)
            color = np.array(CLASS_COLORS[cls_id][::-1]) / 255.0
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

def predict(backbone, head, img):
    """Run BEVFormer inference on a single image.

    Args:
        backbone: ailia.Net for image backbone (ResNet50 + FPN)
        head: ailia.Net for detection head
        img: BGR image (H, W, 3)

    Returns:
        list of detection dicts
    """
    # Preprocess
    img_input = preprocess_image(img)

    # Extract image features with backbone
    features = backbone.predict([img_input])
    if isinstance(features, list):
        features = features[0]

    # BEV feature generation
    # For single-frame inference without temporal context,
    # we use the backbone features directly resized to BEV grid
    bev_features = features
    if bev_features.shape[2] != BEV_H or bev_features.shape[3] != BEV_W:
        # Interpolate to BEV grid size using numpy
        from scipy.ndimage import zoom
        zoom_h = BEV_H / bev_features.shape[2]
        zoom_w = BEV_W / bev_features.shape[3]
        bev_features = zoom(bev_features, (1, 1, zoom_h, zoom_w), order=1)
    bev_features = bev_features.astype(np.float32)

    # Detection head
    output = head.predict([bev_features])
    cls_scores, bbox_preds = output[0], output[1]

    # Post-process
    detections = post_process(cls_scores, bbox_preds, args.threshold)

    return detections


def recognize_from_image(backbone, head):
    """Run inference on image(s)."""
    for image_path in args.input:
        logger.info(image_path)

        img = imread(image_path)
        if img is None:
            logger.error(f'Could not read image: {image_path}')
            continue

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                detections = predict(backbone, head, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                logger.info(f'\tailia processing time {estimation_time} ms')
                if i != 0:
                    total_time += estimation_time

            logger.info(
                f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            detections = predict(backbone, head, img)

        # Log results
        logger.info(f'Detected {len(detections)} objects')
        for det in detections:
            logger.info(
                f'  {det["class_name"]}: score={det["score"]:.3f}, '
                f'center=({det["bbox_3d"]["center"][0]:.1f}, '
                f'{det["bbox_3d"]["center"][1]:.1f}, '
                f'{det["bbox_3d"]["center"][2]:.1f})')

        # Draw BEV visualization
        fig = draw_bev_detections(detections, img)

        save_path = get_savepath(args.savepath, image_path, ext='.png')
        fig.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        logger.info(f'saved at : {save_path}')

        if args.write_json:
            json_file = '%s.json' % save_path.rsplit('.', 1)[0]
            with open(json_file, 'w') as f:
                json.dump(detections, f, indent=2)
            logger.info(f'JSON saved at : {json_file}')

    logger.info('Script finished successfully.')


def recognize_from_video(backbone, head):
    """Run inference on video."""
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

        detections = predict(backbone, head, frame)

        # Draw BEV as overlay
        fig = draw_bev_detections(detections, frame)
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
    # model files check and download
    check_and_download_models(
        WEIGHT_BACKBONE_PATH, MODEL_BACKBONE_PATH, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_HEAD_PATH, MODEL_HEAD_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    memory_mode = ailia.get_memory_mode(
        reduce_constant=True, ignore_input_with_initializer=True,
        reduce_interstage=True, reuse_interstage=False)
    backbone = ailia.Net(
        MODEL_BACKBONE_PATH, WEIGHT_BACKBONE_PATH,
        env_id=env_id, memory_mode=memory_mode)
    head = ailia.Net(
        MODEL_HEAD_PATH, WEIGHT_HEAD_PATH,
        env_id=env_id, memory_mode=memory_mode)

    if args.video is not None:
        recognize_from_video(backbone, head)
    else:
        recognize_from_image(backbone, head)


if __name__ == '__main__':
    main()
