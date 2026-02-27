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

IMAGE_PATH = 'demo.jpg'
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


def prepare_multi_cam_input(img):
    """Prepare 6-camera input from a single image.

    In a real application, this would load 6 separate camera images.
    For the demo, we replicate the front camera image to all 6 views.

    Args:
        img: BGR image (H, W, 3)

    Returns:
        images: (1, 6, 3, H, W) float32
    """
    cam_img = preprocess_image(img)

    # Replicate to 6 cameras
    multi_cam = np.stack([cam_img] * NUM_CAMS, axis=0)

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
    return detections


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


def draw_bev_detections(detections, img=None):
    """Draw detection results in Bird's Eye View."""
    pc = POINT_CLOUD_RANGE

    if img is not None:
        fig, (ax_img, ax_bev) = plt.subplots(1, 2, figsize=(20, 8))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax_img.imshow(img_rgb)
        ax_img.set_title('Camera Image')
        ax_img.axis('off')
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
            center[0], center[1] + size[0] / 2 + 1.0,
            f'{cls_name} {score:.2f}',
            fontsize=7, color=color, ha='center')

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

def predict_onnx(session, img):
    """Run inference with ONNX Runtime."""
    images = prepare_multi_cam_input(img)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: images})
    cls_scores, bbox_preds = outputs[0], outputs[1]
    detections = post_process(cls_scores, bbox_preds, args.threshold)
    return detections


def predict_ailia(net, img):
    """Run inference with ailia SDK."""
    images = prepare_multi_cam_input(img)
    output = net.predict([images])
    cls_scores, bbox_preds = output[0], output[1]
    detections = post_process(cls_scores, bbox_preds, args.threshold)
    return detections


def recognize_from_image(predictor, predict_fn):
    """Run inference on image(s)."""
    for image_path in args.input:
        logger.info(image_path)

        img = imread(image_path)
        if img is None:
            logger.error(f'Could not read image: {image_path}')
            continue

        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                detections = predict_fn(predictor, img)
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
            detections = predict_fn(predictor, img)

        logger.info(f'Detected {len(detections)} objects')
        for det in detections:
            logger.info(
                f'  {det["class_name"]}: score={det["score"]:.3f}, '
                f'center=({det["bbox_3d"]["center"][0]:.1f}, '
                f'{det["bbox_3d"]["center"][1]:.1f}, '
                f'{det["bbox_3d"]["center"][2]:.1f})')

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


def recognize_from_video(predictor, predict_fn):
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

        detections = predict_fn(predictor, frame)

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
