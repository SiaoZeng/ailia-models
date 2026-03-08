import sys
import time
from logging import getLogger

import numpy as np
import cv2
import open_clip
from PIL import Image

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser
from model_utils import check_and_download_models
from detector_utils import load_image

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_VIT_L14_PATH = "aesthetic_predictor_vit_l_14.onnx"
MODEL_VIT_L14_PATH = "aesthetic_predictor_vit_l_14.onnx.prototxt"
WEIGHT_VIT_B32_PATH = "aesthetic_predictor_vit_b_32.onnx"
MODEL_VIT_B32_PATH = "aesthetic_predictor_vit_b_32.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/aesthetic-predictor/"

IMAGE_PATH = "demo.jpg"
SAVE_IMAGE_PATH = "output.png"

CLIP_MODEL_CONFIGS = {
    "vit_l_14": {
        "weight_path": WEIGHT_VIT_L14_PATH,
        "model_path": MODEL_VIT_L14_PATH,
        "clip_name": "ViT-L-14",
        "pretrained": "openai",
        "feature_dim": 768,
    },
    "vit_b_32": {
        "weight_path": WEIGHT_VIT_B32_PATH,
        "model_path": MODEL_VIT_B32_PATH,
        "clip_name": "ViT-B-32",
        "pretrained": "openai",
        "feature_dim": 512,
    },
}

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("LAION Aesthetic Predictor", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "-m",
    "--model_type",
    default="vit_l_14",
    choices=list(CLIP_MODEL_CONFIGS.keys()),
    help="CLIP model variant (default: vit_l_14)",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Main functions
# ======================


def extract_clip_features(clip_model, preprocess, img):
    """Extract normalized image features using open_clip."""
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    image_tensor = preprocess(pil_image).unsqueeze(0)

    features = clip_model.encode_image(image_tensor)
    features /= features.norm(dim=-1, keepdim=True)

    return features.detach().numpy().astype(np.float32)


def predict(net, image_features):
    """Run aesthetic predictor on normalized CLIP features."""
    if not args.onnx:
        output = net.predict([image_features])
    else:
        output = net.run(None, {"image_features": image_features})

    score = float(output[0][0][0])
    return score


def recognize_from_image(models):
    net = models["net"]
    clip_model = models["clip_model"]
    preprocess = models["preprocess"]

    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                features = extract_clip_features(clip_model, preprocess, img)
                score = predict(net, features)
                end = int(round(time.time() * 1000))
                estimation_time = end - start
                logger.info(f"\tailia processing estimation time {estimation_time} ms")
                if i != 0:
                    total_time_estimation += estimation_time
            logger.info(
                f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
            )
        else:
            features = extract_clip_features(clip_model, preprocess, img)
            score = predict(net, features)

        logger.info(f"Aesthetic score: {score:.4f}")

        # Overlay score on image and save
        label = f"Aesthetic score: {score:.2f}"
        cv2.putText(
            img,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(args.savepath, img)
        logger.info(f"saved at : {args.savepath}")

    logger.info("Script finished successfully.")


def main():
    logger.info("=== LAION Aesthetic Predictor ===")

    config = CLIP_MODEL_CONFIGS[args.model_type]
    weight_path = config["weight_path"]
    model_path = config["model_path"]

    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # Load CLIP model via open_clip
    logger.info(
        f"Loading CLIP model: {config['clip_name']} ({config['pretrained']})..."
    )
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        config["clip_name"], pretrained=config["pretrained"]
    )
    clip_model.eval()
    logger.info("CLIP model loaded.")

    env_id = args.env_id

    # Load aesthetic predictor (linear layer)
    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=env_id)
    else:
        import onnxruntime

        cuda = 0 < ailia.get_gpu_environment_id()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )
        net = onnxruntime.InferenceSession(weight_path, providers=providers)

    models = {
        "net": net,
        "clip_model": clip_model,
        "preprocess": preprocess,
    }

    recognize_from_image(models)


if __name__ == "__main__":
    main()
