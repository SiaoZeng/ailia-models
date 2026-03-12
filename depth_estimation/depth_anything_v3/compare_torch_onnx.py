"""Compare Depth Anything V3 PyTorch vs ONNX inference results.

Usage:
    python compare_torch_onnx.py -ec vits
    python compare_torch_onnx.py -ec vitb
    python compare_torch_onnx.py -ec vitl
"""

import sys
import argparse
import os

import cv2
import numpy as np
import torch
import onnxruntime as ort

# ---- configuration --------------------------------------------------------

PROCESS_RES = 504
PATCH_SIZE = 14

model_name_map = {
    'vits': 'depth-anything/DA3-Small',
    'vitb': 'depth-anything/DA3-Base',
    'vitl': 'depth-anything/DA3-Large',
}

config_name_map = {
    'vits': 'depth_anything_3.configs.da3-small',
    'vitb': 'depth_anything_3.configs.da3-base',
    'vitl': 'depth_anything_3.configs.da3-large',
}

onnx_weight_map = {
    'vits': 'depth_anything_v3_da3_vits.onnx',
    'vitb': 'depth_anything_v3_da3_vitb.onnx',
    'vitl': 'depth_anything_v3_da3_vitl.onnx',
}


# ---- preprocessing (same as depth_anything_v3.py) -------------------------

def nearest_multiple(x, p):
    down = (x // p) * p
    up = down + p
    return up if abs(up - x) <= abs(x - down) else down


def preprocess(image):
    h, w = image.shape[:2]
    longest = max(w, h)
    scale = PROCESS_RES / float(longest)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    image = cv2.resize(image, (new_w, new_h), interpolation=interp)

    final_w = max(1, nearest_multiple(new_w, PATCH_SIZE))
    final_h = max(1, nearest_multiple(new_h, PATCH_SIZE))
    if final_w != new_w or final_h != new_h:
        upscale = (final_w > new_w) or (final_h > new_h)
        interp2 = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
        image = cv2.resize(image, (final_w, final_h), interpolation=interp2)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = image.transpose(2, 0, 1).astype(np.float32)
    return image


# ---- PyTorch model loading -------------------------------------------------

def load_torch_model(encoder):
    from depth_anything_3.cfg import load_config, create_object
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    config_name = config_name_map[encoder]
    print(f'[Torch] Loading config {config_name}...')
    cfg = load_config(config_name)
    model = create_object(cfg)

    model_name = model_name_map[encoder]
    print(f'[Torch] Downloading weights from {model_name}...')
    weight_path = hf_hub_download(repo_id=model_name, filename='model.safetensors')
    state_dict = load_file(weight_path)
    state_dict = {
        k[len('model.'):] if k.startswith('model.') else k: v
        for k, v in state_dict.items()
    }
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f'[Torch] WARNING: Missing keys: {result.missing_keys}')
    if result.unexpected_keys:
        print(f'[Torch] WARNING: Unexpected keys: {result.unexpected_keys}')
    model.eval()
    return model


def torch_inference(model, input_np):
    """Run PyTorch inference. input_np: (1, 3, H, W) float32 numpy array."""
    input_tensor = torch.from_numpy(input_np)
    # DA3 expects (B, V, 3, H, W) where V=1
    input_tensor = input_tensor.unsqueeze(1)
    with torch.no_grad():
        output = model(input_tensor)
    depth = output.depth.numpy()  # (B, H, W)
    return depth


# ---- ONNX model loading ---------------------------------------------------

def load_onnx_model(encoder):
    onnx_path = onnx_weight_map[encoder]
    if not os.path.exists(onnx_path):
        # Try downloading via ailia model utils pattern
        remote = f'https://storage.googleapis.com/ailia-models/depth_anything_v3/{onnx_path}'
        print(f'[ONNX] Downloading {remote}...')
        import urllib.request
        urllib.request.urlretrieve(remote, onnx_path)
    print(f'[ONNX] Loading {onnx_path}...')
    session = ort.InferenceSession(onnx_path)
    return session


def onnx_inference(session, input_np):
    """Run ONNX inference. input_np: (1, 3, H, W) float32 numpy array."""
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: input_np})
    return result[0]


# ---- comparison ------------------------------------------------------------

def compare_results(torch_depth, onnx_depth):
    """Compare two depth maps and print detailed statistics."""
    print('\n' + '=' * 60)
    print('Comparison Results')
    print('=' * 60)

    print(f'Torch output shape: {torch_depth.shape}')
    print(f'ONNX  output shape: {onnx_depth.shape}')
    print(f'Torch output range: [{torch_depth.min():.6f}, {torch_depth.max():.6f}]')
    print(f'ONNX  output range: [{onnx_depth.min():.6f}, {onnx_depth.max():.6f}]')

    # Ensure same shape for comparison
    if torch_depth.shape != onnx_depth.shape:
        print(f'WARNING: Shape mismatch! Torch={torch_depth.shape}, ONNX={onnx_depth.shape}')
        # Try to match shapes
        if torch_depth.ndim == 3 and onnx_depth.ndim == 2:
            torch_depth = torch_depth[0]
        elif torch_depth.ndim == 2 and onnx_depth.ndim == 3:
            onnx_depth = onnx_depth[0]

    diff = np.abs(torch_depth - onnx_depth)
    rel_diff = diff / (np.abs(torch_depth) + 1e-8)

    print(f'\nAbsolute difference:')
    print(f'  Mean: {diff.mean():.8f}')
    print(f'  Max:  {diff.max():.8f}')
    print(f'  Std:  {diff.std():.8f}')
    print(f'  Median: {np.median(diff):.8f}')

    print(f'\nRelative difference:')
    print(f'  Mean: {rel_diff.mean() * 100:.6f}%')
    print(f'  Max:  {rel_diff.max() * 100:.6f}%')

    # Correlation
    t_flat = torch_depth.flatten()
    o_flat = onnx_depth.flatten()
    corr = np.corrcoef(t_flat, o_flat)[0, 1]
    print(f'\nPearson correlation: {corr:.10f}')

    # Normalized comparison (how it would look after post-processing)
    def normalize(d):
        return (d - d.min()) / (d.max() - d.min() + 1e-8) * 255.0

    t_norm = normalize(torch_depth)
    o_norm = normalize(onnx_depth)
    norm_diff = np.abs(t_norm - o_norm)
    print(f'\nAfter normalization to [0, 255]:')
    print(f'  Mean pixel diff: {norm_diff.mean():.4f}')
    print(f'  Max pixel diff:  {norm_diff.max():.4f}')
    print(f'  % pixels with diff > 1: {(norm_diff > 1).mean() * 100:.2f}%')
    print(f'  % pixels with diff > 5: {(norm_diff > 5).mean() * 100:.2f}%')
    print(f'  % pixels with diff > 10: {(norm_diff > 10).mean() * 100:.2f}%')

    # Check if structural similarity is maintained
    print(f'\n{"=" * 60}')
    if corr > 0.999:
        print('VERDICT: Excellent match (correlation > 0.999)')
    elif corr > 0.99:
        print('VERDICT: Good match (correlation > 0.99)')
    elif corr > 0.95:
        print('VERDICT: Moderate match (correlation > 0.95) - some degradation')
    else:
        print('VERDICT: Poor match (correlation <= 0.95) - significant degradation!')
    print('=' * 60)

    return diff, corr


def save_comparison_images(torch_depth, onnx_depth, output_prefix='compare'):
    """Save visual comparison images."""
    def normalize_and_colormap(d):
        while d.ndim > 2:
            d = d[0]
        d = (d - d.min()) / (d.max() - d.min() + 1e-8) * 255.0
        d = d.astype(np.uint8)
        return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)

    torch_vis = normalize_and_colormap(torch_depth)
    onnx_vis = normalize_and_colormap(onnx_depth)

    # Difference heatmap
    while torch_depth.ndim > 2:
        torch_depth = torch_depth[0]
    while onnx_depth.ndim > 2:
        onnx_depth = onnx_depth[0]

    diff = np.abs(torch_depth - onnx_depth)
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8) * 255.0
    diff_vis = cv2.applyColorMap(diff_norm.astype(np.uint8), cv2.COLORMAP_JET)

    cv2.imwrite(f'{output_prefix}_torch.png', torch_vis)
    cv2.imwrite(f'{output_prefix}_onnx.png', onnx_vis)
    cv2.imwrite(f'{output_prefix}_diff.png', diff_vis)
    print(f'\nSaved: {output_prefix}_torch.png, {output_prefix}_onnx.png, {output_prefix}_diff.png')


def main():
    parser = argparse.ArgumentParser(description='Compare Torch vs ONNX for DA3')
    parser.add_argument('--encoder', '-ec', type=str, default='vits',
                        help='model type: vits, vitb, vitl')
    parser.add_argument('--input', '-i', type=str, default='demo.png',
                        help='input image path')
    parser.add_argument('--save-images', action='store_true',
                        help='save comparison images')
    args = parser.parse_args()

    # Load input image
    print(f'Loading input image: {args.input}')
    img = cv2.imread(args.input)
    if img is None:
        print(f'Error: Cannot read {args.input}')
        sys.exit(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    # Preprocess
    input_np = preprocess(img)[None]  # (1, 3, H, W)
    print(f'Preprocessed input shape: {input_np.shape}')

    # Load models
    torch_model = load_torch_model(args.encoder)
    onnx_session = load_onnx_model(args.encoder)

    # Run inference
    print('\nRunning PyTorch inference...')
    torch_depth = torch_inference(torch_model, input_np)
    print(f'PyTorch output shape: {torch_depth.shape}')

    print('Running ONNX inference...')
    onnx_depth = onnx_inference(onnx_session, input_np)
    print(f'ONNX output shape: {onnx_depth.shape}')

    # Compare
    diff, corr = compare_results(torch_depth, onnx_depth)

    # Save images if requested
    if args.save_images:
        save_comparison_images(torch_depth, onnx_depth,
                               output_prefix=f'compare_{args.encoder}')


if __name__ == '__main__':
    main()
