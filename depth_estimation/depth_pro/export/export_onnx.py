import sys
import os
import argparse

import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='output onnx file path'
    )
    parser.add_argument(
        '--opset', type=int, default=18,
        help='ONNX opset version'
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Clone the ml-depth-pro repo and install dependencies before running:
    #   git clone https://github.com/apple/ml-depth-pro.git
    #   cd ml-depth-pro
    #   pip install -e .
    # Download pretrained weights:
    #   source get_pretrained_models.sh
    # Then run this script from within the ml-depth-pro directory.
    try:
        from depth_pro.depth_pro import (
            create_model_and_transforms,
            DEFAULT_MONODEPTH_CONFIG_DICT,
        )
    except ImportError:
        print("Error: Cannot import depth_pro.")
        print("Please clone the ml-depth-pro repository:")
        print("  git clone https://github.com/apple/ml-depth-pro.git")
        print("  cd ml-depth-pro")
        print("  pip install -e .")
        print("  source get_pretrained_models.sh")
        print("Then run this script from within the ml-depth-pro directory.")
        sys.exit(1)

    print('Loading DepthPro model...')
    model, transform = create_model_and_transforms(
        config=DEFAULT_MONODEPTH_CONFIG_DICT,
        device=torch.device("cpu"),
        precision=torch.float32,
    )
    model.eval()

    if args.output is None:
        output_path = 'depth_pro.onnx'
    else:
        output_path = args.output

    # DepthPro expects 1536x1536 input
    img_size = model.img_size
    dummy_input = torch.randn(1, 3, img_size, img_size)

    print(f'Exporting DepthPro model to {output_path}...')
    print(f'Input size: {img_size}x{img_size}')

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=args.opset,
        input_names=['image'],
        output_names=['canonical_inverse_depth', 'fov_deg'],
    )

    # Re-save with external data to ensure small tensors stay inline
    data_path = output_path + '.data'
    if os.path.exists(data_path):
        print('Re-saving ONNX with external data (keeping small tensors inline)...')
        model_onnx = onnx.load(output_path, load_external_data=True)
        onnx.save_model(
            model_onnx,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(output_path) + '.data',
            size_threshold=1024,
        )
        print(f'External data saved to {output_path}.data')

    print(f'Successfully exported to {output_path}')


if __name__ == '__main__':
    main()
