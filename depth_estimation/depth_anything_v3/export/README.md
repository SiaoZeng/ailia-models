# Depth Anything V3 ONNX Export

## Usage

```bash
pip install depth-anything-3 safetensors huggingface_hub onnx
python export_onnx.py --encoder vits  # or vitb, vitl
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--encoder` | `vits` | Model type: `vits`, `vitb`, `vitl` |
| `--output` | `da3_{encoder}.onnx` | Output ONNX file path |
| `--opset` | `17` | ONNX opset version |
| `--input_height` | `336` | Input image height (must be multiple of 14) |
| `--input_width` | `504` | Input image width (must be multiple of 14) |

## Dynamic Shape

Exported ONNX models support dynamic shapes for batch, height, and width dimensions:

- Input `image`: `[batch, 3, height, width]`
- Output `depth`: `[batch, 1, height, width]`

## Note: torch.cartesian_prod patch

`depth_anything_3` package uses `torch.cartesian_prod` in `PositionGetter` (`depth_anything_3/model/dinov2/layers/rope.py`), which is not supported by the ONNX exporter. Before running the export, this must be replaced with an equivalent using `torch.meshgrid` + `torch.stack`.

Apply the following patch to the installed package before export:

```python
# In depth_anything_3/model/dinov2/layers/rope.py
# PositionGetter.__call__()

# Before (not supported by ONNX):
positions = torch.cartesian_prod(y_coords, x_coords)

# After (ONNX-compatible):
grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
positions = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
```

The file to patch is located at:
```
<python_site_packages>/depth_anything_3/model/dinov2/layers/rope.py
```
