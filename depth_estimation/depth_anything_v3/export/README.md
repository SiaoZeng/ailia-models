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

Apply the following patch to the installed package before export.

File to patch: `<python_site_packages>/depth_anything_3/model/dinov2/layers/rope.py`

### Before (original code)

```python
class PositionGetter:
    def __init__(self):
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()
```

### After (ONNX-compatible)

```python
class PositionGetter:
    def __init__(self):
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            positions = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()
```

`torch.cartesian_prod` is not supported by the ONNX opset. `torch.meshgrid` + `torch.stack` produces the same (y, x) coordinate pairs and is fully ONNX-compatible.
