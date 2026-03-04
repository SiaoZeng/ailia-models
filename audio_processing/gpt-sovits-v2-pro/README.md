# GPT-SoVITS V2 Pro

### Input
- A synthesis text and reference audio and reference text for voice cloning

### Output
The Voice file is output as .wav which path is defined as `SAVE_WAV_PATH` in `gpt-sovits-v2-pro.py `.

### Requirements
This model requires pyopenjtalk for g2p.

```
pip3 install -r requirements.txt
```

### Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample sentence and sample audio,
```
python3 gpt-sovits-v2-pro.py
```

Run with audio prompt.

```
python3 gpt-sovits-v2-pro.py -i "ax株式会社ではAIの実用化のための技術を開発しています。" --ref_audio reference_audio_captured_by_ax.wav --ref_text "水をマレーシアから買わなくてはならない。"
```

Run for english.

```
python3 gpt-sovits-v2-pro.py -i "Hello world. We are testing speech synthesis." --text_language en --ref_audio reference_audio_captured_by_ax.wav --ref_text "水をマレーシアから買わなくてはならない。" --ref_language ja
```

### ONNX Export

To export the PyTorch model to ONNX, run the export script from within the GPT-SoVITS repository (20250606v2pro branch):

```
git clone -b 20250606v2pro https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS/GPT_SoVITS
python3 /path/to/export/export_onnx.py \
    --sovits_path SoVITS_weights_v2pro/your_model.pth \
    --gpt_path GPT_weights_v2pro/your_model.ckpt \
    --output_dir onnx_output
```

### Reference
[GPT-SoVITS (20250606v2pro)](https://github.com/RVC-Boss/GPT-SoVITS/tree/20250606v2pro)

### Framework
PyTorch 2.5.0

### Model Format
ONNX opset = 17

### Netron

#### Normal model

- [cnhubert.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro/cnhubert.onnx.prototxt)
- [t2s_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro/t2s_encoder.onnx.prototxt)
- [t2s_fsdec.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro/t2s_fsdec.onnx.prototxt)
- [t2s_sdec.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro/t2s_sdec.onnx.prototxt)
- [vits.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro/vits.onnx.prototxt)
