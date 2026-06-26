# Qwen3-TTS

## Input

- Input text (String or Text file)
- Reference audio (WAV file) - For Voice Clone
- Reference text (Text file) - Transcript of the reference audio

## Output

- Synthesized audio (WAV file)

## Requirements
- Python 3.12 or higher
- [ailia SDK](https://ailia.jp/sdk/) (Version 1.6.1 or higher recommended)

Install the required Python libraries:
```bash
pip install -r requirements.txt
```

## Usage

This model supports **Voice Clone (Zero-shot Voice Conversion)** by default. You need to provide a text to synthesize, a reference audio file of the target speaker, and its corresponding text transcript.

For the sample wav,
```bash
$ python3 qwen3-tts.py
```

You can directly pass the text you want to synthesize using the `--input` argument.

```bash
python qwen3-tts.py --input "Hello, this is a test of voice cloning." --ref_audio clone_2.wav --ref_text clone_2.txt --savepath output.wav
```

### Language

By default the language is auto-detected (`--language Auto`). If the synthesized speech is pronounced in the wrong language (for example, Japanese text being read with a Chinese accent), specify the target language explicitly with the `--language` option.

```bash
python qwen3-tts.py --input "こんにちは、今日はいい天気ですね。" --language japanese
```

Supported languages: `Auto` (default), `chinese`, `english`, `japanese`, `korean`, `german`, `french`, `russian`, `portuguese`, `spanish`, `italian`.

### Options

- `-i`, `--input` Direct input text string to synthesize. (e.g. `Hello, this is a test of voice cloning.`)
- `--ref_audio` Reference audio file path for Voice Clone mode. (e.g. `clone_2.wav`)
- `--ref_text` Reference text file path containing the transcript of the reference audio. (e.g. `clone_2.txt`)
- `--language` Target language for synthesis. `Auto` for auto detection, or one of `chinese`, `english`, `japanese`, `korean`, `german`, `french`, `russian`, `portuguese`, `spanish`, `italian`. (default: `Auto`)
- `-s`, `--savepath` Save path for the output synthesized audio. (default: `output.wav`)

## Model Format

ONNX opset = 17

## Reference

- [Qwen3-TTS Official Repository](https://github.com/QwenLM/Qwen3-TTS)

## Framework

PyTorch
