# Cohere Transcribe App

A Streamlit application that transcribes audio locally using the Cohere Transcribe ONNX model (2B parameters, #1 on HF Open ASR Leaderboard).

## Architecture

```
Audio File / URL
       |
       v
  librosa (load + resample to 16kHz)
       |
       v
  CohereAsrProcessor (mel spectrogram + prompt tokens)
       |
       v
  ONNX Encoder (Conformer) --> encoder hidden states
       |
       v
  ONNX Decoder (autoregressive greedy + KV cache) --> token IDs
       |
       v
  Tokenizer decode --> transcript text
```

Long audio is split into 30s chunks with 5s overlap, each chunk processed independently.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers.git
streamlit run app.py
```

Note: `transformers` must be installed from source for `cohere_asr` model support.

## Supported Languages

Arabic, Chinese, Dutch, English, French, German, Greek, Italian, Japanese, Korean, Polish, Portuguese, Spanish, Vietnamese

## Model

- Source: [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- ONNX: [onnx-community/cohere-transcribe-03-2026-ONNX](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX)
- Quantization options: INT8, Q4, FP16
- Runs on CPU (~3.5 GB RAM)
