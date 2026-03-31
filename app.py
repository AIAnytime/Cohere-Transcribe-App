import streamlit as st
import os
import sys
import tempfile
import time
import json
import numpy as np
from pathlib import Path
from datetime import timedelta

import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor
import librosa

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_ID = "onnx-community/cohere-transcribe-03-2026-ONNX"

LANGUAGES = {
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Polish": "pl",
    "Greek": "el",
    "Arabic": "ar",
    "Vietnamese": "vi",
    "Chinese (Mandarin)": "zh",
    "Japanese": "ja",
    "Korean": "ko",
}

QUANTIZATION_OPTIONS = {
    "INT8 (Recommended)": {
        "encoder": "onnx/encoder_model_quantized.onnx",
        "decoder": "onnx/decoder_model_merged_quantized.onnx",
        "size": "~3 GB",
    },
    "Q4 (Smallest & fastest)": {
        "encoder": "onnx/encoder_model_q4.onnx",
        "decoder": "onnx/decoder_model_merged_q4.onnx",
        "size": "~2.1 GB",
    },
    "FP16 (Higher quality)": {
        "encoder": "onnx/encoder_model_fp16.onnx",
        "decoder": "onnx/decoder_model_merged_fp16.onnx",
        "size": "~3.8 GB",
    },
}

N_LAYERS = 8
N_HEADS = 8
HEAD_DIM = 128
MAX_NEW_TOKENS = 448
CHUNK_LENGTH_S = 30
OVERLAP_S = 5
SAMPLE_RATE = 16000

SUPPORTED_AUDIO_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "webm", "aac", "wma", "opus"]


# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Cohere Transcribe",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.7;
    }
    .stat-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .stat-card h3 { margin: 0; font-size: 1.5rem; color: #2d3748; }
    .stat-card p  { margin: 0; font-size: 0.85rem; color: #718096; }
    .transcript-box {
        background: #f8f9fa;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        font-size: 1.05rem;
        line-height: 1.8;
        max-height: 500px;
        overflow-y: auto;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# ONNX Transcriber (raw onnxruntime)
# ──────────────────────────────────────────────
class OnnxTranscriber:
    """Runs Cohere-Transcribe via raw ONNX Runtime sessions."""

    def __init__(self, encoder_path, decoder_path, processor):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.inter_op_num_threads = os.cpu_count()
        opts.intra_op_num_threads = os.cpu_count()

        self.encoder = ort.InferenceSession(encoder_path, sess_options=opts)
        self.decoder = ort.InferenceSession(decoder_path, sess_options=opts)
        self.processor = processor

        # Cache decoder I/O names
        self._dec_in_names = [i.name for i in self.decoder.get_inputs()]
        self._dec_out_names = [o.name for o in self.decoder.get_outputs()]

    # ── Encoder ──────────────────────────────
    def encode(self, input_features):
        """input_features: np float32 (1, seq, 128) -> (1, enc_seq, 1024)"""
        (hidden,) = self.encoder.run(
            ["last_hidden_state"],
            {"input_features": input_features},
        )
        return hidden

    # ── Decoder (greedy, one chunk) ──────────
    def decode_greedy(self, encoder_hidden, prompt_ids, eos_id):
        """Autoregressive greedy decode for a single encoder chunk."""

        batch = 1
        generated = list(prompt_ids)
        prompt_len = len(prompt_ids)

        # Initial KV caches – empty
        past = {}
        for i in range(N_LAYERS):
            for loc in ("decoder", "encoder"):
                for kv in ("key", "value"):
                    past[f"past_key_values.{i}.{loc}.{kv}"] = np.zeros(
                        (batch, N_HEADS, 0, HEAD_DIM), dtype=np.float32
                    )

        # First pass: feed all prompt tokens
        input_ids = np.array([prompt_ids], dtype=np.int64)
        position_ids = np.arange(prompt_len, dtype=np.int64).reshape(1, -1)
        attn_mask = np.ones((batch, prompt_len), dtype=np.int64)

        feeds = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "position_ids": position_ids,
            "num_logits_to_keep": np.array(1, dtype=np.int64),
            "encoder_hidden_states": encoder_hidden,
            **past,
        }

        outputs = self.decoder.run(self._dec_out_names, feeds)
        logits = outputs[0]  # (1, 1, vocab)
        next_id = int(np.argmax(logits[0, -1]))

        # Update KV caches from outputs[1:]
        present = {}
        for name, val in zip(self._dec_out_names[1:], outputs[1:]):
            present[name.replace("present.", "past_key_values.")] = val

        if next_id == eos_id:
            return generated[prompt_len:]

        generated.append(next_id)
        cur_pos = prompt_len

        # Autoregressive loop
        for _ in range(MAX_NEW_TOKENS - 1):
            cur_pos += 1
            past_len = present[f"past_key_values.0.decoder.key"].shape[2]

            input_ids = np.array([[next_id]], dtype=np.int64)
            position_ids = np.array([[cur_pos]], dtype=np.int64)
            attn_mask = np.ones((batch, past_len + 1), dtype=np.int64)

            feeds = {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "position_ids": position_ids,
                "num_logits_to_keep": np.array(1, dtype=np.int64),
                "encoder_hidden_states": encoder_hidden,
                **present,
            }

            outputs = self.decoder.run(self._dec_out_names, feeds)
            logits = outputs[0]
            next_id = int(np.argmax(logits[0, -1]))

            new_present = {}
            for name, val in zip(self._dec_out_names[1:], outputs[1:]):
                new_present[name.replace("present.", "past_key_values.")] = val
            present = new_present

            if next_id == eos_id:
                break
            generated.append(next_id)

        return generated[prompt_len:]

    # ── Full transcription with chunking ─────
    def transcribe(self, audio_path, language="en", progress_cb=None):
        """Transcribe an audio file. Returns dict with 'text' key."""
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        total_samples = len(audio)
        chunk_samples = CHUNK_LENGTH_S * SAMPLE_RATE
        overlap_samples = OVERLAP_S * SAMPLE_RATE
        stride_samples = chunk_samples - overlap_samples

        # Build chunks
        chunks = []
        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunks.append(audio[start:end])
            start += stride_samples

        eos_id = self.processor.tokenizer.eos_token_id or 3
        all_texts = []

        for idx, chunk_audio in enumerate(chunks):
            if progress_cb:
                progress_cb(idx, len(chunks))

            # Preprocess
            inputs = self.processor(
                audio=chunk_audio,
                language=language,
                sampling_rate=SAMPLE_RATE,
                return_tensors="np",
            )
            input_features = inputs.input_features
            prompt_ids = inputs.decoder_input_ids[0].tolist()

            # Encode
            encoder_hidden = self.encode(input_features)

            # Decode
            token_ids = self.decode_greedy(encoder_hidden, prompt_ids, eos_id)

            # Decode tokens to text
            text = self.processor.tokenizer.decode(token_ids, skip_special_tokens=True)
            all_texts.append(text.strip())

        if progress_cb:
            progress_cb(len(chunks), len(chunks))

        return {"text": " ".join(all_texts)}


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_transcriber(quant_key):
    """Download ONNX files and build the transcriber."""
    variant = QUANTIZATION_OPTIONS[quant_key]

    enc_path = hf_hub_download(MODEL_ID, variant["encoder"])
    # Also download the .onnx_data sidecar(s) so onnxruntime can find them
    _download_data_files(variant["encoder"])

    dec_path = hf_hub_download(MODEL_ID, variant["decoder"])
    _download_data_files(variant["decoder"])

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return OnnxTranscriber(enc_path, dec_path, processor)


def _download_data_files(onnx_file):
    """Download .onnx_data sidecar files (weights stored externally by ONNX)."""
    base = onnx_file + "_data"
    # Try the single-file sidecar first
    try:
        hf_hub_download(MODEL_ID, base)
    except Exception:
        pass
    # Try numbered shards (_data_1, _data_2, …)
    for i in range(1, 10):
        try:
            hf_hub_download(MODEL_ID, f"{base}_{i}")
        except Exception:
            break


# ──────────────────────────────────────────────
# Audio Helpers
# ──────────────────────────────────────────────
def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def download_audio_from_url(url):
    import subprocess
    tmp_dir = tempfile.mkdtemp()
    out_template = os.path.join(tmp_dir, "audio.%(ext)s")

    yt_dlp_bin = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
    if not os.path.exists(yt_dlp_bin):
        yt_dlp_bin = "yt-dlp"

    cmd = [
        yt_dlp_bin, "--extract-audio", "--audio-format", "wav",
        "--audio-quality", "0", "--no-playlist", "-o", out_template, url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr[:500]}")

    for f in sorted(os.listdir(tmp_dir)):
        if f.endswith(".wav"):
            return os.path.join(tmp_dir, f)
    raise FileNotFoundError("Could not find the downloaded audio file.")


def get_audio_duration(path):
    return librosa.get_duration(path=path)


def fmt_duration(seconds):
    td = timedelta(seconds=int(seconds))
    parts = str(td).split(":")
    if int(parts[0]) == 0:
        return f"{int(parts[1])}:{parts[2]}"
    return str(td)


# ──────────────────────────────────────────────
# UI Components
# ──────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>Cohere Transcribe</h1>
        <p>State-of-the-art speech recognition &middot; 2B params &middot; 14 languages &middot; runs locally via ONNX</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()


def render_sidebar():
    with st.sidebar:
        st.markdown("## Settings")

        quant = st.radio(
            "Model Quantization",
            list(QUANTIZATION_OPTIONS.keys()),
            index=0,
            help="Trade-off between model size and transcription quality",
        )
        st.caption(f"Download: {QUANTIZATION_OPTIONS[quant]['size']}")

        st.markdown("---")
        lang = st.selectbox("Language", list(LANGUAGES.keys()), index=0)

        st.markdown("---")
        st.markdown(
            "**Model:** [cohere-transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)  \n"
            "**ONNX:** [onnx-community](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX)  \n"
            "**#1** on HF Open ASR Leaderboard"
        )
        st.caption("Peak RAM: ~3.5-4 GB")

    return quant, LANGUAGES[lang]


def render_stats(duration, elapsed, text):
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f'<div class="stat-card"><h3>{fmt_duration(duration)}</h3><p>Audio Duration</p></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="stat-card"><h3>{elapsed:.1f}s</h3><p>Processing Time</p></div>', unsafe_allow_html=True)
    with cols[2]:
        speed = duration / elapsed if elapsed > 0 else 0
        st.markdown(f'<div class="stat-card"><h3>{speed:.1f}x</h3><p>Realtime Speed</p></div>', unsafe_allow_html=True)
    with cols[3]:
        wc = len(text.split())
        st.markdown(f'<div class="stat-card"><h3>{wc:,}</h3><p>Words</p></div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    render_header()
    quant_key, lang_code = render_sidebar()

    # ── Model loading ──
    status = st.empty()
    status.info("Loading model… (first run downloads ONNX weights — may take a few minutes)")
    try:
        transcriber = load_transcriber(quant_key)
        status.success("Model loaded!")
        time.sleep(0.5)
        status.empty()
    except Exception as e:
        status.error(f"Failed to load model: {e}")
        st.exception(e)
        st.stop()

    # ── Input ──
    tab_upload, tab_url = st.tabs(["Upload Audio File", "From URL"])
    audio_path = None
    audio_name = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Drop an audio file here",
            type=SUPPORTED_AUDIO_TYPES,
            help="WAV, MP3, M4A, FLAC, OGG, WebM, AAC, WMA, Opus",
        )
        if uploaded:
            audio_path = save_uploaded_file(uploaded)
            audio_name = uploaded.name
            st.audio(uploaded)

    with tab_url:
        st.caption("YouTube, Twitter/X, TikTok, and 1000+ sites via yt-dlp")
        url = st.text_input("Paste a URL", placeholder="https://www.youtube.com/watch?v=...")
        if url:
            if st.button("Download Audio", use_container_width=True):
                with st.spinner("Downloading…"):
                    try:
                        audio_path = download_audio_from_url(url)
                        audio_name = Path(audio_path).name
                        st.session_state["url_audio_path"] = audio_path
                        st.session_state["url_audio_name"] = audio_name
                        st.success("Audio downloaded!")
                    except Exception as e:
                        st.error(f"Download failed: {e}")
            if "url_audio_path" in st.session_state:
                audio_path = st.session_state["url_audio_path"]
                audio_name = st.session_state["url_audio_name"]
                st.audio(audio_path)

    # ── Transcribe ──
    if audio_path and os.path.exists(audio_path):
        st.markdown("")
        if st.button("Transcribe", type="primary", use_container_width=True):
            try:
                duration = get_audio_duration(audio_path)
            except Exception:
                duration = 0

            progress = st.progress(0, text="Transcribing…")

            def on_progress(done, total):
                pct = int(100 * done / total) if total else 0
                progress.progress(pct, text=f"Chunk {done}/{total}")

            start = time.time()
            try:
                result = transcriber.transcribe(audio_path, language=lang_code, progress_cb=on_progress)
                elapsed = time.time() - start
                progress.progress(100, text="Done!")
                time.sleep(0.3)
                progress.empty()

                st.session_state["result"] = result
                st.session_state["elapsed"] = elapsed
                st.session_state["duration"] = duration
                st.session_state["audio_name"] = audio_name
            except Exception as e:
                progress.empty()
                st.error(f"Transcription failed: {e}")
                st.exception(e)

    # ── Results ──
    if "result" in st.session_state:
        result = st.session_state["result"]
        elapsed = st.session_state["elapsed"]
        duration = st.session_state["duration"]
        audio_name = st.session_state.get("audio_name", "audio")
        text = result.get("text", "").strip()

        st.markdown("---")
        st.markdown("### Transcription Results")
        if duration > 0:
            render_stats(duration, elapsed, text)
            st.markdown("")

        st.markdown(f'<div class="transcript-box">{text}</div>', unsafe_allow_html=True)
        st.markdown("")

        col1, col2 = st.columns(2)
        base = Path(audio_name).stem if audio_name else "transcription"
        with col1:
            st.download_button(
                "Download .txt", data=text,
                file_name=f"{base}_transcript.txt", mime="text/plain",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "Download .json", data=json.dumps(result, indent=2, default=str),
                file_name=f"{base}_transcript.json", mime="application/json",
                use_container_width=True,
            )

        with st.expander("Raw text (select & copy)", expanded=False):
            st.text_area("", value=text, height=300, label_visibility="collapsed")


if __name__ == "__main__":
    main()
