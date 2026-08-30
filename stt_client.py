import asyncio
import sys
import os
import io
import wave
import struct
import math
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------
# Add NVIDIA CUDA DLL directories to PATH (pip-installed wheels)
# ---------------------------------------------------------------------------
_venv_nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
if _venv_nvidia.is_dir():
    for _dll_dir in _venv_nvidia.rglob("bin"):
        if _dll_dir.is_dir():
            os.add_dll_directory(str(_dll_dir))
            os.environ["PATH"] = str(_dll_dir) + os.pathsep + os.environ.get("PATH", "")

# ---------------------------------------------------------------------------
# Faster-Whisper (local, free) — loaded once at startup
# ---------------------------------------------------------------------------
try:
    from faster_whisper import WhisperModel
    _whisper_model: WhisperModel | None = None

    def _get_whisper_model(model_size: str = "base.en") -> WhisperModel:
        global _whisper_model
        if _whisper_model is None:
            # Try GPU first, fall back to CPU
            try:
                print(f"stt_client: Loading Faster-Whisper model '{model_size}' on GPU (CUDA)...", file=sys.stderr)
                _whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")
                print("stt_client: Faster-Whisper model loaded on GPU.", file=sys.stderr)
            except Exception as gpu_exc:
                print(f"stt_client: GPU failed ({gpu_exc}), falling back to CPU...", file=sys.stderr)
                _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
                print("stt_client: Faster-Whisper model loaded on CPU.", file=sys.stderr)
        return _whisper_model

except ImportError:
    WhisperModel = None
    _whisper_model = None
    print("stt_client: faster-whisper not installed. Run: pip install faster-whisper", file=sys.stderr)

# ---------------------------------------------------------------------------
# WebRTC VAD — accurate voice activity detection
# ---------------------------------------------------------------------------
try:
    try:
        import webrtcvad  # original (requires build tools)
    except ImportError:
        import webrtcvad_wheels as webrtcvad  # pre-built wheels version
    _vad_available = True
except ImportError:
    webrtcvad = None
    _vad_available = False
    print("stt_client: webrtcvad not installed. Falling back to RMS VAD.", file=sys.stderr)

import config as _config

# ---------------------------------------------------------------------------
# Timing config
# Each chunk from audio_capture.py = 1024 samples @ 16000 Hz = 64ms
# WebRTC VAD needs 10ms, 20ms, or 30ms frames → we use 20ms = 320 samples
# ---------------------------------------------------------------------------

VAD_FRAME_SAMPLES     = 320          # 20ms @ 16kHz
VAD_FRAME_BYTES       = VAD_FRAME_SAMPLES * 2  # int16
VAD_AGGRESSIVENESS    = 2            # 0–3; 2 = balanced
MIN_SPEECH_FRAMES     = 5            # At least 5 VAD frames (100ms) of speech
SILENCE_END_FRAMES    = 15           # 15 consecutive silent frames (300ms) = end of utterance
MAX_BUFFER_CHUNKS     = 80           # Hard cap: 8s of audio

# ---------------------------------------------------------------------------
# Fallback RMS VAD (if webrtcvad unavailable)
# ---------------------------------------------------------------------------
SILENCE_RMS_THRESHOLD = 50
MIN_SPEECH_RATIO      = 0.10

def _compute_rms(pcm_bytes: bytes) -> float:
    if len(pcm_bytes) < 2:
        return 0.0
    num_samples = len(pcm_bytes) // 2
    samples = struct.unpack_from(f"<{num_samples}h", pcm_bytes)
    rms = math.sqrt(sum(s * s for s in samples) / num_samples)
    return rms

# ---------------------------------------------------------------------------
# Whisper hallucination filter
# ---------------------------------------------------------------------------
_HALLUCINATIONS = {
    "", ".", "..", "...", "....", ".....", " ", "  ",
    "you", "You", "the", "The", "a", "A",
    "uh", "um", "hmm", "hm", "ah", "oh", "Oh",
    "ok", "OK", "okay", "Okay",
    "bye", "Bye",
    "thank you", "Thank you", "Thanks", "thanks",
    "Thank you.", "Thank you!",
    "you.", "you..", "you...",
    "You are new to me", "You are new to me.",
    "You are welcome", "You are welcome.",
    "I'm sorry", "I'm sorry.",
    "I don't know", "I don't know.",
    "I see", "I see.",
    "I know", "I know.",
    "Hello", "Hello.", "Hi", "Hi.",
    "Bye bye", "Bye bye.",
    "Subtitles by", "Subtitles by the",
    "[ Silence ]", "[Silence]", "(silence)", "(Silence)",
    "[ BLANK_AUDIO ]", "[BLANK_AUDIO]",
    "www.mooji.org",
}

def _is_hallucination(text: str) -> bool:
    stripped = text.strip().strip(".,!?…")
    if stripped in _HALLUCINATIONS or text.strip() in _HALLUCINATIONS:
        return True
    words = stripped.split()
    if len(words) <= 1:
        return True
    non_dot = sum(1 for c in text if c not in " .")
    if len(text) > 0 and non_dot / len(text) < 0.2:
        return True
    return False


# ---------------------------------------------------------------------------
# Local Faster-Whisper transcription — in-memory, no disk I/O
# ---------------------------------------------------------------------------

async def _transcribe(chunks: list[bytes], model_size: str = "base.en") -> str | None:
    """Transcribe audio chunks using local Faster-Whisper. No temp files."""
    if WhisperModel is None:
        print("stt_client: faster-whisper not available.", file=sys.stderr)
        return None

    audio_data = b"".join(chunks)

    # Write WAV to in-memory buffer — no disk I/O
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_data)
    buf.seek(0)

    try:
        # Run blocking Faster-Whisper in thread pool to not block event loop
        model = _get_whisper_model(model_size)
        segments, info = await asyncio.to_thread(
            model.transcribe,
            buf,
            language="en",
            beam_size=5,
            vad_filter=False,  # We handle VAD ourselves
        )
        text = " ".join(seg.text for seg in segments).strip()
        if not text or _is_hallucination(text):
            return None
        return text
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"stt_client: Faster-Whisper error: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# WebRTC VAD helper — splits 64ms chunks into 20ms frames
# ---------------------------------------------------------------------------

def _is_speech_webrtc(vad_instance, chunk: bytes) -> bool:
    """Check if a 64ms chunk contains speech by splitting into 20ms frames."""
    speech_frames = 0
    total_frames = 0
    offset = 0
    while offset + VAD_FRAME_BYTES <= len(chunk):
        frame = chunk[offset:offset + VAD_FRAME_BYTES]
        try:
            if vad_instance.is_speech(frame, 16000):
                speech_frames += 1
        except Exception:
            pass
        total_frames += 1
        offset += VAD_FRAME_BYTES
    if total_frames == 0:
        return False
    return (speech_frames / total_frames) >= 0.5


# ---------------------------------------------------------------------------
# Main STT loop — Faster-Whisper with WebRTC VAD
# ---------------------------------------------------------------------------

async def run_stt(audio_queue: asyncio.Queue, llm_queue: asyncio.Queue,
                  signals, openai_api_key: str = ""):
    """
    Local STT pipeline:
      Audio Queue → WebRTC VAD → Utterance Buffer
        → 300ms silence → Faster-Whisper → Hallucination filter → LLM Queue
    """
    cfg = _config.load_config()
    model_size = cfg.get("WHISPER_MODEL", "base.en")

    # Pre-load the model so first transcription has no delay
    if WhisperModel is not None:
        await asyncio.to_thread(_get_whisper_model, model_size)
    else:
        print("stt_client: faster-whisper unavailable — STT disabled.", file=sys.stderr)
        signals.status_update.emit("disconnected")
        return

    # Set up VAD
    if _vad_available:
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        print(f"stt_client: WebRTC VAD active (aggressiveness={VAD_AGGRESSIVENESS}).", file=sys.stderr)
    else:
        vad = None
        print("stt_client: Using RMS fallback VAD.", file=sys.stderr)

    signals.status_update.emit("connected")
    print(f"stt_client: Ready — model='{model_size}', VAD={'webrtcvad' if vad else 'rms'}.", file=sys.stderr)

    utterance_buf: list[bytes] = []
    silence_count = 0
    speech_frame_count = 0
    speaking = False

    async def _flush(buf: list[bytes]):
        """Send buffer to Faster-Whisper and forward result to LLM queue."""
        if not buf:
            return
        transcript = await _transcribe(buf, model_size)
        if transcript:
            print(f"stt_client [Whisper]: {transcript!r}", file=sys.stderr)
            signals.interim_transcript.emit(transcript)
            signals.final_transcript.emit(transcript)
            try:
                llm_queue.put_nowait(transcript)
            except asyncio.QueueFull:
                pass

    try:
        while True:
            chunk = await audio_queue.get()

            # Determine if this chunk has speech
            if vad is not None:
                is_loud = _is_speech_webrtc(vad, chunk)
            else:
                is_loud = _compute_rms(chunk) > SILENCE_RMS_THRESHOLD

            if is_loud:
                utterance_buf.append(chunk)
                silence_count = 0
                speech_frame_count += 1
                speaking = True
            else:
                if speaking:
                    utterance_buf.append(chunk)  # keep trailing silence for context
                    silence_count += 1
                    if silence_count >= SILENCE_END_FRAMES and speech_frame_count >= MIN_SPEECH_FRAMES:
                        buf_copy = utterance_buf[:]
                        utterance_buf = []
                        silence_count = 0
                        speech_frame_count = 0
                        speaking = False
                        asyncio.create_task(_flush(buf_copy))

            # Hard cap: flush if buffer is getting too long
            if len(utterance_buf) >= MAX_BUFFER_CHUNKS:
                buf_copy = utterance_buf[:]
                utterance_buf = []
                silence_count = 0
                speech_frame_count = 0
                speaking = False
                asyncio.create_task(_flush(buf_copy))

    except asyncio.CancelledError:
        if utterance_buf and speaking:
            await _flush(utterance_buf)
        print("stt_client: shutting down.", file=sys.stderr)
        raise