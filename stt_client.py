import asyncio
import sys
import wave
import struct
import math
import tempfile
import os
from collections import deque

from openai import AsyncOpenAI
import config as _config

# ---------------------------------------------------------------------------
# Timing config (Whisper batch mode)
# Each chunk from audio_capture.py = 1600 samples @ 16000 Hz = 0.1 seconds
# ---------------------------------------------------------------------------

# End-of-utterance detection config
# Instead of waiting a fixed window, we fire Whisper the moment the
# person stops talking (silence after speech).
MIN_SPEECH_CHUNKS   = 8   # Need at least 0.8s of speech before triggering
SILENCE_END_CHUNKS  = 5   # 0.5s of silence after speech = end of utterance
MAX_BUFFER_CHUNKS   = 80  # Rolling 8-second context window (hard cap)

# ---------------------------------------------------------------------------
# Voice Activity Detection (energy-based) — used by Whisper path only
# Whisper hallucinates (".", "you", "Thank you") when given silence.
# ---------------------------------------------------------------------------
SILENCE_RMS_THRESHOLD = 50  # 0–32768 range; lower = more sensitive
MIN_SPEECH_RATIO = 0.10     # At least 10% of chunks must be "loud"

# ---------------------------------------------------------------------------
# Whisper hallucination filter
# ---------------------------------------------------------------------------
_HALLUCINATIONS = {
    # Empty / whitespace
    "", ".", "..", "...", "....", ".....", " ", "  ",
    # Single filler words
    "you", "You", "the", "The", "a", "A",
    "uh", "um", "hmm", "hm", "ah", "oh", "Oh",
    "ok", "OK", "okay", "Okay",
    "bye", "Bye",
    # Common Whisper silence hallucinations
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


def _compute_rms(pcm_bytes: bytes) -> float:
    if len(pcm_bytes) < 2:
        return 0.0
    num_samples = len(pcm_bytes) // 2
    samples = struct.unpack_from(f"<{num_samples}h", pcm_bytes)
    rms = math.sqrt(sum(s * s for s in samples) / num_samples)
    return rms


def _has_speech(chunks: list[bytes]) -> bool:
    if not chunks:
        return False
    loud = sum(1 for c in chunks if _compute_rms(c) > SILENCE_RMS_THRESHOLD)
    return (loud / len(chunks)) >= MIN_SPEECH_RATIO


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
# Deepgram streaming STT
# ---------------------------------------------------------------------------

async def _run_deepgram(audio_queue: asyncio.Queue, llm_queue: asyncio.Queue,
                        signals, api_key: str):
    """Live streaming STT via Deepgram Nova-2."""
    try:
        from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
    except ImportError:
        print("stt_client: deepgram-sdk not installed. Run: pip install deepgram-sdk",
              file=sys.stderr)
        return

    dg = DeepgramClient(api_key)
    connection = dg.listen.asyncwebsocket.v("1")

    async def on_message(self, result, **kwargs):
        try:
            sentence = result.channel.alternatives[0].transcript.strip()
            if not sentence:
                return
            is_final = result.is_final
            if is_final:
                print(f"stt_client [Deepgram]: {sentence!r}", file=sys.stderr)
                signals.interim_transcript.emit(sentence)
                signals.final_transcript.emit(sentence)
                await llm_queue.put(sentence)
            else:
                signals.interim_transcript.emit(sentence)
        except Exception as exc:
            print(f"stt_client: Deepgram message error: {exc}", file=sys.stderr)

    async def on_error(self, error, **kwargs):
        print(f"stt_client: Deepgram error: {error}", file=sys.stderr)
        signals.status_update.emit("disconnected")

    connection.on(LiveTranscriptionEvents.Transcript, on_message)
    connection.on(LiveTranscriptionEvents.Error, on_error)

    options = LiveOptions(
        model="nova-2",
        language="en",
        encoding="linear16",
        sample_rate=16000,
        channels=1,
        interim_results=True,
        utterance_end_ms=1000,
        vad_events=True,
    )

    print("stt_client: Connecting to Deepgram...", file=sys.stderr)
    started = await connection.start(options)
    if not started:
        print("stt_client: Deepgram connection failed.", file=sys.stderr)
        signals.status_update.emit("disconnected")
        return

    print("stt_client: Deepgram connected.", file=sys.stderr)
    signals.status_update.emit("connected")

    try:
        while True:
            chunk = await audio_queue.get()
            await connection.send(chunk)
    except asyncio.CancelledError:
        print("stt_client: Deepgram shutting down...", file=sys.stderr)
        await connection.finish()
        raise


# ---------------------------------------------------------------------------
# Whisper batch STT  —  end-of-utterance triggered
# ---------------------------------------------------------------------------

async def _transcribe(client: AsyncOpenAI, chunks: list[bytes]) -> str | None:
    """Send audio chunks to Whisper and return cleaned transcript or None."""
    audio_data = b"".join(chunks)
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as f:
            with wave.open(f, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data)
        with open(temp_path, "rb") as audio_file:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="en",
            )
        text = response.strip()
        return None if _is_hallucination(text) else text
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"stt_client: Whisper error: {exc}", file=sys.stderr)
        return None
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


async def _run_whisper(audio_queue: asyncio.Queue, llm_queue: asyncio.Queue,
                       signals, api_key: str):
    """
    End-of-utterance triggered Whisper STT.

    State machine:
      SILENCE  →  (speech chunk detected)  →  SPEAKING
      SPEAKING →  (silence for 0.5s)        →  TRIGGER Whisper → SILENCE
      SPEAKING →  (buffer hits hard cap)    →  TRIGGER Whisper → SILENCE
    """
    client = AsyncOpenAI(api_key=api_key)
    print("stt_client: Whisper STT initialised (end-of-utterance mode).", file=sys.stderr)
    signals.status_update.emit("connected")

    utterance_buf: list[bytes] = []   # chunks for the current utterance
    silence_count = 0                 # consecutive silent chunks since last speech
    speaking = False

    async def _flush(buf: list[bytes]):
        """Send buffer to Whisper and forward result."""
        if not buf:
            return
        transcript = await _transcribe(client, buf)
        if transcript:
            print(f"stt_client [Whisper]: {transcript!r}", file=sys.stderr)
            signals.interim_transcript.emit(transcript)
            signals.final_transcript.emit(transcript)
            await llm_queue.put(transcript)

    try:
        while True:
            chunk = await audio_queue.get()
            is_loud = _compute_rms(chunk) > SILENCE_RMS_THRESHOLD

            if is_loud:
                # Active speech
                utterance_buf.append(chunk)
                silence_count = 0
                speaking = True
            else:
                # Silence chunk
                if speaking:
                    utterance_buf.append(chunk)   # include trailing silence for context
                    silence_count += 1
                    if silence_count >= SILENCE_END_CHUNKS:
                        # Person stopped talking → fire immediately
                        buf_copy = utterance_buf[:]
                        utterance_buf = []
                        silence_count = 0
                        speaking = False
                        asyncio.create_task(_flush(buf_copy))

            # Hard cap: send if buffer gets very long (e.g., long continuous speech)
            if len(utterance_buf) >= MAX_BUFFER_CHUNKS:
                buf_copy = utterance_buf[:]
                utterance_buf = []
                silence_count = 0
                speaking = False
                asyncio.create_task(_flush(buf_copy))

    except asyncio.CancelledError:
        # Flush any remaining speech on shutdown
        if utterance_buf and speaking:
            await _flush(utterance_buf)
        print("stt_client: Whisper shutting down...", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# Public entry point — dispatches to Deepgram or Whisper based on config
# ---------------------------------------------------------------------------

async def run_stt(audio_queue: asyncio.Queue, llm_queue: asyncio.Queue,
                  signals, openai_api_key: str):
    cfg = _config.load_config()
    provider = cfg.get("STT_PROVIDER", "whisper").lower()
    deepgram_key = cfg.get("DEEPGRAM_API_KEY", "")

    if provider == "deepgram" and deepgram_key:
        print("stt_client: Using Deepgram Nova-2 (streaming).", file=sys.stderr)
        await _run_deepgram(audio_queue, llm_queue, signals, deepgram_key)
    else:
        if provider == "deepgram" and not deepgram_key:
            print("stt_client: Deepgram key missing — falling back to Whisper.",
                  file=sys.stderr)
        if not openai_api_key:
            print("stt_client: OPENAI_API_KEY missing. STT disabled.", file=sys.stderr)
            return
        await _run_whisper(audio_queue, llm_queue, signals, openai_api_key)