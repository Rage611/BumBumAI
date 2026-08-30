import asyncio
import sys
import numpy as np
from main_ui import pause_event
try:
    import pyaudiowpatch as pyaudio  # WASAPI loopback support
except ImportError:
    import pyaudio  # fallback to plain PyAudio

OUTPUT_SAMPLE_RATE = 16000
SAMPLE_WIDTH = pyaudio.paInt16
CHANNELS = 1
FRAMES_PER_BUFFER = 1024
BYTES_PER_SAMPLE = 2
CHUNK_SIZE = FRAMES_PER_BUFFER * BYTES_PER_SAMPLE


def list_audio_devices():
    pa = pyaudio.PyAudio()
    count = pa.get_device_count()
    for i in range(count):
        info = pa.get_device_info_by_index(i)
        name = info.get("name", "Unknown")
        max_inputs = info.get("maxInputChannels", 0)
        is_loopback = info.get("isLoopbackDevice", False)
        loopback_tag = " [LOOPBACK]" if is_loopback else ""
        print(f"[{i}] {name}  (max input channels: {max_inputs}){loopback_tag}")
    pa.terminate()


def _device_name(info):
    return info.get("name", "Unknown")


def _device_rate(info):
    return int(info.get("defaultSampleRate") or OUTPUT_SAMPLE_RATE)


def _resample_numpy(data: bytes, input_rate: int, output_rate: int) -> bytes:
    """
    Drop-in replacement for audioop.ratecv using numpy linear interpolation.
    Works on Python 3.13+ where audioop has been removed.
    """
    if input_rate == output_rate:
        return data
    samples = np.frombuffer(data, dtype=np.int16)
    num_output = int(len(samples) * output_rate / input_rate)
    if num_output == 0:
        return b""
    resampled = np.interp(
        np.linspace(0, len(samples), num_output, endpoint=False),
        np.arange(len(samples)),
        samples,
    ).astype(np.int16)
    return resampled.tobytes()


def find_wasapi_loopback(pa: pyaudio.PyAudio) -> int | None:
    """
    Find the best WASAPI loopback device.
    PyAudioWPATCH exposes output devices as [Loopback] input-capable devices.
    Prefer Speaker loopback (where interviewer audio plays) over Headphone or HDMI.
    """
    count = pa.get_device_count()
    loopback_devices = []

    for i in range(count):
        info = pa.get_device_info_by_index(i)
        if info.get("isLoopbackDevice", False) and info.get("maxInputChannels", 0) > 0:
            loopback_devices.append((i, info.get("name", "")))

    if not loopback_devices:
        # Fallback: search for Stereo Mix
        for i in range(count):
            info = pa.get_device_info_by_index(i)
            name = _device_name(info).lower()
            if info.get("maxInputChannels", 0) > 0 and "stereo" in name and "mix" in name:
                print(f"audio_capture: Stereo Mix fallback: [{i}] {_device_name(info)}", file=sys.stderr)
                return i
        return None

    # Prefer JBL Headphones, then Headphone, then Speaker, then anything
    for priority_keyword in ("jbl", "headphone", "speaker", ""):
        for idx, name in loopback_devices:
            if priority_keyword in name.lower():
                print(f"audio_capture: WASAPI loopback selected: [{idx}] {name!r}", file=sys.stderr)
                return idx
    return None


def _make_safe_push(queue, chunk):
    def _push():
        try:
            queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass
    return _push


async def run_capture(queue, loop, mic_device_index=None, system_device_index=None):
    del mic_device_index  # Not used — system audio only for now

    pa = pyaudio.PyAudio()
    active = [True]

    if system_device_index is None:
        system_device_index = find_wasapi_loopback(pa)

    if system_device_index is None:
        pa.terminate()
        raise RuntimeError(
            "No WASAPI loopback or Stereo Mix device found. "
            "Install PyAudioWPATCH or enable Stereo Mix in Windows sound settings."
        )

    print(f"audio_capture: capturing from device index: {system_device_index}", file=sys.stderr)

    # Pre-compute input_rate ONCE — never ask Windows inside the callback
    info = pa.get_device_info_by_index(system_device_index)
    input_rate = _device_rate(info)
    needs_resample = (input_rate != OUTPUT_SAMPLE_RATE)
    print(f"audio_capture: device rate={input_rate}, needs_resample={needs_resample}", file=sys.stderr)

    # Raw buffer — callback dumps raw bytes here, resampling happens on async side
    raw_buf = bytearray()
    drop_count = [0]

    def sys_callback(in_data, frame_count, time_info, status):
        """Ultra-fast callback — only copies bytes, no math, no Windows calls."""
        try:
            if not active[0]:
                return (None, pyaudio.paComplete)

            # Log if PortAudio reports an overflow (sound fell on the floor)
            if status:
                drop_count[0] += 1
                if drop_count[0] <= 10 or drop_count[0] % 50 == 0:
                    print(f"audio_capture: ⚠ buffer overflow #{drop_count[0]} (status={status})", file=sys.stderr)

            if not pause_event.is_set():
                return (None, pyaudio.paContinue)

            # Just copy raw bytes — no resampling, no device lookup
            raw_buf.extend(in_data)
            return (None, pyaudio.paContinue)
        except Exception as exc:
            print(f"audio_capture: callback error: {exc}", file=sys.stderr)
            active[0] = False
            return (None, pyaudio.paAbort)

    # Determine how many raw bytes make one output chunk
    # Input: input_rate samples/sec * 2 bytes/sample * (FRAMES_PER_BUFFER / input_rate) sec
    raw_frame_bytes = FRAMES_PER_BUFFER * BYTES_PER_SAMPLE  # one callback worth of raw bytes

    try:
        sys_stream = pa.open(
            format=SAMPLE_WIDTH,
            channels=CHANNELS,
            rate=input_rate,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
            stream_callback=sys_callback,
            input_device_index=system_device_index,
        )
    except OSError as exc:
        pa.terminate()
        raise RuntimeError(
            f"Failed to open audio stream on device {system_device_index!r}: {exc}"
        ) from exc

    sys_stream.start_stream()
    print("audio_capture: stream started.", file=sys.stderr)

    resampled_buf = bytearray()  # accumulates resampled 16kHz data

    try:
        while True:
            # Poll every 10ms — process whatever raw bytes the callback dumped
            await asyncio.sleep(0.01)

            if len(raw_buf) < raw_frame_bytes:
                continue

            # Grab all available raw bytes
            raw_data = bytes(raw_buf)
            raw_buf.clear()

            # Resample OUTSIDE the callback — no time pressure here
            if needs_resample:
                resampled = _resample_numpy(raw_data, input_rate, OUTPUT_SAMPLE_RATE)
            else:
                resampled = raw_data

            # Accumulate resampled data (48kHz→16kHz means 3 callbacks to fill 1 chunk)
            resampled_buf.extend(resampled)

            # Chop into fixed-size chunks and push to queue
            while len(resampled_buf) >= CHUNK_SIZE:
                chunk = bytes(resampled_buf[:CHUNK_SIZE])
                del resampled_buf[:CHUNK_SIZE]
                try:
                    queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    pass

    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        active[0] = False
        if sys_stream.is_active():
            sys_stream.stop_stream()
        await asyncio.sleep(0.1)
        sys_stream.close()
        pa.terminate()
        if drop_count[0] > 0:
            print(f"audio_capture: total buffer overflows during session: {drop_count[0]}", file=sys.stderr)
        print("audio_capture: stream closed.", file=sys.stderr)
