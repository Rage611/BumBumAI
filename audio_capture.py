import asyncio
import audioop
import sys
from main_ui import pause_event
import pyaudio

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
        print(f"[{i}] {name}  (max input channels: {max_inputs})")
    pa.terminate()


def _device_name(info):
    return info.get("name", "Unknown")


def _device_rate(info):
    return int(info.get("defaultSampleRate") or OUTPUT_SAMPLE_RATE)


def _can_open_input(pa, device_index):
    info = pa.get_device_info_by_index(device_index)
    if info.get("maxInputChannels", 0) < 1:
        return False

    try:
        stream = pa.open(
            format=SAMPLE_WIDTH,
            channels=CHANNELS,
            rate=_device_rate(info),
            input=True,
            input_device_index=device_index,
            frames_per_buffer=FRAMES_PER_BUFFER,
        )
        stream.close()
        return True
    except OSError:
        return False


def find_input_device(pa, keywords):
    count = pa.get_device_count()
    for i in range(count):
        info = pa.get_device_info_by_index(i)
        name = _device_name(info).lower()
        if all(keyword in name for keyword in keywords) and _can_open_input(pa, i):
            return i
    return None


def resolve_system_device(system_device_index=None):
    pa = pyaudio.PyAudio()
    try:
        if system_device_index is not None and not _can_open_input(pa, system_device_index):
            print(
                f"audio_capture: system device {system_device_index} cannot be opened; auto-selecting",
                file=sys.stderr,
            )
            system_device_index = None

        if system_device_index is None:
            system_device_index = find_input_device(pa, ("stereo", "mix"))

        return system_device_index
    finally:
        pa.terminate()


def _make_safe_push(queue, chunk):
    def _push():
        try:
            queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass
    return _push


async def run_capture(queue, loop, mic_device_index=None, system_device_index=None):
    del mic_device_index

    pa = pyaudio.PyAudio()
    active = [True]

    system_device_index = resolve_system_device(system_device_index)

    if system_device_index is None:
        pa.terminate()
        raise RuntimeError("No working system audio input device found.")

    print(f"audio_capture: system device index: {system_device_index}", file=sys.stderr)

    sys_buf = bytearray()

    def _push_audio():
        while len(sys_buf) >= CHUNK_SIZE:
            chunk = bytes(sys_buf[:CHUNK_SIZE])
            del sys_buf[:CHUNK_SIZE]
            loop.call_soon_threadsafe(_make_safe_push(queue, chunk))

    def _resample(in_data, input_rate, state):
        if input_rate == OUTPUT_SAMPLE_RATE:
            return in_data, state
        return audioop.ratecv(
            in_data,
            BYTES_PER_SAMPLE,
            CHANNELS,
            input_rate,
            OUTPUT_SAMPLE_RATE,
            state,
        )

    sys_rate = [OUTPUT_SAMPLE_RATE]
    sys_rate_state = [None]

    def sys_callback(in_data, frame_count, time_info, status):
        try:
            if not active[0]:
                return (None, pyaudio.paComplete)
            if not pause_event.is_set():
                return (None, pyaudio.paContinue)
            in_data, sys_rate_state[0] = _resample(
                in_data,
                sys_rate[0],
                sys_rate_state[0],
            )
            sys_buf.extend(in_data)
            _push_audio()
            return (None, pyaudio.paContinue)
        except Exception as exc:
            print(f"audio_capture: callback error: {exc}", file=sys.stderr)
            active[0] = False
            return (None, pyaudio.paAbort)

    def _open_stream(device_index, callback):
        info = pa.get_device_info_by_index(device_index)
        rate = _device_rate(info)
        kwargs = dict(
            format=SAMPLE_WIDTH,
            channels=CHANNELS,
            rate=rate,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
            stream_callback=callback,
            input_device_index=device_index,
        )
        try:
            return pa.open(**kwargs), rate
        except OSError as exc:
            raise RuntimeError(
                f"Failed to open audio stream on device index {device_index!r}: {exc}"
            ) from exc

    sys_stream, sys_rate[0] = _open_stream(system_device_index, sys_callback)

    sys_stream.start_stream()

    try:
        while True:
            await asyncio.sleep(0.01)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        active[0] = False
        if sys_stream.is_active():
            sys_stream.stop_stream()
        await asyncio.sleep(0.1)
        sys_stream.close()
        pa.terminate()
