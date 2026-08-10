import pyaudiowpatch as pyaudio
pa = pyaudio.PyAudio()
print("=== All Audio Devices ===")
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    name = info.get("name", "Unknown")
    inputs = info.get("maxInputChannels", 0)
    is_loopback = info.get("isLoopbackDevice", False)
    print(f"[{i}] {name!r}  inputs={inputs}  loopback={is_loopback}")
pa.terminate()
