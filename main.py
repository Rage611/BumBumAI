import sys
import asyncio
import threading
import os
from dotenv import load_dotenv

from PyQt6.QtWidgets import QApplication

from main_ui import MainWindow, signals, hotkey_signals
import audio_capture
import stt_client
import llm_client

try:
    import keyboard
except ImportError:
    keyboard = None

loop = None
async_thread = None
DEEPGRAM_API_KEY = ""
GROQ_API_KEY = ""

def _env_device_index(name):
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        print(f"WARNING: {name} must be a number; ignoring {value!r}", file=sys.stderr)
        return None

async def _main_pipeline():
    audio_queue = asyncio.Queue(maxsize=200)
    llm_queue = asyncio.Queue(maxsize=10)
    mic_device_index = _env_device_index("MIC_DEVICE_INDEX")
    system_device_index = _env_device_index("SYSTEM_DEVICE_INDEX")
    try:
        await asyncio.gather(
            audio_capture.run_capture(audio_queue, loop, mic_device_index, system_device_index),
            stt_client.run_stt(audio_queue, llm_queue, signals, DEEPGRAM_API_KEY),
            llm_client.run_llm(llm_queue, signals, GROQ_API_KEY),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"main: pipeline error: {exc}", file=sys.stderr)
        signals.status_update.emit("disconnected")

def on_quit():
    async def shutdown():
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    asyncio.run_coroutine_threadsafe(shutdown(), loop)
    if keyboard is not None:
        keyboard.unhook_all_hotkeys()
    async_thread.join(timeout=2)

if __name__ == "__main__":
    load_dotenv()
    DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

    if not DEEPGRAM_API_KEY:
        print("ERROR: DEEPGRAM_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    audio_capture.list_audio_devices()

    loop = asyncio.new_event_loop()

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    async_thread = threading.Thread(
        target=loop.run_forever,
        name="AsyncWorker",
        daemon=True,
    )
    async_thread.start()

    asyncio.run_coroutine_threadsafe(_main_pipeline(), loop)

    if keyboard is None:
        print("main: keyboard module not installed; global hotkeys disabled.", file=sys.stderr)
    else:
        try:
            keyboard.add_hotkey("f9", lambda: hotkey_signals.toggle_pause.emit(), suppress=True)
            keyboard.add_hotkey("f10", lambda: hotkey_signals.clear_ui.emit(), suppress=True)
        except Exception as exc:
            print(f"main: failed to register global hotkeys: {exc}", file=sys.stderr)

    app.aboutToQuit.connect(on_quit)

    sys.exit(app.exec())
