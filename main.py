import sys
import asyncio
import threading
import os

from PyQt6.QtWidgets import QApplication

from main_ui import MainWindow, SettingsDialog, signals, hotkey_signals
import audio_capture
import stt_client
import llm_client
import config

try:
    import keyboard
except ImportError:
    keyboard = None


loop = None
async_thread = None


def _env_device_index(name):
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        print(f"WARNING: {name} must be a number; ignoring {value!r}", file=sys.stderr)
        return None


async def _main_pipeline(groq_keys: list[str]):
    audio_queue = asyncio.Queue(maxsize=200)
    llm_queue = asyncio.Queue(maxsize=10)
    system_device_index = _env_device_index("SYSTEM_DEVICE_INDEX")
    try:
        await asyncio.gather(
            audio_capture.run_capture(audio_queue, loop, None, system_device_index),
            stt_client.run_stt(audio_queue, llm_queue, signals),
            llm_client.run_llm(llm_queue, signals, groq_keys),
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
    app = QApplication(sys.argv)

    keys = config.load_config()
    groq_keys = keys.get("GROQ_API_KEYS", [keys.get("GROQ_API_KEY", "")])
    google_keys = keys.get("GOOGLE_API_KEYS", [])

    # Prompt for settings if no keys configured at all
    if not any(k.strip() for k in groq_keys):
        dialog = SettingsDialog()
        if dialog.exec() == 1:
            keys = config.load_config()
            groq_keys = keys.get("GROQ_API_KEYS", [])
            google_keys = keys.get("GOOGLE_API_KEYS", [])

    if not any(k.strip() for k in groq_keys):
        print("ERROR: At least one Groq API key is required to run Parakeet.", file=sys.stderr)
        sys.exit(1)

    # List audio devices for debugging
    audio_capture.list_audio_devices()

    # Configure vision (Google Gemini)
    llm_client.configure_vision(google_keys)

    loop = asyncio.new_event_loop()

    window = MainWindow(loop=loop)
    window.show()

    async_thread = threading.Thread(
        target=loop.run_forever,
        name="AsyncWorker",
        daemon=True,
    )
    async_thread.start()

    asyncio.run_coroutine_threadsafe(_main_pipeline(groq_keys), loop)

    if keyboard is None:
        print("main: keyboard module not installed; global hotkeys disabled.", file=sys.stderr)
    else:
        try:
            keyboard.add_hotkey("f7", lambda: hotkey_signals.toggle_vision.emit(), suppress=True)
            keyboard.add_hotkey("f8", lambda: hotkey_signals.open_settings.emit(), suppress=True)
            keyboard.add_hotkey("f9", lambda: hotkey_signals.toggle_pause.emit(), suppress=True)
            keyboard.add_hotkey("f10", lambda: hotkey_signals.clear_ui.emit(), suppress=True)
        except Exception as exc:
            print(f"main: failed to register global hotkeys: {exc}", file=sys.stderr)

    app.aboutToQuit.connect(on_quit)
    sys.exit(app.exec())