"""
vision_capture.py
-----------------
Captures a region of the screen on a fixed interval and emits the result
as a base64-encoded PNG string via a PyQt6 signal.

Default capture region: right half of a 1920×1080 primary monitor.
Override at instantiation time via the `region` parameter.
"""

import base64
import io
import sys

from PyQt6.QtCore import QThread, pyqtSignal

try:
    import mss
    import mss.tools
except ImportError:
    mss = None
    print("vision_capture: 'mss' not installed — screen capture disabled.", file=sys.stderr)

try:
    from PIL import Image
except ImportError:
    Image = None
    print("vision_capture: 'Pillow' not installed — screen capture disabled.", file=sys.stderr)

# Default region: right half of a 1920×1080 display.
_DEFAULT_REGION = {"top": 0, "left": 960, "width": 960, "height": 1080}

# How often (seconds) a new screenshot is taken while active.
_CAPTURE_INTERVAL_MS = 4000


class VisionThread(QThread):
    """
    Background thread that takes a screenshot every `interval_ms` milliseconds
    and emits it as a base64-encoded PNG string.

    Signals
    -------
    image_captured(str)
        Emitted with the base64 PNG string of the captured region.
        Only emitted when `is_active` is True.
    """

    image_captured = pyqtSignal(str)

    def __init__(self, region: dict | None = None, interval_ms: int = _CAPTURE_INTERVAL_MS, parent=None):
        super().__init__(parent)
        self.region = region or _DEFAULT_REGION
        self.interval_ms = interval_ms
        # Master switch — flip externally to enable/disable capture without
        # stopping the thread entirely.
        self.is_active: bool = False
        self._stop_requested: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self):
        """Request the thread loop to exit at the next iteration."""
        self._stop_requested = True
        self.is_active = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _capture_base64(self) -> str | None:
        """Take one screenshot of `self.region` and return it as base64."""
        if mss is None or Image is None:
            print("vision_capture: dependencies missing, skipping capture.", file=sys.stderr)
            return None

        try:
            with mss.mss() as sct:
                raw = sct.grab(self.region)
                # Convert the mss ScreenShot object → PIL Image → PNG bytes
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as exc:
            print(f"vision_capture: capture error: {exc}", file=sys.stderr)
            return None

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self):
        """Main loop — runs on the worker thread."""
        while not self._stop_requested:
            if self.is_active:
                b64 = self._capture_base64()
                if b64:
                    self.image_captured.emit(b64)

            # Use QThread.msleep so Qt can process events between captures.
            self.msleep(self.interval_ms)
