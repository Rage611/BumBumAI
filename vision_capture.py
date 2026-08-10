import base64
import collections
import sys

from PyQt6.QtCore import QTimer, QObject, pyqtSignal, QBuffer, QIODevice
from PyQt6.QtWidgets import QApplication

BUFFER_INTERVAL_MS = 1500
BUFFER_SIZE = 3

class VisionCapture(QObject):
    image_captured = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_active: bool = False
        self._buffer: collections.deque[str] = collections.deque(maxlen=BUFFER_SIZE)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(BUFFER_INTERVAL_MS)

    def stop(self):
        self._timer.stop()
        self.is_active = False

    def get_latest_frame(self) -> str | None:
        return self._buffer[-1] if self._buffer else None

    def dispatch_latest(self):
        frame = self.get_latest_frame()
        if frame is None:
            frame = self._capture_base64()
        if frame:
            self.image_captured.emit(frame)
        else:
            print("vision_capture: no frame available to dispatch.", file=sys.stderr)

    def _on_tick(self):
        b64 = self._capture_base64()
        if b64:
            self._buffer.append(b64)

    def _capture_base64(self) -> str | None:
        try:
            app = QApplication.instance()
            if app is None:
                return None
            screen = app.primaryScreen()
            if screen is None:
                return None
            pixmap = screen.grabWindow(0)
            if pixmap.isNull():
                return None
            if pixmap.width() > 1920:
                pixmap = pixmap.scaledToWidth(1920)
            buf = QBuffer()
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            pixmap.save(buf, "JPEG", 85)
            jpeg_bytes = buf.data().data()
            buf.close()
            return base64.b64encode(jpeg_bytes).decode("utf-8")
        except Exception as exc:
            print(f"vision_capture: capture error: {exc}", file=sys.stderr)
            return None
