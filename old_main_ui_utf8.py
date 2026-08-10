import sys
import html
import time
import asyncio
import threading
import ctypes
import ctypes.wintypes
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QTextEdit, QFrame)
from PyQt6.QtGui import QTextCursor, QShortcut, QKeySequence
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt6.QtCore import Qt
import config
from vision_capture import VisionThread
import llm_client


class WorkerSignals(QObject):
    interim_transcript = pyqtSignal(str)
    final_transcript = pyqtSignal(str)
    llm_token = pyqtSignal(str)
    llm_start = pyqtSignal()
    llm_end = pyqtSignal()
    status_update = pyqtSignal(str)


signals = WorkerSignals()


class HotkeySignals(QObject):
    toggle_pause = pyqtSignal()
    clear_ui = pyqtSignal()
    open_settings = pyqtSignal()
    toggle_vision = pyqtSignal()


hotkey_signals = HotkeySignals()
pause_event = threading.Event()
pause_event.set()

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API Configuration")
        self.setFixedSize(400, 260)
        
        layout = QVBoxLayout(self)
        
        self.dg_label = QLabel("Deepgram API Key:")
        self.dg_input = QLineEdit()
        self.dg_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.groq_label = QLabel("Groq API Key:")
        self.groq_input = QLineEdit()
        self.groq_input.setEchoMode(QLineEdit.EchoMode.Password)

        self.gemini_label = QLabel("Gemini API Key (Vision ΓÇö F7):")
        self.gemini_input = QLineEdit()
        self.gemini_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.save_btn = QPushButton("Save & Restart")
        self.save_btn.clicked.connect(self.save_keys)
        
        layout.addWidget(self.dg_label)
        layout.addWidget(self.dg_input)
        layout.addWidget(self.groq_label)
        layout.addWidget(self.groq_input)
        layout.addWidget(self.gemini_label)
        layout.addWidget(self.gemini_input)
        layout.addWidget(self.save_btn)
        
        self.load_existing()

    def load_existing(self):
        keys = config.load_config()
        self.dg_input.setText(keys.get("DEEPGRAM_API_KEY", ""))
        self.groq_input.setText(keys.get("GROQ_API_KEY", ""))
        self.gemini_input.setText(keys.get("GEMINI_API_KEY", ""))

    def save_keys(self):
        keys = {
            "DEEPGRAM_API_KEY": self.dg_input.text().strip(),
            "GROQ_API_KEY": self.groq_input.text().strip(),
            "GEMINI_API_KEY": self.gemini_input.text().strip(),
        }
        config.save_config(keys)
        self.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interview Assistant")
        self.setMinimumSize(380, 500)
        self.resize(450, 800)
        self.setMaximumWidth(450)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowOpacity(0.8)
        self.setStyleSheet("background-color: #0D0D0D;")
        self._final_sentences = []
        self._current_interim = ""
        self._is_streaming = False
        self._cursor_visible = False
        self._cursor_char_shown = False
        self.cursor_timer = QTimer(self)
        self.cursor_timer.timeout.connect(self._on_cursor_toggle)
        self.is_paused = False
        self.vision_active = False
        self._init_ui()
        self._connect_signals()
        self.current_opacity = 0.8
        QShortcut(QKeySequence("Ctrl+="), self).activated.connect(self._increase_opacity)
        QShortcut(QKeySequence("Ctrl+-"), self).activated.connect(self._decrease_opacity)
        hotkey_signals.toggle_pause.connect(self._on_toggle_pause)
        hotkey_signals.clear_ui.connect(self._on_clear_ui)
        hotkey_signals.open_settings.connect(self.show_settings)
        hotkey_signals.toggle_vision.connect(self.toggle_vision_mode)
        # --- Vision thread (starts paused; activated via F7) ---
        self._vision_thread = VisionThread()
        self._vision_thread.image_captured.connect(self.handle_new_image)
        self._vision_thread.start()   # thread is alive but is_active=False
        self._apply_capture_exclusion()

    def _apply_capture_exclusion(self):
        WDA_EXCLUDEFROMCAPTURE = 0x00000011
        try:
            hwnd = ctypes.c_void_p(int(self.winId()))
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        except Exception:
            pass

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        ai_section = QWidget()
        ai_layout = QVBoxLayout(ai_section)
        ai_layout.setContentsMargins(12, 12, 12, 6)
        ai_layout.setSpacing(6)
        ai_header = QLabel("AI ASSISTANT")
        ai_header.setStyleSheet("""
            QLabel {
                color: #888888;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
            }
        """)
        self.ai_edit = QTextEdit()
        self.ai_edit.setReadOnly(True)
        self.ai_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.ai_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.ai_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.ai_edit.setStyleSheet("""
            QTextEdit {
                background-color: #111111;
                color: #00FF88;
                border: none;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 14px;
                padding: 8px;
            }
        """)
        ai_layout.addWidget(ai_header)
        ai_layout.addWidget(self.ai_edit)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Plain)
        divider.setFixedHeight(1)
        divider.setStyleSheet("background-color: #1A1A1A; border: none;")

        transcript_section = QWidget()
        transcript_layout = QVBoxLayout(transcript_section)
        transcript_layout.setContentsMargins(12, 6, 12, 12)
        transcript_layout.setSpacing(6)
        transcript_header = QLabel("LIVE TRANSCRIPT")
        transcript_header.setStyleSheet("""
            QLabel {
                color: #888888;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
            }
        """)
        self.transcript_edit = QTextEdit()
        self.transcript_edit.setReadOnly(True)
        self.transcript_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.transcript_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.transcript_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.transcript_edit.setStyleSheet("""
            QTextEdit {
                background-color: #0A0A0A;
                color: #666666;
                border: none;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)
        transcript_layout.addWidget(transcript_header)
        transcript_layout.addWidget(self.transcript_edit)

        self.dot_indicator = QLabel()
        self.dot_indicator.setFixedSize(8, 8)
        self.dot_indicator.setStyleSheet("background-color: #444444; border-radius: 4px;")
        self.status_label = QLabel("Groq ┬╖ Llama 3.3 70B")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #555555;
                font-family: sans-serif;
                font-size: 10px;
            }
        """)
        # Vision indicator ΓÇö hidden by default, shown when vision is active.
        self.vision_label = QLabel("≡ƒæü∩╕Å VISION ON")
        self.vision_label.setStyleSheet("""
            QLabel {
                color: #00FF88;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                font-weight: bold;
                padding: 0 6px;
            }
        """)
        self.vision_label.setVisible(False)

        status_bar = QWidget()
        status_bar.setFixedHeight(28)
        status_bar.setStyleSheet("background-color: #080808; border: none;")
        sb_layout = QHBoxLayout(status_bar)
        sb_layout.setContentsMargins(12, 0, 12, 0)
        sb_layout.addWidget(self.dot_indicator, 0, Qt.AlignmentFlag.AlignVCenter)
        sb_layout.addWidget(self.vision_label, 0, Qt.AlignmentFlag.AlignVCenter)
        sb_layout.addStretch()
        sb_layout.addWidget(self.status_label, 0, Qt.AlignmentFlag.AlignVCenter)

        root.addWidget(ai_section, 3)
        root.addWidget(divider)
        root.addWidget(transcript_section, 1)
        root.addWidget(status_bar)

    def _connect_signals(self):
        signals.interim_transcript.connect(self._on_interim_transcript)
        signals.final_transcript.connect(self._on_final_transcript)
        signals.llm_token.connect(self._on_llm_token)
        signals.llm_start.connect(self._on_llm_start)
        signals.llm_end.connect(self._on_llm_end)
        signals.status_update.connect(self._on_status_update)

    def _render_transcript(self):
        parts = []
        for s in self._final_sentences:
            escaped = html.escape(s)
            parts.append(f'<div style="color:#E0E0E0;white-space:pre-wrap;">{escaped}</div>')
        if self._current_interim:
            escaped = html.escape(self._current_interim)
            parts.append(f'<div style="color:#666666;white-space:pre-wrap;">{escaped}</div>')
        self.transcript_edit.setHtml("".join(parts))
        sb = self.transcript_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _ai_doc_cursor_at_end(self):
        cursor = QTextCursor(self.ai_edit.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        return cursor

    def _ai_remove_cursor_char(self):
        if not self._cursor_char_shown:
            return
        cursor = self._ai_doc_cursor_at_end()
        block_text = cursor.block().text()
        if block_text.endswith("Γûï"):
            cursor.deletePreviousChar()
            self._cursor_char_shown = False

    def _ai_insert_cursor_char(self):
        if self._cursor_char_shown:
            return
        cursor = self._ai_doc_cursor_at_end()
        cursor.insertText("Γûï")
        self._cursor_char_shown = True

    def _on_interim_transcript(self, text):
        self._current_interim = text
        self._render_transcript()

    def _on_final_transcript(self, text):
        self._final_sentences.append(text)
        self._current_interim = ""
        self._render_transcript()

    def _on_llm_start(self):
        self._is_streaming = True
        self._cursor_visible = True
        self._cursor_char_shown = False
        # Append a separator between responses instead of clearing, so the
        # user can scroll up and review previous answers during the interview.
        cursor = self._ai_doc_cursor_at_end()
        if not self.ai_edit.document().isEmpty():
            cursor.insertText("\n\n" + "ΓöÇ" * 32 + "\n\n")
        cursor = self._ai_doc_cursor_at_end()
        cursor.insertText("Γûï")
        self._cursor_char_shown = True
        # Scroll to bottom so the new (streaming) answer is visible.
        sb = self.ai_edit.verticalScrollBar()
        sb.setValue(sb.maximum())
        self.cursor_timer.start(600)

    def _on_llm_token(self, token):
        self._ai_remove_cursor_char()
        cursor = self._ai_doc_cursor_at_end()
        cursor.insertText(token)
        if self._is_streaming:
            cursor.insertText("Γûï")
            self._cursor_char_shown = True
        sb = self.ai_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_llm_end(self):
        self._is_streaming = False
        self.cursor_timer.stop()
        self._cursor_visible = False
        self._ai_remove_cursor_char()
        sb = self.ai_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_cursor_toggle(self):
        self._cursor_visible = not self._cursor_visible
        if self._cursor_visible:
            self._ai_insert_cursor_char()
        else:
            self._ai_remove_cursor_char()

    def _on_status_update(self, status):
        colors = {
            "connected": "#00FF88",
            "disconnected": "#FF4444",
            "initializing": "#444444",
        }
        color = colors.get(status, "#444444")
        self.dot_indicator.setStyleSheet(f"background-color: {color}; border-radius: 4px;")

    def _increase_opacity(self):
        self.current_opacity = min(1.0, round(self.current_opacity + 0.1, 1))
        self.setWindowOpacity(self.current_opacity)

    def _decrease_opacity(self):
        self.current_opacity = max(0.1, round(self.current_opacity - 0.1, 1))
        self.setWindowOpacity(self.current_opacity)

    def show_settings(self):
        """Open the API key settings dialog (triggered by F8 hotkey)."""
        dlg = SettingsDialog(self)
        if dlg.exec():
            # Keys were saved ΓÇö notify the user so they know to restart.
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Keys Saved",
                "API keys updated.\nPlease restart the app for the new keys to take effect.",
            )

    # ------------------------------------------------------------------
    # Vision mode
    # ------------------------------------------------------------------

    def toggle_vision_mode(self):
        """F7 handler ΓÇö flip vision on/off and update UI accordingly."""
        self.vision_active = not self.vision_active
        self._vision_thread.is_active = self.vision_active

        if self.vision_active:
            # Green border on the AI text box as a subtle live indicator.
            self.ai_edit.setStyleSheet("""
                QTextEdit {
                    background-color: #111111;
                    color: #00FF88;
                    border: 1px solid #00FF88;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 14px;
                    padding: 8px;
                }
            """)
            self.vision_label.setVisible(True)
        else:
            self.ai_edit.setStyleSheet("""
                QTextEdit {
                    background-color: #111111;
                    color: #00FF88;
                    border: none;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 14px;
                    padding: 8px;
                }
            """)
            self.vision_label.setVisible(False)

    def handle_new_image(self, base64_image: str):
        """
        Called (on the main thread via Qt signal) every time VisionThread
        emits a new screenshot.  Schedules an async Gemini call on the
        existing event loop that drives the audio/STT/LLM pipeline.
        """
        # Collect the latest spoken text as context for Gemini.
        audio_ctx = " ".join(self._final_sentences[-2:]) if self._final_sentences else ""

        # Retrieve the shared asyncio loop created in main.py.
        # We import lazily to avoid a circular dependency at module load time.
        try:
            import main as _main_module
            loop = _main_module.loop
        except Exception:
            loop = None

        if loop is None or not loop.is_running():
            print("main_ui: no running event loop ΓÇö vision response skipped.", file=sys.stderr)
            return

        asyncio.run_coroutine_threadsafe(
            llm_client.generate_vision_response(base64_image, audio_ctx, signals),
            loop,
        )

    # ------------------------------------------------------------------
    # Pause / clear / settings
    # ------------------------------------------------------------------

    def _on_toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            pause_event.clear()
            self.setWindowTitle("Interview Assistant  ΓÅ╕ [PAUSED]")
            self.dot_indicator.setStyleSheet("background-color: #FF8800; border-radius: 4px;")
        else:
            pause_event.set()
            self.setWindowTitle("Interview Assistant")
            self.dot_indicator.setStyleSheet("background-color: #00FF88; border-radius: 4px;")

    def _on_clear_ui(self):
        self._final_sentences = []
        self._current_interim = ""
        self.transcript_edit.clear()
        self._is_streaming = False
        self._cursor_char_shown = False
        self.cursor_timer.stop()
        self.ai_edit.clear()


if __name__ == "__main__":
    import threading

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    def _demo():
        time.sleep(1.0)
        signals.status_update.emit("initializing")
        time.sleep(1.0)
        signals.status_update.emit("connected")
        sentences = [
            "Can you explain the difference between a process and a thread?",
            "How does the GIL affect Python concurrency?",
        ]
        for sentence in sentences:
            words = sentence.split()
            interim = ""
            for word in words:
                interim += word + " "
                signals.interim_transcript.emit(interim.strip())
                time.sleep(0.12)
            signals.final_transcript.emit(sentence)
            time.sleep(0.4)
            signals.llm_start.emit()
            time.sleep(0.2)
            response = (
                "ΓÇó A process has its own memory space; threads share memory within a process.\n"
                "ΓÇó Threads are lighter weight but require synchronisation primitives.\n"
                "ΓÇó Use multiprocessing for CPU-bound work, threads for I/O-bound work."
            )
            for token in response.split(" "):
                signals.llm_token.emit(token + " ")
                time.sleep(0.08)
            signals.llm_end.emit()
            time.sleep(1.2)

    t = threading.Thread(target=_demo, daemon=True)
    t.start()
    sys.exit(app.exec())
