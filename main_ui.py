import sys
import html
import time
import asyncio
import threading
import ctypes
import ctypes.wintypes
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QTextEdit, QFrame,
                             QDialog, QLineEdit, QPushButton, QComboBox, QScrollArea,
                             QGroupBox, QMessageBox)
from PyQt6.QtGui import QTextCursor, QShortcut, QKeySequence
from PyQt6.QtCore import Qt
import config
from vision_capture import VisionCapture
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

# Available Groq models
MODEL_OPTIONS = [
    ("— None (skip this slot) —",    None,    None),
    ("Llama 3.3 70B  (Groq)",        "groq",  "llama-3.3-70b-versatile"),
    ("Llama 3.1 8B — Fast  (Groq)",  "groq",  "llama-3.1-8b-instant"),
]

DEFAULT_CHAIN = [
    {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    {"provider": "groq", "model": "llama-3.1-8b-instant"},
]

MAX_KEYS = 5  # Max API keys per provider


# ---------------------------------------------------------------------------
# Settings Dialog — multi-key support, no restart required
# ---------------------------------------------------------------------------
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parakeet — Settings")
        self.setMinimumWidth(480)
        self.setStyleSheet("""
            QDialog { background-color: #111; color: #ddd; }
            QLabel { color: #aaa; font-size: 12px; }
            QLabel#section_label { color: #00FF88; font-weight: bold; font-size: 13px; margin-top: 8px; }
            QLineEdit {
                background: #1a1a1a; color: #eee; border: 1px solid #333;
                border-radius: 4px; padding: 5px 8px; font-size: 12px;
            }
            QLineEdit:focus { border: 1px solid #00FF88; }
            QPushButton {
                background: #00FF88; color: #000; border: none;
                border-radius: 4px; padding: 8px 16px; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover { background: #00cc70; }
            QComboBox {
                background: #1a1a1a; color: #eee; border: 1px solid #333;
                border-radius: 4px; padding: 4px 8px;
            }
            QGroupBox {
                border: 1px solid #2a2a2a; border-radius: 6px;
                margin-top: 8px; padding: 8px;
                color: #888; font-size: 11px;
            }
            QGroupBox::title { color: #555; }
        """)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: #111; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(6)
        layout.setContentsMargins(16, 16, 16, 16)

        # --- Groq API Keys ---
        groq_label = QLabel("Groq API Keys  (rotates on rate limit — add up to 5 free keys)")
        groq_label.setObjectName("section_label")
        layout.addWidget(groq_label)

        self.groq_inputs: list[QLineEdit] = []
        groq_box = QGroupBox("Keys rotate in order: Key 1 → Key 2 → ... → next model")
        groq_inner = QVBoxLayout(groq_box)
        groq_inner.setSpacing(4)
        for i in range(MAX_KEYS):
            row = QHBoxLayout()
            lbl = QLabel(f"Key {i+1}:")
            lbl.setFixedWidth(50)
            inp = QLineEdit()
            inp.setEchoMode(QLineEdit.EchoMode.Password)
            inp.setPlaceholderText(f"gsk_... (slot {i+1})")
            self.groq_inputs.append(inp)
            row.addWidget(lbl)
            row.addWidget(inp)
            groq_inner.addLayout(row)
        layout.addWidget(groq_box)

        # --- Google AI Studio Keys ---
        google_label = QLabel("Google AI Studio Keys  (for Vision — Gemini 2.5 Flash, free)")
        google_label.setObjectName("section_label")
        layout.addWidget(google_label)

        self.google_inputs: list[QLineEdit] = []
        google_box = QGroupBox("Keys rotate in order: Key 1 → Key 2 → ... on quota")
        google_inner = QVBoxLayout(google_box)
        google_inner.setSpacing(4)
        for i in range(MAX_KEYS):
            row = QHBoxLayout()
            lbl = QLabel(f"Key {i+1}:")
            lbl.setFixedWidth(50)
            inp = QLineEdit()
            inp.setEchoMode(QLineEdit.EchoMode.Password)
            inp.setPlaceholderText(f"AIza... (slot {i+1})")
            self.google_inputs.append(inp)
            row.addWidget(lbl)
            row.addWidget(inp)
            google_inner.addLayout(row)
        layout.addWidget(google_box)

        # --- Whisper Model ---
        model_label = QLabel("Local Whisper Model")
        model_label.setObjectName("section_label")
        layout.addWidget(model_label)

        self.whisper_combo = QComboBox()
        self.whisper_combo.addItem("base.en  (faster, ~150ms)")
        self.whisper_combo.addItem("small.en  (more accurate, ~400ms)")
        layout.addWidget(self.whisper_combo)

        # --- LLM Failover Order ---
        chain_label = QLabel("LLM Failover Order  (after all keys for a model are exhausted)")
        chain_label.setObjectName("section_label")
        layout.addWidget(chain_label)

        self.priority_combos: list[QComboBox] = []
        ranks = ["1st Preference", "2nd Preference"]
        for rank in ranks:
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(rank)
            lbl.setFixedWidth(120)
            combo = QComboBox()
            for label, _, _ in MODEL_OPTIONS:
                combo.addItem(label)
            row.addWidget(lbl)
            row.addWidget(combo)
            layout.addWidget(row_widget)
            self.priority_combos.append(combo)

        # --- Save button ---
        layout.addSpacing(8)
        self.save_btn = QPushButton("Save  (takes effect immediately — no restart needed)")
        self.save_btn.clicked.connect(self.save_settings)
        layout.addWidget(self.save_btn)

        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self.load_existing()

    # ------------------------------------------------------------------
    def _model_index(self, provider, model_id) -> int:
        for i, (_, p, m) in enumerate(MODEL_OPTIONS):
            if p == provider and m == model_id:
                return i
        return 0

    def load_existing(self):
        cfg = config.load_config()

        groq_keys = cfg.get("GROQ_API_KEYS", [""] * MAX_KEYS)
        for i, inp in enumerate(self.groq_inputs):
            inp.setText(groq_keys[i] if i < len(groq_keys) else "")

        google_keys = cfg.get("GOOGLE_API_KEYS", [""] * MAX_KEYS)
        for i, inp in enumerate(self.google_inputs):
            inp.setText(google_keys[i] if i < len(google_keys) else "")

        whisper = cfg.get("WHISPER_MODEL", "base.en")
        self.whisper_combo.setCurrentIndex(1 if "small" in whisper else 0)

        chain = cfg.get("LLM_CHAIN", DEFAULT_CHAIN)
        chain = list(chain) + [{"provider": None, "model": None}] * len(self.priority_combos)
        for i, combo in enumerate(self.priority_combos):
            entry = chain[i]
            if isinstance(entry, dict):
                p, m = entry.get("provider"), entry.get("model")
            else:
                p, m = entry[0], entry[1]
            combo.setCurrentIndex(self._model_index(p, m))

    def save_settings(self):
        groq_keys = [inp.text().strip() for inp in self.groq_inputs]
        google_keys = [inp.text().strip() for inp in self.google_inputs]

        if not any(groq_keys):
            QMessageBox.warning(self, "Missing Key", "At least one Groq API key is required.")
            return

        # Build chain from dropdowns
        chain = []
        seen = set()
        for combo in self.priority_combos:
            idx = combo.currentIndex()
            _, provider, model_id = MODEL_OPTIONS[idx]
            if provider is None:
                continue
            key = (provider, model_id)
            if key in seen:
                QMessageBox.warning(self, "Duplicate", f"Each model can only appear once.")
                return
            seen.add(key)
            chain.append({"provider": provider, "model": model_id})

        if not chain:
            chain = DEFAULT_CHAIN

        whisper_model = "small.en" if self.whisper_combo.currentIndex() == 1 else "base.en"

        cfg = config.load_config()
        cfg["GROQ_API_KEYS"]   = groq_keys
        cfg["GOOGLE_API_KEYS"] = google_keys
        cfg["WHISPER_MODEL"]   = whisper_model
        cfg["LLM_CHAIN"]       = chain
        cfg["LLM_MODEL"]       = chain[0]["model"]
        cfg["LLM_PROVIDER"]    = chain[0]["provider"]
        config.save_config(cfg)

        # Apply new keys immediately without restart
        llm_client.configure_vision(google_keys)

        QMessageBox.information(self, "Saved", "Settings saved and applied!\nGroq key rotation and vision keys are now active.")
        self.accept()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, loop=None):
        super().__init__()
        self._loop = loop
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

        # Token buffer for smooth UI rendering
        self._token_buffer: list[str] = []
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._flush_token_buffer)
        self._render_timer.start(33)  # ~30 FPS

        self._init_ui()
        self._connect_signals()
        self.current_opacity = 0.8
        QShortcut(QKeySequence("Ctrl+="), self).activated.connect(self._increase_opacity)
        QShortcut(QKeySequence("Ctrl+-"), self).activated.connect(self._decrease_opacity)
        hotkey_signals.toggle_pause.connect(self._on_toggle_pause)
        hotkey_signals.clear_ui.connect(self._on_clear_ui)
        hotkey_signals.open_settings.connect(self.show_settings)
        hotkey_signals.toggle_vision.connect(self.toggle_vision_mode)
        self._vision_capture = VisionCapture(parent=self)
        self._vision_capture.image_captured.connect(self.handle_new_image)
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
        self.status_label = QLabel(self._model_label())
        self.status_label.setStyleSheet("""
            QLabel {
                color: #555555;
                font-family: sans-serif;
                font-size: 10px;
            }
        """)
        self.vision_label = QLabel("📷 VISION ON")
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
        if cursor.block().text().endswith("█"):
            cursor.deletePreviousChar()
            self._cursor_char_shown = False

    def _ai_insert_cursor_char(self):
        if self._cursor_char_shown:
            return
        cursor = self._ai_doc_cursor_at_end()
        cursor.insertText("█")
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
        self._token_buffer.clear()
        cursor = self._ai_doc_cursor_at_end()
        if not self.ai_edit.document().isEmpty():
            cursor.insertText("\n\n" + "─" * 32 + "\n\n")
        cursor = self._ai_doc_cursor_at_end()
        cursor.insertText("█")
        self._cursor_char_shown = True
        self.ai_edit.verticalScrollBar().setValue(self.ai_edit.verticalScrollBar().maximum())
        self.cursor_timer.start(600)

    def _on_llm_token(self, token):
        """Buffer tokens and let the render timer flush them at ~30fps."""
        self._token_buffer.append(token)

    def _flush_token_buffer(self):
        """Called every 33ms — flushes buffered tokens to the UI in one shot."""
        if not self._token_buffer:
            return
        text = "".join(self._token_buffer)
        self._token_buffer.clear()
        self._ai_remove_cursor_char()
        cursor = self._ai_doc_cursor_at_end()
        cursor.insertText(text)
        if self._is_streaming:
            cursor.insertText("█")
            self._cursor_char_shown = True
        self.ai_edit.verticalScrollBar().setValue(self.ai_edit.verticalScrollBar().maximum())

    def _on_llm_end(self):
        self._is_streaming = False
        self.cursor_timer.stop()
        self._cursor_visible = False
        # Flush any remaining buffered tokens
        if self._token_buffer:
            self._flush_token_buffer()
        self._ai_remove_cursor_char()
        self.ai_edit.verticalScrollBar().setValue(self.ai_edit.verticalScrollBar().maximum())

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
        dlg = SettingsDialog(self)
        dlg.exec()
        # No restart needed — settings applied immediately in save_settings()

    def toggle_vision_mode(self):
        self.vision_active = not self.vision_active
        if self.vision_active:
            self._vision_capture.is_active = True
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
            self._vision_capture.dispatch_latest()
        else:
            self._vision_capture.is_active = False
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

    @staticmethod
    def _model_label() -> str:
        try:
            cfg = config.load_config()
            model = cfg.get("LLM_MODEL", "llama-3.3-70b-versatile")
            provider = cfg.get("LLM_PROVIDER", "groq").upper()
            return f"{provider} · {model}"
        except Exception:
            return "GROQ · Llama 3.3 70B"

    def handle_new_image(self, base64_image: str):
        audio_ctx = " ".join(self._final_sentences[-2:]) if self._final_sentences else ""
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(
            llm_client.generate_vision_response(base64_image, audio_ctx, signals),
            loop,
        )

    def _on_toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            pause_event.clear()
            self.setWindowTitle("Interview Assistant  ⏸ [PAUSED]")
            self.dot_indicator.setStyleSheet("background-color: #FF8800; border-radius: 4px;")
        else:
            pause_event.set()
            self.setWindowTitle("Interview Assistant")
            self.dot_indicator.setStyleSheet("background-color: #00FF88; border-radius: 4px;")

    def _on_clear_ui(self):
        self._final_sentences = []
        self._current_interim = ""
        self._token_buffer.clear()
        self.transcript_edit.clear()
        self._is_streaming = False
        self._cursor_char_shown = False
        self.cursor_timer.stop()
        self.ai_edit.clear()
