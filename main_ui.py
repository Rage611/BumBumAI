import sys
import html
import time
import ctypes
import ctypes.wintypes
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QTextEdit, QSplitter, QFrame)
from PyQt6.QtGui import QTextCursor


class WorkerSignals(QObject):
    interim_transcript = pyqtSignal(str)
    final_transcript = pyqtSignal(str)
    llm_token = pyqtSignal(str)
    llm_start = pyqtSignal()
    llm_end = pyqtSignal()
    status_update = pyqtSignal(str)


signals = WorkerSignals()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interview Assistant")
        self.setMinimumSize(900, 600)
        self.resize(1200, 700)
        self.setStyleSheet("background-color: #0D0D0D;")
        self._final_sentences = []
        self._current_interim = ""
        self._is_streaming = False
        self._cursor_visible = False
        self._cursor_char_shown = False
        self.cursor_timer = QTimer(self)
        self.cursor_timer.timeout.connect(self._on_cursor_toggle)
        self._init_ui()
        self._connect_signals()
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

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #1A1A1A;
                width: 1px;
            }
        """)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)
        left_header = QLabel("LIVE TRANSCRIPT")
        left_header.setStyleSheet("""
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
                background-color: #111111;
                color: #E0E0E0;
                border: none;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                padding: 8px;
            }
        """)
        left_layout.addWidget(left_header)
        left_layout.addWidget(self.transcript_edit)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Plain)
        separator.setFixedHeight(2)
        separator.setStyleSheet("background-color: #1A1A1A; border: none;")
        right_header = QLabel("AI ASSISTANT")
        right_header.setStyleSheet("""
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
        right_layout.addWidget(separator)
        right_layout.addWidget(right_header)
        right_layout.addWidget(self.ai_edit)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 600])

        self.dot_indicator = QLabel()
        self.dot_indicator.setFixedSize(8, 8)
        self.dot_indicator.setStyleSheet("background-color: #444444; border-radius: 4px;")
        self.status_label = QLabel("Groq · Llama 3.3 70B")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #555555;
                font-family: sans-serif;
                font-size: 10px;
            }
        """)
        status_bar = QWidget()
        status_bar.setFixedHeight(28)
        status_bar.setStyleSheet("background-color: #080808; border: none;")
        sb_layout = QHBoxLayout(status_bar)
        sb_layout.setContentsMargins(12, 0, 12, 0)
        sb_layout.addWidget(self.dot_indicator, 0, Qt.AlignmentFlag.AlignVCenter)
        sb_layout.addStretch()
        sb_layout.addWidget(self.status_label, 0, Qt.AlignmentFlag.AlignVCenter)

        root.addWidget(splitter, 1)
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
        if block_text.endswith("▋"):
            cursor.deletePreviousChar()
            self._cursor_char_shown = False

    def _ai_insert_cursor_char(self):
        if self._cursor_char_shown:
            return
        cursor = self._ai_doc_cursor_at_end()
        cursor.insertText("▋")
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
        self.ai_edit.clear()
        cursor = self._ai_doc_cursor_at_end()
        cursor.insertText("▋")
        self._cursor_char_shown = True
        self.cursor_timer.start(600)

    def _on_llm_token(self, token):
        self._ai_remove_cursor_char()
        cursor = self._ai_doc_cursor_at_end()
        cursor.insertText(token)
        if self._is_streaming:
            cursor.insertText("▋")
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
                "• A process has its own memory space; threads share memory within a process.\n"
                "• Threads are lighter weight but require synchronisation primitives.\n"
                "• Use multiprocessing for CPU-bound work, threads for I/O-bound work."
            )
            for token in response.split(" "):
                signals.llm_token.emit(token + " ")
                time.sleep(0.08)
            signals.llm_end.emit()
            time.sleep(1.2)

    t = threading.Thread(target=_demo, daemon=True)
    t.start()
    sys.exit(app.exec())
