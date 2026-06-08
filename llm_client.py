import asyncio
import sys
from collections import deque

from groq import AsyncGroq

SYSTEM_PROMPT = (
    "You are an expert Technical Interviewer and Software Engineering Mentor. "
    "Your goal is to provide high-quality, actionable, and educational explanations.\n"
    "Guidelines:\n"
    "• Be Comprehensive: If asked a technical concept, explain the 'how' and 'why' clearly.\n"
    "• Provide Code: If asked to write code, provide the full solution inside a code block.\n"
    "• Formatting: Use Markdown (bolding, code blocks, lists) to ensure your output is structured and easy to scan.\n"
    "• Tone: Professional, encouraging, and clear. Avoid filler phrases like 'Great question'.\n"
    "• Balance: Be as concise as possible while remaining fully informative. Do not cut off useful details."
)

_current_task = None
_history = deque(maxlen=6)


async def _generate(transcript, client, signals):
    signals.llm_start.emit()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(_history)
    messages.append({"role": "user", "content": transcript})
    full_response = ""
    try:
        stream = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            stream=True,
            max_tokens=300,
            temperature=0.3,
            top_p=0.9,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content
            if token:
                full_response += token
                signals.llm_token.emit(token)
        _history.append({"role": "user", "content": transcript})
        _history.append({"role": "assistant", "content": full_response})
        signals.llm_end.emit()
    except asyncio.CancelledError:
        signals.llm_end.emit()
        raise
    except Exception as exc:
        print(f"llm_client: generation error: {exc}", file=sys.stderr)
        signals.llm_end.emit()


async def run_llm(llm_queue, signals, api_key):
    global _current_task
    client = AsyncGroq(api_key=api_key)
    try:
        while True:
            transcript = await llm_queue.get()
            while not llm_queue.empty():
                transcript = llm_queue.get_nowait()

            if _current_task is not None and not _current_task.done():
                _current_task.cancel()
                try:
                    await _current_task
                except asyncio.CancelledError:
                    pass

            _current_task = asyncio.create_task(_generate(transcript, client, signals))

    except asyncio.CancelledError:
        if _current_task is not None and not _current_task.done():
            _current_task.cancel()
            try:
                await _current_task
            except asyncio.CancelledError:
                pass
        raise
