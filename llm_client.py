import asyncio
import sys
from collections import deque
from pathlib import Path

from groq import AsyncGroq

try:
    _resume_path = Path(__file__).parent / "resume.md"
    RESUME_CONTEXT = _resume_path.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    RESUME_CONTEXT = "No resume provided."
    print("llm_client: resume.md not found — context injection disabled.", file=sys.stderr)
except Exception as _exc:
    RESUME_CONTEXT = "No resume provided."
    print(f"llm_client: failed to read resume.md: {_exc}", file=sys.stderr)

SYSTEM_PROMPT = f"""You are ME, sitting in a live technical interview. You are acting as my direct voice.
Your goal is to generate the exact words I should say out loud.

STRICT CONSTRAINTS & BEHAVIOR:
1. EXTREME BREVITY: For basic technical questions (e.g., "What is HTML vs JS?"), give a maximum 1 to 2 sentence answer. Get straight to the technical point. NO fluff, NO analogies (e.g., "think of it like a house"), and NO concluding summary sentences.
2. RESUME RESTRICTION: DO NOT mention my projects, resume, or past experience UNLESS the interviewer explicitly asks about my background, a specific project, or asks a behavioral question. For general technical questions, give general answers.
3. TONE: Casual, spoken, conversational. Use short, punchy sentences. Sound like a confident human engineer, not an AI essay. 
4. FORMATTING: Use absolutely NO markdown bolding (**). Keep text plain and use simple line breaks.

CRITICAL CODING RULES:
• If asked for code (like a LeetCode problem), ALWAYS provide the solution in C++.
• You MUST remove all comments from the generated code.

MY BACKGROUND & PROJECTS (USE ONLY IF EXPLICITLY ASKED ABOUT MY EXPERIENCE):
--- MY RESUME ---
{RESUME_CONTEXT}
--- END RESUME ---"""

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
    finally:
        close = getattr(client, "close", None) or getattr(client, "aclose", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result