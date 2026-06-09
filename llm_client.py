import asyncio
import sys
from collections import deque
from pathlib import Path

import groq
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
1. EXTREME BREVITY: For basic technical questions (e.g., "What is HTML vs JS?"), give a maximum 5 to 7 sentence answer, and the answer should be simple, not too much technical, like a human. Get straight to the technical point. NO fluff, NO analogies, and NO concluding summary sentences.
2. THE "ELEVATOR PITCH" OVERRIDE: If the interviewer asks "tell me about yourself," "introduce yourself," or "walk me through your background," you MUST deliver a confident, conversational 4 to 5 sentence elevator pitch. Introduce yourself as a Computer Science undergrad at Bennett University and a freelance software developer based in New Delhi. Briefly mention your expertise in the MERN stack, your background in competitive programming with C++, and highlight your drive for building production-grade AI tools like Synex AI and Voicify. Sound human, passionate, and professional.
3. THE GUARD GRID PIVOT (CRITICAL): I have an unlisted recent project called "Guard Grid" (a Next.js/Supabase app for TMS Security Services). If the interviewer asks a general question about Next.js, Supabase, database security (RLS), multi-tenant architecture, OR a behavioral question like "tell me about a recent technical challenge," you MUST pivot and use Guard Grid as your real-world example. Use phrasing like, "Actually, I recently implemented this in a project I built called Guard Grid..."
4. RESUME TRIGGER & EXACT NAMES: Before answering, ALWAYS check if the interviewer's question contains words matching my projects (like "Synex", "TMS Security", "Voicify", "Guard Grid"). If they do, you MUST use the resume context to answer as me. CRITICAL: You MUST use the exact proper names of my projects and companies. Never generalize my work.
5. TONE: Casual, spoken, conversational. Use short, punchy sentences. Sound like a confident human engineer, not an AI essay. 
6. FORMATTING: Use absolutely NO markdown bolding (**). Keep text plain and use simple line breaks.

CRITICAL CODING RULES:
• If asked for code (like a LeetCode problem), ALWAYS provide the solution in C++.
• You MUST remove all comments from the generated code.

--- MY RESUME (Context for my background and specific projects) ---
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
        # Normal cancellation when a new question interrupts the current one.
        signals.llm_end.emit()
        raise

    except groq.RateLimitError as exc:
        # HTTP 429 — quota exhausted on this API key.
        print(f"llm_client: rate limit hit: {exc}", file=sys.stderr)
        _RATE_LIMIT_MSG = (
            "\n\n🚨 [SYSTEM ALERT]: GROQ API LIMIT REACHED. "
            "PLEASE PRESS F8 TO SWAP API KEYS. 🚨\n\n"
        )
        signals.llm_token.emit(_RATE_LIMIT_MSG)
        signals.llm_end.emit()

    except groq.APIStatusError as exc:
        # Other 4xx / 5xx responses (auth failure, server error, etc.).
        print(f"llm_client: API status error {exc.status_code}: {exc}", file=sys.stderr)
        _API_ERR_MSG = (
            f"\n\n🚨 [SYSTEM ALERT]: API ERROR {exc.status_code} — "
            "CHECK LOGS. 🚨\n\n"
        )
        signals.llm_token.emit(_API_ERR_MSG)
        signals.llm_end.emit()

    except Exception as exc:
        # Network drop, timeout, or any other unexpected failure.
        print(f"llm_client: generation error: {exc}", file=sys.stderr)
        _CONN_LOST_MSG = (
            "\n\n🚨 [SYSTEM ALERT]: CONNECTION LOST. 🚨\n\n"
        )
        signals.llm_token.emit(_CONN_LOST_MSG)
        signals.llm_end.emit()

async def run_llm(llm_queue, signals, api_key):
    global _current_task
    client = AsyncGroq(api_key=api_key)
    try:
        while True:
            transcript = await llm_queue.get()
            while not llm_queue.empty():
                transcript = llm_queue.get_nowait()

            # --- Noise / filler guard ---
            # Affirmative sounds ("hmm", "yeah", "okay") or mic bleed from the
            # interviewer while the user reads the answer out loud produce very
            # short STT strings.  Anything under 4 words is almost certainly
            # not a real question, so we drop it silently.
            if len(transcript.split()) < 4:
                print(
                    f"llm_client: transcript too short ({len(transcript.split())} word(s)) — skipped.",
                    file=sys.stderr,
                )
                llm_queue.task_done() if hasattr(llm_queue, "task_done") else None
                continue

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