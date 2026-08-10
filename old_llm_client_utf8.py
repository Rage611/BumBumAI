import asyncio
import sys
from collections import deque
from pathlib import Path

import groq
from groq import AsyncGroq

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    _GENAI_AVAILABLE = False
    print("llm_client: 'google-generativeai' not installed ΓÇö vision disabled.", file=sys.stderr)

try:
    _resume_path = Path(__file__).parent / "resume.md"
    RESUME_CONTEXT = _resume_path.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    RESUME_CONTEXT = "No resume provided."
    print("llm_client: resume.md not found ΓÇö context injection disabled.", file=sys.stderr)
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
ΓÇó If asked for code (like a LeetCode problem), ALWAYS provide the solution in C++.
ΓÇó You MUST remove all comments from the generated code.

--- MY RESUME (Context for my background and specific projects) ---
{RESUME_CONTEXT}
--- END RESUME ---"""

# ---------------------------------------------------------------------------
# Gemini vision configuration
# ---------------------------------------------------------------------------

GEMINI_VISION_PROMPT = (
    "You are a stealth interview assistant. "
    "Read the provided image of a Google Meet chat or shared screen. "
    "Identify any coding questions or technical questions. "
    "Provide the direct answer. "
    "CRITICAL: If it is a coding question, output ONLY C++ code. "
    "Remove ALL comments."
)

# Populated lazily on first vision call (key injected via configure_gemini()).
_gemini_model = None


def configure_gemini(api_key: str) -> None:
    """Initialise the Gemini client. Call once at startup with the key."""
    global _gemini_model, _GENAI_AVAILABLE
    if not _GENAI_AVAILABLE:
        print("llm_client: google-generativeai unavailable, skipping Gemini init.", file=sys.stderr)
        return
    try:
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel(
            model_name="gemini-3.5-flash",
            system_instruction=GEMINI_VISION_PROMPT,
        )
        print("llm_client: Gemini 3.5 Flash initialised.", file=sys.stderr)
    except Exception as exc:
        print(f"llm_client: Gemini init failed: {exc}", file=sys.stderr)


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
        # HTTP 429 ΓÇö quota exhausted on this API key.
        print(f"llm_client: rate limit hit: {exc}", file=sys.stderr)
        _RATE_LIMIT_MSG = (
            "\n\n≡ƒÜ¿ [SYSTEM ALERT]: GROQ API LIMIT REACHED. "
            "PLEASE PRESS F8 TO SWAP API KEYS. ≡ƒÜ¿\n\n"
        )
        signals.llm_token.emit(_RATE_LIMIT_MSG)
        signals.llm_end.emit()

    except groq.APIStatusError as exc:
        # Other 4xx / 5xx responses (auth failure, server error, etc.).
        print(f"llm_client: API status error {exc.status_code}: {exc}", file=sys.stderr)
        _API_ERR_MSG = (
            f"\n\n≡ƒÜ¿ [SYSTEM ALERT]: API ERROR {exc.status_code} ΓÇö "
            "CHECK LOGS. ≡ƒÜ¿\n\n"
        )
        signals.llm_token.emit(_API_ERR_MSG)
        signals.llm_end.emit()

    except Exception as exc:
        # Network drop, timeout, or any other unexpected failure.
        print(f"llm_client: generation error: {exc}", file=sys.stderr)
        _CONN_LOST_MSG = (
            "\n\n≡ƒÜ¿ [SYSTEM ALERT]: CONNECTION LOST. ≡ƒÜ¿\n\n"
        )
        signals.llm_token.emit(_CONN_LOST_MSG)
        signals.llm_end.emit()


async def generate_vision_response(base64_image: str, current_audio_transcript: str, signals) -> None:
    """
    Send a base64-encoded screenshot + the current audio transcript to
    Gemini 1.5 Flash and stream the response tokens to the UI via `signals`.

    Parameters
    ----------
    base64_image : str
        Base64-encoded PNG captured by VisionThread.
    current_audio_transcript : str
        The most recent STT transcript ΓÇö gives Gemini spoken context.
    signals : WorkerSignals
        PyQt6 signal emitter for llm_start / llm_token / llm_end.
    """
    if not _GENAI_AVAILABLE or _gemini_model is None:
        signals.llm_token.emit(
            "\n\n≡ƒÜ¿ [VISION]: Gemini not configured. "
            "Set GEMINI_API_KEY and restart. ≡ƒÜ¿\n\n"
        )
        return

    import base64 as _b64
    import google.generativeai as genai

    signals.llm_start.emit()
    try:
        image_part = {
            "mime_type": "image/png",
            "data": _b64.b64decode(base64_image),
        }
        context_text = (
            f"[Audio context from interviewer]: {current_audio_transcript}"
            if current_audio_transcript.strip()
            else "[No audio context available]"
        )

        # Gemini SDK generate_content_async supports streaming.
        response = await _gemini_model.generate_content_async(
            [image_part, context_text],
            stream=True,
        )

        async for chunk in response:
            token = getattr(chunk, "text", None)
            if token:
                signals.llm_token.emit(token)

        signals.llm_end.emit()

    except asyncio.CancelledError:
        signals.llm_end.emit()
        raise
    except Exception as exc:
        print(f"llm_client: vision generation error: {exc}", file=sys.stderr)
        signals.llm_token.emit(f"\n\n≡ƒÜ¿ [VISION ERROR]: {exc} ≡ƒÜ¿\n\n")
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
                    f"llm_client: transcript too short ({len(transcript.split())} word(s)) ΓÇö skipped.",
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
