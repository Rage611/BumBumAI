import asyncio
import sys
from collections import deque
from pathlib import Path

from groq import AsyncGroq
import config as _config

# ---------------------------------------------------------------------------
# Resume context injection
# ---------------------------------------------------------------------------
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
1. EXTREME BREVITY: Answer in 3 to 6 sentences MAX. No exceptions. Get straight to the point. NO fluff, NO filler, NO analogies, NO concluding summary sentences. Every word must earn its place.
2. THE "ELEVATOR PITCH" OVERRIDE: If the interviewer asks "tell me about yourself," "introduce yourself," or "walk me through your background," you MUST deliver a confident, conversational 4 to 5 sentence elevator pitch. Introduce yourself as a Computer Science undergrad at Bennett University and a freelance software developer based in New Delhi. Briefly mention your expertise in the MERN stack, your background in competitive programming with C++, and highlight your drive for building production-grade AI tools like Synex AI and Voicify. Sound human, passionate, and professional.
3. THE GUARD GRID PIVOT (CRITICAL): I have an unlisted recent project called "Guard Grid" (a Next.js/Supabase app for TMS Security Services). If the interviewer asks a general question about Next.js, Supabase, database security (RLS), multi-tenant architecture, OR a behavioral question like "tell me about a recent technical challenge," you MUST pivot and use Guard Grid as your real-world example. Use phrasing like, "Actually, I recently implemented this in a project I built called Guard Grid..."
4. RESUME TRIGGER & EXACT NAMES: Before answering, ALWAYS check if the interviewer's question contains words matching my projects (like "Synex", "TMS Security", "Voicify", "Guard Grid"). If they do, you MUST use the resume context to answer as me. CRITICAL: You MUST use the exact proper names of my projects and companies. Never generalize my work.
5. TONE: Casual, spoken, conversational. Use short, punchy sentences. Sound like a confident human engineer, not an AI essay.
6. TELEPROMPTER FORMATTING: You are writing for a teleprompter. You MUST insert a double line break (\\n\\n) after EVERY SINGLE SENTENCE. Write in short, bite-sized fragments so I can naturally pause, breathe, and look at the camera. Use absolutely NO markdown bolding (**). Keep text plain.
7. HINGLISH UNDERSTANDING: If the interviewer asks the question in Hinglish (a mixture of Hindi and English like 'is function me time complexity kya hai'), understand the technical intent perfectly and output your response in clear, confident, spoken English for me to repeat out loud.

CRITICAL CODING RULES:
- If asked for code (like a LeetCode problem), ALWAYS provide the solution in C++.
- You MUST remove all comments from the generated code.

--- MY RESUME (Context for my background and specific projects) ---
{RESUME_CONTEXT}
--- END RESUME ---"""


# ---------------------------------------------------------------------------
# Vision prompt (used by Gemini)
# ---------------------------------------------------------------------------
VISION_PROMPT = (
    "You are a stealth interview assistant analyzing a screenshot.\n\n"
    "STEP 1 - CLASSIFY: Look at the screenshot and determine the question type:\n"
    "  A) LEETCODE / ALGORITHM PROBLEM (LeetCode, HackerRank, coding puzzle requiring full solution)\n"
    "  B) CODE EXPLANATION / REVIEW (VS Code, IDE, code snippet explanation, pointed code, function walkthrough)\n"
    "  C) THEORETICAL QUESTION (concept explanation, definition, comparison)\n"
    "  D) SYSTEM DESIGN (architecture diagram, design discussion)\n"
    "  E) BEHAVIORAL (HR question, tell me about yourself type)\n"
    "  F) OTHER / UNCLEAR\n\n"
    "STEP 2 - RESPOND based on your classification:\n"
    "  If A (LEETCODE / ALGORITHM): Output ONLY the optimal C++ solution. No comments. No explanation.\n"
    "  If B (CODE EXPLANATION / REVIEW): Look for mouse cursor, active cursor line, text selection. Output a confident, spoken 3-5 sentence explanation.\n"
    "  If C (THEORETICAL): Output a concise, spoken-style explanation (4-6 sentences max).\n"
    "  If D (SYSTEM DESIGN): Describe the architecture approach with key components and tradeoffs.\n"
    "  If E (BEHAVIORAL): Output a confident, first-person spoken answer.\n"
    "  If F (UNCLEAR): Describe what you see on screen and answer concisely.\n\n"
    "CRITICAL RULES:\n"
    "- TELEPROMPTER FORMATTING: You MUST insert a double line break (\\n\\n) after EVERY SINGLE SENTENCE. Write in short, bite-sized fragments so I can naturally pause and breathe. Do NOT write long paragraphs!\n"
    "- Do NOT use markdown bolding (**). Keep text plain.\n"
)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
_QUOTA_ERRORS = (
    "429", "rate_limit", "rate limit", "insufficient_quota",
    "billing_hard_limit", "quota exceeded", "capacity",
    "RateLimitError", "insufficient quota",
)

def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k.lower() in msg for k in _QUOTA_ERRORS)


# ---------------------------------------------------------------------------
# ProviderChain — multi-key rotation then model failover
# ---------------------------------------------------------------------------
class ProviderChain:
    """
    Ordered list of (client, model, label) entries.
    Multiple entries with different keys for the same model allow per-key rotation:
      - Groq key1/llama-3.3-70b → Groq key2/llama-3.3-70b → ... → Groq key1/llama-3.1-8b → ...
    Calling failover() advances to the next entry.
    """

    def __init__(self):
        self._providers: list[dict] = []
        self._idx = 0

    def add(self, client, model_name: str, label: str) -> None:
        self._providers.append({"client": client, "model": model_name, "label": label})

    @property
    def empty(self) -> bool:
        return len(self._providers) == 0

    @property
    def client(self):
        return self._providers[self._idx]["client"]

    @property
    def model(self) -> str:
        return self._providers[self._idx]["model"]

    @property
    def label(self) -> str:
        return self._providers[self._idx]["label"]

    def failover(self) -> bool:
        if self._idx + 1 >= len(self._providers):
            return False
        self._idx += 1
        return True

    def reset(self) -> None:
        """Reset to primary provider (call after successful generation)."""
        self._idx = 0

    async def close_all(self):
        for p in self._providers:
            close = getattr(p["client"], "close", None) or getattr(p["client"], "aclose", None)
            if close:
                result = close()
                if asyncio.iscoroutine(result):
                    await result


# ---------------------------------------------------------------------------
# Chain builder — expands each model into N entries (one per API key)
# ---------------------------------------------------------------------------
_DEFAULT_CHAIN = [
    {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    {"provider": "groq", "model": "llama-3.1-8b-instant"},
]

def _build_chain(groq_keys: list[str]) -> ProviderChain:
    """
    Build the provider chain from config.
    For each model in LLM_CHAIN, add one entry per available API key.
    This means key rotation happens before model failover.
    """
    cfg = _config.load_config()
    chain_cfg = cfg.get("LLM_CHAIN", _DEFAULT_CHAIN)
    valid_groq_keys = [k for k in groq_keys if k.strip()]

    chain = ProviderChain()
    seen_combos: set[str] = set()

    for entry in chain_cfg:
        if isinstance(entry, dict):
            provider = entry.get("provider")
            model = entry.get("model")
        else:
            provider, model = entry[0], entry[1]

        if not provider or not model:
            continue

        if provider == "groq":
            for i, key in enumerate(valid_groq_keys):
                combo = f"groq:{model}:key{i}"
                if combo in seen_combos:
                    continue
                seen_combos.add(combo)
                label = f"Groq {model} (key {i+1})"
                chain.add(AsyncGroq(api_key=key), model, label)
        else:
            print(f"llm_client: unknown provider '{provider}' — skipping.", file=sys.stderr)

    return chain


# ---------------------------------------------------------------------------
# Conversation history with rolling summary
# ---------------------------------------------------------------------------
_history: deque = deque(maxlen=20)   # 10 full Q&A exchanges
_chain: ProviderChain | None = None
_current_task: asyncio.Task | None = None
_groq_keys: list[str] = []


async def _summarize_history() -> str:
    """Compress the oldest half of history into a single summary message using the LLM."""
    if _chain is None or _chain.empty:
        return ""
    to_summarize = list(_history)[:10]
    summary_msgs = [
        {"role": "system", "content": "Summarize the following interview Q&A in 3 concise sentences for context. Be factual and brief."},
        {"role": "user", "content": str(to_summarize)},
    ]
    try:
        stream = await _chain.client.chat.completions.create(
            model=_chain.model,
            messages=summary_msgs,
            max_tokens=120,
            stream=True,
        )
        summary_text = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                summary_text += chunk.choices[0].delta.content
        return summary_text.strip()
    except Exception:
        return ""


async def _generate(transcript: str, signals) -> None:
    """Generate a response, rotating through keys then models on quota errors."""
    global _chain
    if _chain is None or _chain.empty:
        signals.llm_token.emit("\n\n🚨 No LLM providers available. 🚨\n\n")
        signals.llm_end.emit()
        return

    # Rolling summary: if history is full, compress the oldest half
    if len(_history) >= 20:
        summary = await _summarize_history()
        # Remove the oldest 10 messages and replace with summary
        for _ in range(10):
            if _history:
                _history.popleft()
        if summary:
            _history.appendleft({"role": "system", "content": f"Earlier interview context (summarized): {summary}"})

    signals.llm_start.emit()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(_history)
    messages.append({"role": "user", "content": transcript})
    full_response = ""

    while True:
        client = _chain.client
        model = _chain.model
        label = _chain.label

        try:
            kwargs: dict = {
                "model": model,
                "messages": messages,
                "stream": True,
                "max_tokens": 150,
                "temperature": 0.3,
                "top_p": 0.9,
            }
            stream = await client.chat.completions.create(**kwargs)
            
            try:
                limit = int(stream.response.headers.get("x-ratelimit-limit-tokens", 0))
                remaining = int(stream.response.headers.get("x-ratelimit-remaining-tokens", 0))
                if limit > 0:
                    pct = max(0, min(100, int(((limit - remaining) / limit) * 100)))
                    signals.token_usage_update.emit(pct)
            except Exception:
                pass

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
            return  # success

        except asyncio.CancelledError:
            signals.llm_end.emit()
            raise

        except Exception as exc:
            if _is_quota_error(exc):
                print(f"llm_client: quota/rate hit on {label} — trying next.", file=sys.stderr)
                if _chain.failover():
                    signals.llm_token.emit(f"\n\n⚡ [Switching to {_chain.label}...]\n\n")
                    full_response = ""
                    continue
                else:
                    signals.llm_token.emit("\n\n🚨 All providers exhausted. 🚨\n\n")
                    signals.llm_end.emit()
                    return
            else:
                print(f"llm_client: error on {label}: {exc}", file=sys.stderr)
                signals.llm_token.emit(f"\n\n🚨 Error ({label}): {exc} 🚨\n\n")
                signals.llm_end.emit()
                return


# ---------------------------------------------------------------------------
# Vision — Google Gemini 2.5 Flash (free tier)
# ---------------------------------------------------------------------------
_google_keys: list[str] = []
_google_key_idx: int = 0


def configure_vision(google_keys: list[str]) -> None:
    global _google_keys
    _google_keys = [k for k in google_keys if k.strip()]
    if _google_keys:
        print(f"llm_client: {len(_google_keys)} Google AI Studio key(s) configured.", file=sys.stderr)
    else:
        print("llm_client: No Google API keys — vision disabled.", file=sys.stderr)


async def generate_vision_response(base64_image: str, current_audio_transcript: str, signals) -> None:
    global _google_key_idx

    if not _google_keys:
        signals.llm_token.emit("\n\n🚨 [VISION]: No Google API key configured. 🚨\n\n")
        signals.llm_end.emit()
        return

    verbal_context = current_audio_transcript.strip() or "No verbal context captured."
    fused_prompt = (
        f"{VISION_PROMPT}\n\n"
        f"Verbal Context from audio: {verbal_context}\n\n"
        "Classify the screenshot and respond."
    )

    signals.llm_start.emit()

    # Try each Google key in rotation on quota error
    attempts = 0
    while attempts < len(_google_keys):
        key = _google_keys[_google_key_idx % len(_google_keys)]
        key_label = f"Google key {(_google_key_idx % len(_google_keys)) + 1}"
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=key)
            response_stream = await asyncio.to_thread(
                client.models.generate_content_stream,
                model="gemini-3.5-flash",
                contents=[
                    types.Part.from_bytes(
                        data=__import__("base64").b64decode(base64_image),
                        mime_type="image/jpeg",
                    ),
                    fused_prompt,
                ],
            )
            for chunk in response_stream:
                text = chunk.text or ""
                if text:
                    signals.llm_token.emit(text)
            signals.llm_end.emit()
            return

        except asyncio.CancelledError:
            signals.llm_end.emit()
            raise
        except Exception as exc:
            if _is_quota_error(exc):
                print(f"llm_client: vision quota on {key_label} — rotating.", file=sys.stderr)
                _google_key_idx += 1
                attempts += 1
                if attempts < len(_google_keys):
                    signals.llm_token.emit(f"\n\n⚡ [Vision switching to Google key {(_google_key_idx % len(_google_keys)) + 1}...]\n\n")
                continue
            else:
                print(f"llm_client: vision error: {exc}", file=sys.stderr)
                signals.llm_token.emit(f"\n\n🚨 [VISION ERROR]: {exc} 🚨\n\n")
                signals.llm_end.emit()
                return

    signals.llm_token.emit("\n\n🚨 All Google API keys exhausted for vision. 🚨\n\n")
    signals.llm_end.emit()


# ---------------------------------------------------------------------------
# Public entry point — debounced LLM runner
# ---------------------------------------------------------------------------
async def run_llm(llm_queue: asyncio.Queue, signals, groq_keys: list[str]) -> None:
    global _current_task, _chain, _groq_keys
    _groq_keys = groq_keys
    _chain = _build_chain(groq_keys)

    if _chain.empty:
        print("llm_client: No valid Groq API keys. LLM disabled.", file=sys.stderr)
        return

    print(f"llm_client: ProviderChain ready — {len(_chain._providers)} slot(s). Primary: {_chain.label}.", file=sys.stderr)

    # Debounce state
    DEBOUNCE_SECONDS = 1.5
    accumulated: list[str] = []
    debounce_task: asyncio.Task | None = None

    async def _fire_after_silence():
        """Wait for silence, then combine accumulated fragments and fire LLM."""
        nonlocal accumulated, debounce_task
        await asyncio.sleep(DEBOUNCE_SECONDS)
        if not accumulated:
            return
        combined = " ".join(accumulated).strip()
        accumulated = []
        debounce_task = None

        if len(combined.split()) < 4:
            return  # too short, ignore

        # Cancel any previous generation
        global _current_task
        if _current_task is not None and not _current_task.done():
            _current_task.cancel()
            try:
                await _current_task
            except asyncio.CancelledError:
                pass

        _current_task = asyncio.create_task(_generate(combined, signals))

    try:
        while True:
            transcript = await llm_queue.get()

            # Accumulate transcript fragment
            accumulated.append(transcript)

            # Reset the debounce timer
            if debounce_task is not None and not debounce_task.done():
                debounce_task.cancel()
                try:
                    await asyncio.shield(asyncio.sleep(0))  # yield
                except asyncio.CancelledError:
                    pass

            debounce_task = asyncio.create_task(_fire_after_silence())

    except asyncio.CancelledError:
        if _current_task is not None and not _current_task.done():
            _current_task.cancel()
            try:
                await _current_task
            except asyncio.CancelledError:
                pass
        raise
    finally:
        await _chain.close_all()