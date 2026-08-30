import asyncio
import sys
from collections import deque


from groq import AsyncGroq
import config as _config

# ---------------------------------------------------------------------------
# System prompt — lean, output-focused, no resume
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are my real-time interview assistant. You hear what the interviewer asks and you give me the EXACT words I should say out loud.

RULES:
1. EXTREMELY SIMPLE ENGLISH: Talk like a normal 21-year-old college student. Use basic everyday words. Do NOT sound like a textbook or an AI.
2. BANNED WORDS: Never use these — orchestrate, leverage, paradigm, pivotal, delve, testament, cutting-edge, in essence, seamlessly, simultaneously, exclusive, cripples, starvation, crucial, vital, classic issue, moreover, furthermore, comprehensive, robust.
   - INSTEAD say: "at the same time", "only one can change it", "makes it really slow", "Yeah so basically...", "What I did was...", "The main problem was...", "So to fix that I just..."
3. ANSWER LENGTH:
   - Quick concept questions: 2-3 sentences max.
   - Deeper technical questions: 4-6 sentences, straight to the point.
4. TELEPROMPTER FORMAT:
   - Put a double line break after EVERY sentence so I can pause and breathe.
   - NO markdown bolding (**). Plain text only.
5. HINGLISH: If the interviewer speaks in Hindi/Hinglish, understand the intent and reply in simple English.
6. DSA / CODING: Whenever a DSA or algorithm problem is mentioned (even if the interviewer says "what's your approach" or "how would you solve this"), ALWAYS output BOTH in this exact order:
   - FIRST: The full C++ code solution. Write it like a beginner — simple for-loops, basic if-else, simple arrays and vectors. No complex STL, no auto, no lambda, no fancy one-liners. Keep variable names simple (i, j, n, arr, ans). Zero comments.
   - THEN: Below the code, write a short spoken-style approach explanation (3-5 sentences) that I can say out loud. Use extremely simple words. Put a double line break after every sentence.
   - NEVER skip the code. NEVER give only the approach without code. ALWAYS give both.
7. NEVER ASK QUESTIONS BACK: You must NEVER ask the interviewer for clarification, more details, or the full problem statement. NEVER say things like "Could you tell me more?", "What exactly do they want?", "Let me know the details". You are a teleprompter — you ONLY output answers. If the question is incomplete or unclear, just answer with whatever information you have. Make reasonable assumptions and give the best possible answer immediately.
"""

# ---------------------------------------------------------------------------
# Vision prompt (used by Gemini)
# ---------------------------------------------------------------------------
VISION_PROMPT = (
    "You are a stealth interview assistant analyzing a screenshot and the interviewer's verbal question.\n\n"
    "Look at the screenshot and the interviewer's verbal context. Based on what they are asking, provide the EXACT response I should say out loud or type.\n\n"
    "CRITICAL RULES FOR DSA / CODING:\n"
    "- If the interviewer asks for the CODE or SOLUTION, output C++ that a beginner would write. Simple for-loops, basic if-else, simple arrays and vectors. No complex STL, no auto, no lambda, no fancy one-liners. No comments. Then below the code, write a short 3-5 sentence spoken approach explanation.\n"
    "- If the interviewer ONLY asks 'Walk me through your approach', 'How would you solve this', or 'Explain the logic', output ONLY a spoken-style step-by-step approach in 3-6 sentences using EXTREMELY SIMPLE WORDS. Do NOT output code in this case.\n\n"
    "CRITICAL RULES FOR GENERAL QUESTIONS:\n"
    "- Output a confident, spoken-style explanation using EXTREMELY SIMPLE, BASIC ENGLISH.\n"
    "- Speak like a normal 21-year-old student casually talking to a friend.\n"
    "- DO NOT use textbook/formal words like: orchestrate, leverage, paradigm, pivotal, delve, testament, simultaneously, exclusive, cripples, starvation, crucial, vital, classic issue, moreover, furthermore.\n"
    "- INSTEAD USE: 'at the same time', 'only one person can change it', 'makes it really slow'.\n"
    "- TELEPROMPTER FORMATTING: You MUST insert a double line break (\\n\\n) after EVERY SINGLE SENTENCE. Write in short, bite-sized fragments so I can naturally pause and breathe.\n"
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

    def add(self, p_type: str, client, model_name: str, label: str) -> None:
        self._providers.append({"type": p_type, "client": client, "model": model_name, "label": label})

    @property
    def empty(self) -> bool:
        return len(self._providers) == 0

    @property
    def type(self) -> str:
        return self._providers[self._idx]["type"]

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
    {"provider": "groq",     "model": "openai/gpt-oss-120b"},
    {"provider": "google",   "model": "gemini-3.6-flash"},
    {"provider": "cerebras", "model": "gpt-oss-120b"},
]

def _build_chain() -> ProviderChain:
    """Reads LLM_CHAIN and all API key lists from config.json. No arguments needed."""
    cfg = _config.load_config()
    chain_cfg = cfg.get("LLM_CHAIN", _DEFAULT_CHAIN)

    google_keys   = [k for k in cfg.get("GOOGLE_API_KEYS",   []) if k.strip()]
    groq_keys     = [k for k in cfg.get("GROQ_API_KEYS",     []) if k.strip()]
    cerebras_keys = [k for k in cfg.get("CEREBRAS_API_KEYS", []) if k.strip()]

    chain = ProviderChain()

    for entry in chain_cfg:
        provider = entry.get("provider") if isinstance(entry, dict) else entry[0]
        model    = entry.get("model")    if isinstance(entry, dict) else entry[1]

        if not provider or not model:
            continue

        if provider == "google":
            try:
                from google import genai
            except ImportError:
                print("llm_client: google-genai not installed — skipping Google.", file=sys.stderr)
                continue
            for i, key in enumerate(google_keys):
                chain.add("google", genai.Client(api_key=key), model, f"Google {model} (key {i+1})")

        elif provider == "groq":
            for i, key in enumerate(groq_keys):
                chain.add("groq", AsyncGroq(api_key=key), model, f"Groq {model} (key {i+1})")

        elif provider == "cerebras":
            for i, key in enumerate(cerebras_keys):
                chain.add(
                    "cerebras",
                    AsyncGroq(api_key=key, base_url="https://api.cerebras.ai/v1"),
                    model,
                    f"Cerebras {model} (key {i+1})",
                )
        else:
            print(f"llm_client: unknown provider '{provider}' — skipping.", file=sys.stderr)

    return chain


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_history: deque = deque(maxlen=30)
_chain: ProviderChain | None = None
_current_task: asyncio.Task | None = None



async def _generate(transcript: str, signals) -> None:
    """Generate a response, rotating through keys then models on quota errors."""
    global _chain
    if _chain is None or _chain.empty:
        signals.llm_token.emit("\n\n🚨 No LLM providers available. 🚨\n\n")
        signals.llm_end.emit()
        return

    # Strictly keep only the last 5 turns (10 messages) to save massive tokens
    while len(_history) > 10:
        _history.popleft()

    signals.llm_start.emit()
    
    # Consolidate history to prevent strict-API errors (e.g. consecutive 'user' roles)
    consolidated_history = []
    for msg in _history:
        if consolidated_history and consolidated_history[-1]["role"] == msg["role"]:
            consolidated_history[-1]["content"] += "\n" + msg["content"]
        else:
            consolidated_history.append({"role": msg["role"], "content": msg["content"]})

    if consolidated_history and consolidated_history[-1]["role"] == "user":
        consolidated_history[-1]["content"] += "\n" + transcript
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + consolidated_history
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + consolidated_history + [{"role": "user", "content": transcript}]
        
    full_response = ""

    while True:
        client = _chain.client
        model = _chain.model
        label = _chain.label
        p_type = _chain.type

        try:
            if p_type == "gemini":
                from google.genai import types
                gemini_contents = []
                for m in messages:
                    gemini_contents.append(types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part.from_text(text=m["content"])]))
                stream = await client.aio.models.generate_content_stream(
                    model=model,
                    contents=gemini_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.3,
                    )
                )
                
                async for chunk in stream:
                    token = chunk.text or ""
                    if token:
                        full_response += token
                        signals.llm_token.emit(token)

            elif p_type in ["groq", "cerebras"]:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": 350,
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
            if transcript.strip():
                _history.append({"role": "user", "content": transcript})
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


def rebuild_chain() -> None:
    """Rebuild the provider chain from the latest config. Call after saving settings or switching models."""
    global _chain
    _chain = _build_chain()
    label = _chain.label if not _chain.empty else "empty"
    print(f"llm_client: Chain rebuilt — {len(_chain._providers)} slot(s). Primary: {label}.", file=sys.stderr)


async def run_llm(llm_queue: asyncio.Queue, signals) -> None:
    global _current_task, _chain
    _chain = _build_chain()

    if _chain.empty:
        print("llm_client: No valid API keys found. LLM disabled.", file=sys.stderr)
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