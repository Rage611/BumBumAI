import asyncio
import sys
from collections import deque
from pathlib import Path

import groq
from groq import AsyncGroq
from openai import AsyncOpenAI
import config as _config

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
6. FORMATTING: Use absolutely NO markdown bolding (**). Keep text plain and use simple line breaks.
7. HINGLISH UNDERSTANDING: If the interviewer asks the question in Hinglish (a mixture of Hindi and English like 'is function me time complexity kya hai'), understand the technical intent perfectly and output your response in clear, confident, spoken English for me to repeat out loud.

CRITICAL CODING RULES:
- If asked for code (like a LeetCode problem), ALWAYS provide the solution in C++.
- You MUST remove all comments from the generated code.

--- MY RESUME (Context for my background and specific projects) ---
{RESUME_CONTEXT}
--- END RESUME ---"""


# ---------------------------------------------------------------------------
# Vision configuration
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
    "  If B (CODE EXPLANATION / REVIEW): Look for mouse cursor, active cursor line, text selection, or specific code highlighted/pointed at in VS Code/IDE. Output a confident, spoken 3-5 sentence explanation answering what that code snippet does, how it works, and its purpose.\n"
    "  If C (THEORETICAL): Output a concise, spoken-style explanation (4-6 sentences max).\n"
    "  If D (SYSTEM DESIGN): Describe the architecture approach with key components and tradeoffs.\n"
    "  If E (BEHAVIORAL): Output a confident, first-person spoken answer.\n"
    "  If F (UNCLEAR): Describe what you see on screen and answer concisely.\n\n"
    "CRITICAL RULES:\n"
    "- Look closely for mouse pointer position, highlighted text selection, or active IDE line in VS Code / screen.\n"
    "- If asked 'what does this code do' or shown an IDE, explain the code logic in spoken first-person voice. Do NOT output raw code unless asked to write code.\n"
    "- Do NOT use markdown bolding (**). Keep text plain.\n"
)

_openai_api_key = ""

def configure_openai(api_key: str) -> None:
    global _openai_api_key
    _openai_api_key = api_key
    if api_key:
        print("llm_client: OpenAI configured.", file=sys.stderr)


# ---------------------------------------------------------------------------
# ProviderChain — automatic failover on quota exhaustion
# ---------------------------------------------------------------------------

# Error signatures that indicate quota/billing limit (not auth or server errors)
_QUOTA_ERRORS = (
    "429", "rate_limit", "rate limit", "insufficient_quota",
    "billing_hard_limit", "quota exceeded", "capacity",
    "RateLimitError", "insufficient quota",
)

def _is_quota_error(exc: Exception) -> bool:
    """Return True if this exception looks like a quota/rate-limit hit."""
    msg = str(exc).lower()
    return any(k.lower() in msg for k in _QUOTA_ERRORS)


class ProviderChain:
    """
    Holds an ordered list of (client, model_name, label) providers.
    On quota exhaustion, call failover() to advance to the next one.
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
        """Try to advance to the next provider. Returns False if exhausted."""
        if self._idx + 1 >= len(self._providers):
            return False
        self._idx += 1
        return True

    async def close_all(self):
        for p in self._providers:
            close = getattr(p["client"], "close", None) or getattr(p["client"], "aclose", None)
            if close:
                result = close()
                if asyncio.iscoroutine(result):
                    await result


_DEFAULT_CHAIN = [
    {"provider": "openai", "model": "gpt-5.6-luna"},
    {"provider": "groq",   "model": "llama-3.3-70b-versatile"},
    {"provider": "groq",   "model": "llama-3.1-8b-instant"},
    {"provider": "openai", "model": "gpt-4o-mini"},
]

def _build_chain(openai_key: str, groq_key: str) -> ProviderChain:
    """
    Build the provider chain in the exact order the user configured
    in Settings (LLM_CHAIN). Falls back to _DEFAULT_CHAIN if not set.
    Skips any entry whose required API key is missing.
    """
    cfg = _config.load_config()
    chain_cfg = cfg.get("LLM_CHAIN", _DEFAULT_CHAIN)

    chain = ProviderChain()
    seen: set[str] = set()

    for entry in chain_cfg:
        if isinstance(entry, dict):
            provider = entry.get("provider")
            model    = entry.get("model")
        else:
            provider, model = entry[0], entry[1]

        if not provider or not model:
            continue

        key_id = f"{provider}:{model}"
        if key_id in seen:
            continue
        seen.add(key_id)

        if provider == "openai" and openai_key:
            label = f"OpenAI {model}"
            chain.add(AsyncOpenAI(api_key=openai_key), model, label)
        elif provider == "groq" and groq_key:
            label = f"Groq {model}"
            chain.add(AsyncGroq(api_key=groq_key), model, label)
        else:
            print(f"llm_client: skipping {provider}/{model} — no key.", file=sys.stderr)

    return chain



# ---------------------------------------------------------------------------
# Text LLM generation with per-call failover
# ---------------------------------------------------------------------------
_current_task = None
_history: deque = deque(maxlen=6)
_chain: ProviderChain | None = None   # set by run_llm at startup


async def _generate(transcript: str, signals) -> None:
    """Generate a response, auto-failing over providers on quota errors."""
    global _chain
    if _chain is None or _chain.empty:
        signals.llm_token.emit("\n\n🚨 No LLM providers available. 🚨\n\n")
        signals.llm_end.emit()
        return

    signals.llm_start.emit()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(_history)
    messages.append({"role": "user", "content": transcript})
    full_response = ""

    while True:
        client    = _chain.client
        model     = _chain.model
        label     = _chain.label

        try:
            kwargs: dict = {"model": model, "messages": messages, "stream": True}
            if "llama" in model:
                kwargs["max_tokens"] = 150
                kwargs["temperature"] = 0.3
                kwargs["top_p"] = 0.9
            else:
                kwargs["max_completion_tokens"] = 150

            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                token = chunk.choices[0].delta.content
                if token:
                    full_response += token
                    signals.llm_token.emit(token)

            _history.append({"role": "user",      "content": transcript})
            _history.append({"role": "assistant",  "content": full_response})
            signals.llm_end.emit()
            return  # success — done

        except asyncio.CancelledError:
            signals.llm_end.emit()
            raise

        except Exception as exc:
            if _is_quota_error(exc):
                print(f"llm_client: quota hit on {label} — trying next provider.", file=sys.stderr)
                if _chain.failover():
                    # Notify the UI via a brief inline message
                    signals.llm_token.emit(
                        f"\n\n⚡ [Quota hit — switching to {_chain.label}...]\n\n"
                    )
                    # Reset response so we start fresh on next provider
                    full_response = ""
                    continue
                else:
                    signals.llm_token.emit("\n\n🚨 All providers quota-exhausted. 🚨\n\n")
                    signals.llm_end.emit()
                    return
            else:
                # Non-quota error (auth, server error, network) — show and stop
                print(f"llm_client: generation error on {label}: {exc}", file=sys.stderr)
                signals.llm_token.emit(f"\n\n🚨 Error ({label}): {exc} 🚨\n\n")
                signals.llm_end.emit()
                return


async def generate_vision_response(base64_image: str, current_audio_transcript: str, signals) -> None:
    if not _openai_api_key:
        signals.llm_token.emit("\n\n🚨 [VISION]: OpenAI API Key missing. 🚨\n\n")
        signals.llm_end.emit()
        return

    verbal_context = current_audio_transcript.strip() or "No verbal context captured."
    fused_prompt = (
        "Visual State: [Attached Screenshot]\n\n"
        f"Verbal Context: {verbal_context}\n\n"
        "Instruction: Classify the visual content and respond."
    )

    signals.llm_start.emit()
    client = AsyncOpenAI(api_key=_openai_api_key)

    try:
        messages = [
            {"role": "system", "content": VISION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": fused_prompt},
                ],
            },
        ]
        response = await client.chat.completions.create(
            model="gpt-5.6-luna",
            messages=messages,
            stream=True,
            max_completion_tokens=300,
        )
        async for chunk in response:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content
            if token:
                signals.llm_token.emit(token)
        signals.llm_end.emit()

    except asyncio.CancelledError:
        signals.llm_end.emit()
        raise
    except Exception as exc:
        print(f"llm_client: vision error: {exc}", file=sys.stderr)
        signals.llm_token.emit(f"\n\n🚨 [VISION ERROR]: {exc} 🚨\n\n")
        signals.llm_end.emit()


async def run_llm(llm_queue, signals, groq_api_key: str):
    global _current_task, _chain

    _chain = _build_chain(_openai_api_key, groq_api_key)

    if _chain.empty:
        print("llm_client: No API keys available. LLM disabled.", file=sys.stderr)
        return

    print(f"llm_client: ProviderChain ready — primary: {_chain.label}.", file=sys.stderr)

    try:
        while True:
            transcript = await llm_queue.get()
            if len(transcript.split()) < 4:
                continue

            if _current_task is not None and not _current_task.done():
                _current_task.cancel()
                try:
                    await _current_task
                except asyncio.CancelledError:
                    pass

            _current_task = asyncio.create_task(_generate(transcript, signals))

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