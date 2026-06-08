import asyncio
import json
import sys
import websockets

DEEPGRAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2"
    "&language=en-US"
    "&encoding=linear16"
    "&sample_rate=16000"
    "&channels=1"
    "&interim_results=true"
    "&endpointing=true"
    "&utterance_end_ms=1000"
    "&vad_events=true"
    "&smart_format=true"
    "&punctuate=true"
    "&keywords=MERN:10"
    "&keywords=MongoDB:10"
    "&keywords=Express:10"
    "&keywords=React:10"
    "&keywords=Node.js:10"
    "&keywords=Vite:10"
    "&keywords=Vercel:10"
    "&keywords=Tailwind:10"
    "&keywords=Cloudinary:10"
    "&keywords=Turnstile:10"
    "&keywords=TensorFlow:10"
    "&keywords=OpenCV:10"
    "&keywords=CNN:10"
    "&keywords=Synex%20AI:10"
    "&keywords=Voicify:10"
    "&keywords=TMS%20Security:10"
    "&keywords=Groq:10"
    "&replace=mon%20stack:MERN%20stack"
    "&replace=mon:MERN"
    "&replace=grook:Groq"
    "&replace=sin%20ex:Synex"
    "&replace=voice%20if%20i:Voicify"
)

async def _sender(ws, queue):
    try:
        while True:
            chunk = await queue.get()
            await ws.send(chunk)
    except asyncio.CancelledError:
        try:
            await ws.send(json.dumps({"type": "CloseStream"}))
        except Exception:
            pass
        raise

async def _receiver(ws, signals, llm_queue):
    final_parts = []
    while True:
        try:
            message = await ws.recv()
        except websockets.ConnectionClosed:
            break
        try:
            data = json.loads(message)
        except json.JSONDecodeError as exc:
            print(f"stt_client: malformed JSON: {exc}", file=sys.stderr)
            continue
        msg_type = data.get("type", "")
        if msg_type == "Results":
            alternatives = data.get("channel", {}).get("alternatives", [])
            if not alternatives:
                continue
            transcript = alternatives[0].get("transcript", "").strip()
            if not transcript:
                continue
            if data.get("is_final", False):
                final_parts.append(transcript)
                if data.get("speech_final", False):
                    final_text = " ".join(final_parts).strip()
                    final_parts.clear()
                    signals.final_transcript.emit(final_text)
                    if not llm_queue.full():
                        llm_queue.put_nowait(final_text)
            else:
                signals.interim_transcript.emit(transcript)
        elif msg_type == "UtteranceEnd":
            if final_parts:
                final_text = " ".join(final_parts).strip()
                final_parts.clear()
                signals.final_transcript.emit(final_text)
                if not llm_queue.full():
                    llm_queue.put_nowait(final_text)
        elif msg_type in ("SpeechStarted", "Metadata"):
            continue
        else:
            if msg_type:
                print(f"stt_client: unhandled message type: {msg_type!r}", file=sys.stderr)

def _drain_queue(queue):
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break

async def _connect_and_run(audio_queue, llm_queue, signals, api_key):
    headers = {"Authorization": f"Token {api_key}"}
    signals.status_update.emit("initializing")

    while True:
        _drain_queue(audio_queue)
        try:
            async with websockets.connect(DEEPGRAM_URL, additional_headers=headers) as ws:
                signals.status_update.emit("connected")

                sender_task = asyncio.create_task(_sender(ws, audio_queue))
                receiver_task = asyncio.create_task(_receiver(ws, signals, llm_queue))

                done, pending = await asyncio.wait(
                    {sender_task, receiver_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

                for task in done:
                    if not task.cancelled():
                        exc = task.exception()
                        if exc is not None:
                            print(f"stt_client: task error: {exc}", file=sys.stderr)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"stt_client: connection error: {exc}", file=sys.stderr)

        signals.status_update.emit("disconnected")
        print("Reconnecting to Deepgram...", file=sys.stderr)
        await asyncio.sleep(1.5)
        signals.status_update.emit("initializing")

async def run_stt(audio_queue, llm_queue, signals, api_key):
    try:
        await _connect_and_run(audio_queue, llm_queue, signals, api_key)
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        print(f"stt_client: fatal error in run_stt: {exc}", file=sys.stderr)