import asyncio
import json
import sys

import websockets

DEEPGRAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-3"
    "&language=en-US"
    "&encoding=linear16"
    "&sample_rate=16000"
    "&channels=1"
    "&interim_results=true"
    "&endpointing=300"
    "&utterance_end_ms=1500"
    "&vad_events=true"
    "&smart_format=true"
    "&punctuate=true"
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
            channel = data.get("channel", {})
            alternatives = channel.get("alternatives", [])
            if not alternatives:
                continue
            transcript = alternatives[0].get("transcript", "")
            if not transcript:
                continue
            is_final = data.get("is_final", False)
            speech_final = data.get("speech_final", False)
            if not is_final:
                signals.interim_transcript.emit(transcript)
            elif is_final and speech_final:
                signals.final_transcript.emit(transcript)
                if not llm_queue.full():
                    llm_queue.put_nowait(transcript)

        elif msg_type in ("UtteranceEnd", "SpeechStarted", "Metadata"):
            continue

        else:
            if msg_type:
                print(f"stt_client: unhandled message type: {msg_type!r}", file=sys.stderr)


async def _connect_and_run(audio_queue, llm_queue, signals, api_key):
    headers = {"Authorization": f"Token {api_key}"}
    signals.status_update.emit("initializing")

    while True:
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
