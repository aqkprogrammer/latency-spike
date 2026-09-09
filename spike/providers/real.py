"""Real providers, over raw websockets and SSE rather than vendor SDKs.

Why raw: wire protocols move slower than SDK surfaces, the dependency list stays
at one library, and when a vendor changes something you fix twenty lines here
instead of chasing a breaking release across the project.

Each class carries a VERIFY comment with the doc URL. Check those before you
trust the first number this produces - request shapes drift and a wrong
parameter usually shows up as a plausible-but-wrong latency rather than an error.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator

import aiohttp

SAMPLE_RATE = 16_000


# --------------------------------------------------------------------------- STT
class DeepgramSTT:
    """VERIFY: https://developers.deepgram.com/docs/streaming

    endpointing=false is deliberate. We do our own endpointing and want the
    recogniser to finalise when we tell it to, not on its own silence timer -
    otherwise you are measuring their VAD instead of your pipeline.
    """

    URL = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-3&language=multi&encoding=linear16"
        f"&sample_rate={SAMPLE_RATE}&channels=1"
        "&interim_results=true&endpointing=false&punctuate=true"
    )

    def __init__(self, api_key: str | None = None) -> None:
        self.key = api_key or os.environ["DEEPGRAM_API_KEY"]
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._final = asyncio.Future()
        self._reader: asyncio.Task | None = None
        self._parts: list[str] = []
        self._on_partial = None

    def set_partial_handler(self, fn) -> None:
        self._on_partial = fn

    async def open(self) -> None:
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(
            self.URL, headers={"Authorization": f"Token {self.key}"}, heartbeat=5
        )
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                if data.get("type") == "Results":
                    alt = data["channel"]["alternatives"][0]
                    text = alt.get("transcript", "")
                    if text and data.get("is_final"):
                        self._parts.append(text)
                    if text and self._on_partial is not None:
                        # Interim results are why interim_results=true is in the URL.
                        # Feed the settled text plus the in-flight hypothesis.
                        self._on_partial(" ".join(self._parts + ([] if data.get("is_final") else [text])).strip())
                    # speech_final is Deepgram's own endpoint; with endpointing=false
                    # we rely on the flush from finalise() instead.
                elif data.get("type") == "Metadata" and not self._final.done():
                    self._final.set_result(" ".join(self._parts).strip())
        except Exception as exc:  # noqa: BLE001 - surfaced by finalise()
            if not self._final.done():
                self._final.set_exception(exc)

    async def push(self, frame: bytes) -> None:
        assert self._ws is not None
        await self._ws.send_bytes(frame)

    async def finalise(self) -> str:
        assert self._ws is not None
        await self._ws.send_str(json.dumps({"type": "CloseStream"}))
        return await asyncio.wait_for(self._final, timeout=10)

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()


# --------------------------------------------------------------------------- LLM
class OpenAICompatLLM:
    """Any /v1/chat/completions endpoint with SSE streaming.

    VERIFY: https://platform.openai.com/docs/api-reference/chat/streaming
    Point BASE_URL at any compatible gateway; the shape is the same.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        system: str = "You are a delivery support agent. Reply in one short sentence.",
    ) -> None:
        self.key = api_key or os.environ["LLM_API_KEY"]
        self.base = (base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.system = system

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        body = {
            "model": self.model,
            "stream": True,
            "max_tokens": 60,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": prompt},
            ],
        }
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{self.base}/chat/completions",
                headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                json=body,
            ) as r:
                r.raise_for_status()
                async for raw in r.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"].get("content")
                    except (KeyError, IndexError, json.JSONDecodeError):
                        continue
                    if delta:
                        yield delta


# --------------------------------------------------------------------------- TTS
class ElevenLabsTTS:
    """Streaming input websocket.

    VERIFY: https://elevenlabs.io/docs/api-reference/websockets
    Use a flash/turbo model id here. A quality model will add 200 ms of TTFB and
    make the whole budget unreachable, which is a real finding but a slow way to
    discover it.
    """

    def __init__(self, api_key: str | None = None, voice_id: str | None = None, model: str | None = None) -> None:
        self.key = api_key or os.environ["ELEVENLABS_API_KEY"]
        self.voice = voice_id or os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.model = model or os.environ.get("ELEVENLABS_MODEL", "eleven_flash_v2_5")

    async def stream(self, text: AsyncIterator[str]) -> AsyncIterator[bytes]:
        url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice}/stream-input"
            f"?model_id={self.model}&output_format=pcm_16000&auto_mode=true"
        )
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(url, headers={"xi-api-key": self.key}, heartbeat=5) as ws:
                await ws.send_str(json.dumps({"text": " ", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}))

                async def pump() -> None:
                    async for chunk in text:
                        await ws.send_str(json.dumps({"text": chunk, "try_trigger_generation": True}))
                    await ws.send_str(json.dumps({"text": ""}))  # flush + close input

                task = asyncio.create_task(pump())
                try:
                    async for msg in ws:
                        if msg.type is not aiohttp.WSMsgType.TEXT:
                            continue
                        data = json.loads(msg.data)
                        if data.get("audio"):
                            import base64

                            yield base64.b64decode(data["audio"])
                        if data.get("isFinal"):
                            break
                finally:
                    task.cancel()


# --------------------------------------------------------------------------- VAD
class SileroVAD:
    """Neural VAD via the standalone silero-vad package.

    Still a silence-timeout endpointer on top of a good speech detector - which
    is the honest baseline. Beating this number is the turn-taking engine's job,
    and the gap between this and 220 ms is the size of that prize.
    """

    frame_ms = 32  # 512 samples at 16 kHz - silero's expected window

    def __init__(self, hang_ms: int = 500, threshold: float = 0.5) -> None:
        try:
            from silero_vad import load_silero_vad  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("pip install silero-vad torch, or run with --vad mock") from exc
        import torch  # type: ignore

        self._torch = torch
        self.model = load_silero_vad()
        self.hang_ms = hang_ms
        self.threshold = threshold
        self.silence_ms = 0
        self.fired = False

    def reset(self) -> None:
        self.silence_ms = 0
        self.fired = False
        self.model.reset_states()

    def push(self, frame: bytes) -> bool:
        if self.fired:
            return False
        import numpy as np  # type: ignore

        pcm = np.frombuffer(frame, dtype="<i2").astype("float32") / 32768.0
        if len(pcm) != 512:
            return False
        prob = float(self.model(self._torch.from_numpy(pcm), SAMPLE_RATE).item())
        if prob < self.threshold:
            self.silence_ms += self.frame_ms
        else:
            self.silence_ms = 0
        if self.silence_ms >= self.hang_ms:
            self.fired = True
            return True
        return False


def build(vad_kind: str = "silero", hang_ms: int = 500):
    from . import mock

    vad = SileroVAD(hang_ms=hang_ms) if vad_kind == "silero" else mock.MockVAD()
    return vad, DeepgramSTT(), OpenAICompatLLM(), ElevenLabsTTS()
