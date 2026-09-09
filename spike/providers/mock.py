"""Synthetic providers with configurable, realistic latency distributions.

These exist so the harness is verifiable in ten seconds with no API keys and
no network. If the numbers look wrong in mock mode, the bug is in the harness,
not in a vendor - which is a distinction worth being able to make quickly.

Latencies are lognormal: right-skewed, like real network-bound services, so the
p95 in the report behaves the way a real p95 does.
"""
from __future__ import annotations

import asyncio
import math
import random
from typing import AsyncIterator

FRAME_MS = 20
SAMPLE_RATE = 16_000


def _lognormal_ms(p50: float, spread: float, rng: random.Random) -> float:
    """p50 is the median; spread ~0.25 gives a p95 around 1.5x the median."""
    return max(1.0, p50 * math.exp(rng.gauss(0.0, spread)))


class MockVAD:
    """Fixed-timeout endpointer, i.e. the thing the turn-taking spec says to replace.

    `semantic=True` models a fused endpointer: it commits much sooner because it
    is not waiting out a silence window it does not need.
    """

    frame_ms = FRAME_MS

    def __init__(self, *, semantic: bool = False, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.semantic = semantic
        self.silence_ms = 0
        self.fired = False
        self._threshold = self._pick()

    def _pick(self) -> float:
        return _lognormal_ms(210 if self.semantic else 520, 0.10, self.rng)

    def reset(self) -> None:
        self.silence_ms = 0
        self.fired = False
        self._threshold = self._pick()

    def push(self, frame: bytes) -> bool:
        if self.fired:
            return False
        if _is_silent(frame):
            self.silence_ms += self.frame_ms
        else:
            self.silence_ms = 0
        if self.silence_ms >= self._threshold:
            self.fired = True
            return True
        return False


_SCRIPT = "my delivery did not arrive yesterday can you move it to thursday".split()


class MockSTT:
    """Emits progressive partials as frames arrive, like a real streaming ASR.

    Partials lag the audio by a couple of frames, which is what makes the
    semantic gate's job realistic - it is always reasoning about a transcript
    that is slightly behind the caller.
    """

    def __init__(self, *, p50_ms: float = 65, seed: int = 1, lag_frames: int = 3) -> None:
        self.rng = random.Random(seed)
        self.p50 = p50_ms
        self.frames = 0
        self.lag = lag_frames
        self._on_partial = None
        self._emitted = 0

    def set_partial_handler(self, fn) -> None:
        self._on_partial = fn

    async def open(self) -> None:
        await asyncio.sleep(0)

    async def push(self, frame: bytes) -> None:
        self.frames += 1
        if self._on_partial is None:
            return
        # roughly one word per 180 ms of audio, delayed by the lag
        due = max(0, (self.frames - self.lag) * FRAME_MS) // 180
        if due > self._emitted and due <= len(_SCRIPT):
            self._emitted = due
            self._on_partial(" ".join(_SCRIPT[:due]))

    async def finalise(self) -> str:
        await asyncio.sleep(_lognormal_ms(self.p50, 0.22, self.rng) / 1000.0)
        return "my delivery did not arrive yesterday can you move it to thursday"

    async def close(self) -> None:
        await asyncio.sleep(0)


class MockLLM:
    def __init__(self, *, ttft_ms: float = 290, tok_ms: float = 12, seed: int = 2) -> None:
        self.rng = random.Random(seed)
        self.ttft = ttft_ms
        self.tok = tok_ms

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        await asyncio.sleep(_lognormal_ms(self.ttft, 0.28, self.rng) / 1000.0)
        for word in ("Sorry", "about", "that.", "I", "can", "move", "it", "to", "Thursday."):
            yield word + " "
            await asyncio.sleep(self.tok / 1000.0)


class MockTTS:
    def __init__(self, *, ttfb_ms: float = 115, seed: int = 3) -> None:
        self.rng = random.Random(seed)
        self.ttfb = ttfb_ms

    async def stream(self, text: AsyncIterator[str]) -> AsyncIterator[bytes]:
        first = True
        async for _chunk in text:
            if first:
                await asyncio.sleep(_lognormal_ms(self.ttfb, 0.24, self.rng) / 1000.0)
                first = False
            yield b"\x00" * (SAMPLE_RATE // 50 * 2)


def _is_silent(frame: bytes, threshold: int = 500) -> bool:
    """Cheap RMS gate over 16-bit little-endian PCM."""
    if not frame:
        return True
    total = 0
    n = len(frame) // 2
    for i in range(0, n * 2, 2):
        s = int.from_bytes(frame[i : i + 2], "little", signed=True)
        total += s * s
    return (total / max(1, n)) ** 0.5 < threshold


def build(*, semantic_vad: bool = False, seed: int = 0):
    return (
        MockVAD(semantic=semantic_vad, seed=seed),
        MockSTT(seed=seed + 1),
        MockLLM(seed=seed + 2),
        MockTTS(seed=seed + 3),
    )
