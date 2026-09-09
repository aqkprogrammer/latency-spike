"""Provider protocols.

Deliberately tiny. Everything the harness needs from a vendor is four methods,
which is what makes swapping one a twenty-line job rather than a refactor.
The measurement core never imports a vendor SDK.
"""
from __future__ import annotations

from typing import AsyncIterator, Callable, Protocol, runtime_checkable


@runtime_checkable
class VAD(Protocol):
    """Consumes 16 kHz mono PCM frames, reports when speech has ended."""

    frame_ms: int

    def reset(self) -> None: ...

    def push(self, frame: bytes) -> bool:
        """Return True on the frame at which the endpointer commits to 'done'."""
        ...


@runtime_checkable
class STT(Protocol):
    """Streaming recogniser.

    Partials are not a nicety. The semantic half of the fused endpointer reads
    them, so a recogniser that only yields a final transcript caps you at
    acoustic endpointing and the 300 ms it costs.
    """

    def set_partial_handler(self, fn: Callable[[str], None]) -> None:
        """Called with the transcript so far, as often as the vendor emits one."""
        ...

    async def open(self) -> None: ...
    async def push(self, frame: bytes) -> None: ...
    async def finalise(self) -> str:
        """Signal end of audio and resolve when the final transcript is in hand."""
        ...
    async def close(self) -> None: ...


@runtime_checkable
class LLM(Protocol):
    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield text deltas. The first yield is what TTFT measures."""
        ...


@runtime_checkable
class TTS(Protocol):
    async def stream(self, text: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Yield audio chunks. The first yield is what TTFB measures."""
        ...


Providers = tuple[VAD, STT, LLM, TTS]
Factory = Callable[[], Providers]
