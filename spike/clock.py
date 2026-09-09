"""Monotonic timeline for a single conversational turn.

One Timeline per turn. Marks are absolute monotonic seconds; every derived
figure is a difference between two marks, so clock drift and wall-clock jumps
cannot corrupt a measurement.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


# The hops we care about, in pipeline order, with the budget from the plan.
# Anything not measured here is not on the critical path by definition.
BUDGET_MS: dict[str, int] = {
    "endpoint": 220,   # true end of speech -> VAD/endpointer says "they're done"
    "stt_final": 70,   # endpoint -> final transcript in hand
    "llm_ttft": 300,   # final transcript -> first token out of the model
    "tts_ttfb": 120,   # first token -> first byte of audio
    "network": 80,     # egress + jitter buffer; assumed, not measured offline
}
TARGET_TOTAL_MS = 850

# Ordered mark names. bench.py records these; report.py turns them into hops.
MARKS = ("speech_end", "vad_endpoint", "stt_final", "llm_first_token", "tts_first_byte")

HOPS = (
    ("endpoint", "speech_end", "vad_endpoint"),
    ("stt_final", "vad_endpoint", "stt_final"),
    ("llm_ttft", "stt_final", "llm_first_token"),
    ("tts_ttfb", "llm_first_token", "tts_first_byte"),
)


@dataclass
class Timeline:
    """Marks for one turn. Missing marks make the run incomplete, never wrong."""

    label: str = ""
    marks: dict[str, float] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)

    def mark(self, name: str, at: float | None = None) -> float:
        """Record a mark. First write wins - a hop can only happen once."""
        if name not in self.marks:
            self.marks[name] = time.monotonic() if at is None else at
        return self.marks[name]

    @property
    def complete(self) -> bool:
        return all(m in self.marks for m in MARKS)

    def missing(self) -> list[str]:
        return [m for m in MARKS if m not in self.marks]

    def hop_ms(self, name: str) -> float | None:
        for hop, start, end in HOPS:
            if hop != name:
                continue
            if start in self.marks and end in self.marks:
                # Clamp: a provider can answer before the endpointer commits.
                # That is real and useful, but it is not negative latency.
                return max(0.0, (self.marks[end] - self.marks[start]) * 1000.0)
        return None

    def hops_ms(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for hop, _, _ in HOPS:
            v = self.hop_ms(hop)
            if v is not None:
                out[hop] = v
        return out

    def measured_total_ms(self) -> float | None:
        """speech_end -> first audio byte. The only number that matters."""
        if "speech_end" in self.marks and "tts_first_byte" in self.marks:
            return (self.marks["tts_first_byte"] - self.marks["speech_end"]) * 1000.0
        return None


class Paced:
    """Realtime pacing for audio playback, without accumulated drift.

    Sleeping 20 ms per 20 ms frame drifts badly over a 6-second utterance and
    turns a latency measurement into a throughput measurement. Sleep to an
    absolute deadline instead.
    """

    def __init__(self, frame_seconds: float) -> None:
        self.frame_seconds = frame_seconds
        self.start = time.monotonic()
        self.n = 0

    def deadline(self) -> float:
        return self.start + self.n * self.frame_seconds

    def advance(self) -> float:
        self.n += 1
        return self.deadline()

    def elapsed_at_frame(self, index: int) -> float:
        """Wall time at which frame `index` was scheduled to play."""
        return self.start + index * self.frame_seconds
