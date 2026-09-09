"""Endpointing test cases: synthetic audio with known ground truth.

Each case is a set of speech segments, a schedule of STT partials, and the
answer key - where the endpointer is allowed to fire and where firing is a cut.
These are the five situations from the turn-taking spec, and they are the
scenarios this component should be judged on for the life of the project.

Real recordings are better and you should add them. Synthetic clips are here so
the comparison runs today with exact ground truth, which recordings never have.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

from .endpoint import Context

SAMPLE_RATE = 16_000


@dataclass
class Case:
    name: str
    context: Context
    segments: list[tuple[int, int]]          # (start_ms, end_ms) of speech
    partials: list[tuple[int, str]]          # (at_ms, transcript so far)
    tail_ms: int = 1600                      # trailing silence after last segment
    expect_digits: int = 10
    note: str = ""

    @property
    def true_end_ms(self) -> int:
        return self.segments[-1][1]

    @property
    def total_ms(self) -> int:
        return self.true_end_ms + self.tail_ms

    def partial_at(self, t_ms: float) -> str:
        text = ""
        for at, p in self.partials:
            if at <= t_ms:
                text = p
        return text

    def pcm(self) -> bytes:
        n = int(SAMPLE_RATE * self.total_ms / 1000)
        buf = bytearray(n * 2)
        for start_ms, end_ms in self.segments:
            a = int(SAMPLE_RATE * start_ms / 1000)
            b = int(SAMPLE_RATE * end_ms / 1000)
            dur = max(1, b - a)
            for i in range(a, min(b, n)):
                k = i - a
                # syllable envelope, plus a decay over the final 250 ms so the
                # prosody cue has something real to read
                env = 0.55 + 0.45 * math.sin(2 * math.pi * 3.4 * k / SAMPLE_RATE)
                remaining = dur - k
                if remaining < SAMPLE_RATE * 0.25:
                    env *= 0.35 + 0.65 * (remaining / (SAMPLE_RATE * 0.25))
                v = int(9000 * env * math.sin(2 * math.pi * 168 * k / SAMPLE_RATE))
                struct.pack_into("<h", buf, i * 2, max(-32768, min(32767, v)))
        return bytes(buf)

    def frames(self, frame_ms: int):
        pcm = self.pcm()
        step = int(SAMPLE_RATE * frame_ms / 1000) * 2
        for i in range(0, len(pcm) - step + 1, step):
            yield pcm[i : i + step]


CASES: list[Case] = [
    Case(
        name="complete",
        context=Context.OPEN,
        segments=[(0, 2600)],
        partials=[(600, "can you"), (1400, "can you move my"), (2300, "can you move my delivery"),
                  (2600, "can you move my delivery to thursday")],
        note="Ordinary finished sentence. Both endpointers should be correct; only speed differs.",
    ),
    Case(
        name="dangling",
        context=Context.OPEN,
        segments=[(0, 1400), (2100, 2700)],
        partials=[(700, "i want to move"), (1400, "i want to move it to"),
                  (2400, "i want to move it to thursday"), (2700, "i want to move it to thursday")],
        note="Thinking pause after a preposition. Firing in the 700 ms gap is a cut.",
    ),
    Case(
        name="digits",
        context=Context.DIGITS,
        segments=[(0, 900), (1250, 1600), (2000, 2400)],
        partials=[(900, "nine eight seven"), (1600, "nine eight seven six five"),
                  (2400, "nine eight seven six five four three two one zero")],
        note="Phone number in groups. Firing before the tenth digit is a cut.",
    ),
    Case(
        name="short_answer",
        context=Context.YES_NO,
        segments=[(0, 350)],
        partials=[(350, "haan")],
        tail_ms=1200,
        note="One-word answer to a closed question. The common turn; every ms here repeats all call.",
    ),
    Case(
        name="hinglish_dangling",
        context=Context.OPEN,
        segments=[(0, 1500), (2200, 3000)],
        partials=[(800, "mera order abhi tak"), (1500, "mera order abhi tak nahi aaya usko"),
                  (2700, "mera order abhi tak nahi aaya usko kal bhej dijiye"),
                  (3000, "mera order abhi tak nahi aaya usko kal bhej dijiye")],
        note="Hindi matrix language with English loan nouns, pausing on a postposition.",
    ),
    Case(
        name="hinglish_complete",
        context=Context.OPEN,
        segments=[(0, 1800)],
        partials=[(900, "mera order kahan"), (1800, "mera order kahan hai")],
        note="Ends on 'hai'. An English-tuned lexicon treats a trailing auxiliary as dangling and waits "
             "for speech that never comes - the failure this case exists to catch.",
    ),
]


def by_name(name: str) -> Case:
    for c in CASES:
        if c.name == name:
            return c
    raise SystemExit(f"unknown case '{name}'. try: {', '.join(c.name for c in CASES)}")
