"""Fused endpointer: acoustic + prosodic + semantic, per the turn-taking spec.

The architecture that matters is not the weighted sum - it is that the semantic
signal moves a *threshold* rather than casting a vote. A veto pushes the
threshold to the ceiling and the endpointer simply waits; an accelerator pulls
it down to the short-answer floor. Every decision is therefore explainable in
one line, which is the difference between a component you can tune and one you
can only replace.

Two behaviours are not tuning parameters and should not be changed:
  * a dangling continuation vetoes the endpoint no matter how long the silence
  * when the signals disagree, wait rather than cut
Cutting a caller off costs far more than a moment of hesitation.
"""
from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from enum import Enum

SAMPLE_RATE = 16_000


class Context(str, Enum):
    """What the dialogue layer expects next. Sets the base threshold."""

    YES_NO = "yes_no"
    SELECTION = "selection"
    OPEN = "open"
    DIGITS = "digits"
    ADDRESS = "address"
    SPELLING = "spelling"


BASE_MS: dict[Context, int] = {
    Context.YES_NO: 140,
    Context.SELECTION: 220,
    Context.OPEN: 380,
    Context.DIGITS: 900,
    Context.ADDRESS: 1100,
    Context.SPELLING: 1400,
}

MIN_MS = 120        # never fire faster than this, whatever the signals say
MAX_HOLD_MS = 4000  # ceiling; past this, hand over rather than keep waiting


class Semantic(str, Enum):
    VETO = "veto"              # clause is unfinished - do not endpoint
    ACCELERATE = "accelerate"  # complete short answer - endpoint early
    NEUTRAL = "neutral"


# --------------------------------------------------------------------- lexicons
# English is SVO, so a trailing auxiliary ("it is", "I was") is mid-clause and
# dangles. Hindi is verb-final, so a trailing auxiliary ("kahan hai") is the
# normal way a sentence *ends*. Getting this backwards is the single most common
# reason an English-tuned endpointer cuts Hindi speakers off mid-thought, and it
# is why these two lists cannot be merged.
EN_DANGLERS = {
    # prepositions
    "to", "for", "at", "on", "in", "into", "with", "from", "by", "about", "of",
    "over", "under", "till", "until", "near", "towards", "onto", "upon",
    # conjunctions / subordinators
    "and", "or", "but", "so", "because", "if", "when", "while", "that", "which",
    "than", "though", "unless", "since", "whether",
    # determiners and possessives
    "the", "a", "an", "my", "your", "his", "her", "our", "their", "this",
    "these", "those", "some", "any", "every",
    # auxiliaries and light verbs (mid-clause in English)
    "is", "was", "are", "were", "be", "been", "am", "will", "would", "can",
    "could", "should", "shall", "may", "might", "must", "have", "has", "had",
    "do", "does", "did", "want", "need", "going", "trying", "let",
    # contractions that cannot end a clause
    "i'm", "it's", "there's", "let's", "we're", "you're", "don't", "can't",
}

HI_DANGLERS = {
    # postpositions - the real danglers in a verb-final language
    "ka", "ki", "ke", "ko", "se", "me", "mein", "par", "pe", "tak", "liye",
    "saath", "wala", "wale", "wali",
    # conjunctions
    "aur", "ya", "lekin", "magar", "kyunki", "agar", "jab", "phir", "toh", "to",
    # possessives / determiners
    "mera", "meri", "mere", "aapka", "aapki", "aapke", "apna", "apni", "uska",
    "iska", "unka", "inka", "tumhara", "kuch", "koi",
    # pronoun + postposition. A nominal carrying a postposition is waiting for
    # its verb, so these dangle exactly as the bare postpositions do. Found by
    # the hinglish_dangling case cutting a caller off on "usko".
    "usko", "isko", "unko", "inko", "uske", "iske", "unke", "inke",
    "usse", "isse", "mujhe", "tujhe", "tumhe", "humein", "hume", "aapko",
}

# Deliberately NOT danglers in Hindi: hai, hain, tha, thi, the, hoga, chahiye,
# diya, gaya, aaya - these are how Hindi clauses finish.

FILLERS = {
    "um", "uh", "er", "erm", "hmm", "mmm", "like", "actually", "basically",
    "matlab", "yaani", "arre", "woh", "bas", "aisa",
}

ACCELERATORS = {
    "yes", "yeah", "yep", "yup", "no", "nope", "ok", "okay", "sure", "correct",
    "right", "exactly", "please", "thanks",
    "haan", "han", "ha", "nahi", "nahin", "na", "bilkul", "ji", "achha", "acha",
    "theek", "thik", "sahi",
}

_WORD_NUMS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5, "chhe": 6, "saat": 7,
    "aath": 8, "nau": 9, "shunya": 0,
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _looks_hindi(tokens: list[str]) -> bool:
    """Matrix-language sniff, not a detector.

    Counts Hindi *function* words only. Content-word loans ("order", "delivery")
    are exactly what a token-counting detector gets wrong, so they are ignored -
    see the code-switching section of the turn-taking spec.
    """
    marks = HI_DANGLERS | {"hai", "hain", "tha", "thi", "the", "nahi", "kya",
                           "kahan", "kab", "chahiye", "karo", "kijiye", "dijiye"}
    return sum(1 for t in tokens if t in marks) >= 2


def count_digits(text: str) -> int:
    n = 0
    for t in _tokens(text):
        if t.isdigit():
            n += len(t)
        elif t in _WORD_NUMS:
            n += 1
    return n


@dataclass
class SemanticGate:
    """Classifies a partial transcript. Fast, deterministic, reviewable."""

    context: Context = Context.OPEN
    expect_digits: int = 10

    def classify(self, partial: str) -> tuple[Semantic, str]:
        toks = _tokens(partial)
        if not toks:
            return Semantic.VETO, "no transcript yet"

        last = toks[-1]

        if last in FILLERS:
            return Semantic.VETO, f"trailing filler '{last}'"

        hindi = _looks_hindi(toks)
        danglers = HI_DANGLERS if hindi else EN_DANGLERS
        if last in danglers:
            lang = "hi" if hindi else "en"
            return Semantic.VETO, f"dangling {lang} function word '{last}'"

        if self.context is Context.DIGITS:
            got = count_digits(partial)
            if got < self.expect_digits:
                return Semantic.VETO, f"{got}/{self.expect_digits} digits"
            return Semantic.ACCELERATE, f"{got} digits complete"

        if self.context in (Context.YES_NO, Context.SELECTION) and len(toks) <= 3:
            if any(t in ACCELERATORS for t in toks):
                return Semantic.ACCELERATE, f"short answer '{' '.join(toks)}'"

        if len(toks) == 1 and last in ACCELERATORS:
            return Semantic.ACCELERATE, f"one-word answer '{last}'"

        return Semantic.NEUTRAL, "clause looks complete"


class Prosody:
    """Energy-decay completion cue.

    The cheap half of the prosodic signal. A falling terminal contour is the
    strong cue and needs a pitch tracker; decaying energy correlates with it
    well enough to be worth 15% of a threshold and costs nothing. Upgrade here
    first if you want the next 40 ms.
    """

    def __init__(self, tail_ms: int = 250) -> None:
        self.tail_ms = tail_ms
        self.frames: list[float] = []

    def reset(self) -> None:
        self.frames.clear()

    def push(self, frame: bytes, frame_ms: int) -> None:
        self.frames.append(_rms(frame))
        keep = max(8, int(3000 / max(1, frame_ms)))
        if len(self.frames) > keep:
            del self.frames[: len(self.frames) - keep]

    def falling(self, frame_ms: int) -> float:
        """0.0 flat or rising, 1.0 clearly decaying."""
        voiced = [f for f in self.frames if f > 300]
        if len(voiced) < 6:
            return 0.0
        n_tail = max(2, int(self.tail_ms / max(1, frame_ms)))
        tail = voiced[-n_tail:]
        body = voiced[:-n_tail] or voiced
        mean_tail = sum(tail) / len(tail)
        mean_body = sum(body) / len(body)
        if mean_body <= 0:
            return 0.0
        ratio = mean_tail / mean_body
        return max(0.0, min(1.0, (1.0 - ratio) / 0.5))


@dataclass
class EndpointDecision:
    fired_at_ms: float
    silence_ms: float
    threshold_ms: float
    semantic: Semantic
    reason: str
    prosody: float
    partial: str


@dataclass
class FusedEndpointer:
    """Drop-in for the VAD protocol, plus `update_partial`.

    `push` returns True on the frame at which the endpointer commits. The
    decision, and every rejected candidate, is recorded on `trace` - which is
    what makes a bad cut debuggable after the fact instead of mysterious.
    """

    context: Context = Context.OPEN
    frame_ms: int = 20
    expect_digits: int = 10
    silence_rms: int = 500
    prosody_weight: float = 0.15

    gate: SemanticGate = field(init=False)
    prosody: Prosody = field(default_factory=Prosody)
    partial: str = ""
    saw_partial: bool = False
    silence_ms: float = 0.0
    elapsed_ms: float = 0.0
    speech_seen: bool = False
    fired: bool = False
    decision: EndpointDecision | None = None
    trace: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.gate = SemanticGate(context=self.context, expect_digits=self.expect_digits)

    def reset(self) -> None:
        self.prosody.reset()
        self.partial = ""
        self.saw_partial = False
        self.silence_ms = 0.0
        self.elapsed_ms = 0.0
        self.speech_seen = False
        self.fired = False
        self.decision = None
        self.trace.clear()

    def update_partial(self, text: str) -> None:
        if text:
            self.partial = text
            self.saw_partial = True

    def threshold_ms(self) -> tuple[float, Semantic, str]:
        base = float(BASE_MS[self.context])

        # Degraded mode. No partial has ever arrived, so there is no semantic
        # signal to have an opinion - recognition is down, or nothing is wired
        # to update_partial. Fall back to acoustic and prosody rather than
        # vetoing forever: an empty transcript that is *coming* is evidence to
        # wait for, and one that is never coming is a dead call. Found by
        # running the clip evaluator on a recording with no transcript, where
        # this held to the 4 s ceiling on every utterance.
        if not self.saw_partial:
            bonus = self.prosody.falling(self.frame_ms) * self.prosody_weight
            return max(float(MIN_MS), base * (1.0 - bonus)), Semantic.NEUTRAL, "semantic unavailable"

        sem, why = self.gate.classify(self.partial)

        if sem is Semantic.VETO:
            return float(MAX_HOLD_MS), sem, why
        if sem is Semantic.ACCELERATE:
            return float(max(MIN_MS, BASE_MS[Context.YES_NO])), sem, why

        # Neutral: let a falling contour shave the threshold, bounded.
        bonus = self.prosody.falling(self.frame_ms) * self.prosody_weight
        return max(float(MIN_MS), base * (1.0 - bonus)), sem, why

    def push(self, frame: bytes) -> bool:
        if self.fired:
            return False
        self.elapsed_ms += self.frame_ms
        self.prosody.push(frame, self.frame_ms)

        if _rms(frame) < self.silence_rms:
            self.silence_ms += self.frame_ms
        else:
            self.speech_seen = True
            if self.silence_ms:
                self.trace.append(f"{self.elapsed_ms:.0f}ms speech resumed after {self.silence_ms:.0f}ms silence")
            self.silence_ms = 0.0
            return False

        if not self.speech_seen:
            return False  # leading silence is not a turn ending

        thr, sem, why = self.threshold_ms()

        if self.silence_ms >= MAX_HOLD_MS:
            self._fire(MAX_HOLD_MS, sem, f"ceiling reached ({why})")
            return True
        if self.silence_ms >= thr:
            self._fire(thr, sem, why)
            return True
        return False

    def _fire(self, thr: float, sem: Semantic, why: str) -> None:
        self.fired = True
        self.decision = EndpointDecision(
            fired_at_ms=self.elapsed_ms,
            silence_ms=self.silence_ms,
            threshold_ms=thr,
            semantic=sem,
            reason=why,
            prosody=round(self.prosody.falling(self.frame_ms), 2),
            partial=self.partial,
        )
        self.trace.append(
            f"{self.elapsed_ms:.0f}ms FIRE  silence={self.silence_ms:.0f} thr={thr:.0f} "
            f"{sem.value} ({why})"
        )


class FixedEndpointer:
    """The baseline every stack ships with. Here so the comparison is honest."""

    def __init__(self, hang_ms: int = 500, frame_ms: int = 20, silence_rms: int = 500) -> None:
        self.hang_ms = hang_ms
        self.frame_ms = frame_ms
        self.silence_rms = silence_rms
        self.silence_ms = 0.0
        self.elapsed_ms = 0.0
        self.speech_seen = False
        self.fired = False
        self.decision: EndpointDecision | None = None
        self.trace: list[str] = []

    def reset(self) -> None:
        self.silence_ms = 0.0
        self.elapsed_ms = 0.0
        self.speech_seen = False
        self.fired = False
        self.decision = None
        self.trace.clear()

    def update_partial(self, text: str) -> None:
        return  # a fixed timeout cannot use one, which is the point

    def push(self, frame: bytes) -> bool:
        if self.fired:
            return False
        self.elapsed_ms += self.frame_ms
        if _rms(frame) < self.silence_rms:
            self.silence_ms += self.frame_ms
        else:
            self.speech_seen = True
            self.silence_ms = 0.0
            return False
        if self.speech_seen and self.silence_ms >= self.hang_ms:
            self.fired = True
            self.decision = EndpointDecision(
                self.elapsed_ms, self.silence_ms, float(self.hang_ms),
                Semantic.NEUTRAL, f"fixed {self.hang_ms}ms timeout", 0.0, "",
            )
            self.trace.append(f"{self.elapsed_ms:.0f}ms FIRE  fixed timeout")
            return True
        return False


def _rms(frame: bytes) -> float:
    n = len(frame) // 2
    if n == 0:
        return 0.0
    acc = 0
    for i in range(0, n * 2, 2):
        s = struct.unpack_from("<h", frame, i)[0]
        acc += s * s
    return math.sqrt(acc / n)
