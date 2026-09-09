"""Run both endpointers against a real recording.

Ground truth for "the caller finished here" is the last speech-bearing window
in the file, from spike.audio. Any fire before that is a false cut, by
definition and without needing a human to annotate anything.

The semantic gate needs a transcript arriving progressively. Three sources, in
descending order of fidelity:

  --words caller.words.json   word timings; the honest offline option
  --text  caller.txt          plain transcript, words spread across voiced audio
  (nothing)                   acoustic + prosody only

Without partials the fused endpointer is running on one of its three signals,
so the comparison understates it. The report says so rather than quietly
banking the flattering number.

Generate the sidecar once with spike.transcribe, then iterate on the endpointer
a hundred times for free.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from . import audio as au
from .endpoint import Context, FixedEndpointer, FusedEndpointer

FRAME_MS = 20
GAP_MS = 250  # silence gaps at least this long are worth reporting


@dataclass
class Fire:
    at_ms: float | None
    reason: str
    trace: list


def load_partials(clip: au.Clip, words_path: str | None, text_path: str | None):
    """Return a list of (at_ms, transcript_so_far), or None."""
    if words_path:
        with open(words_path) as fh:
            words = json.load(fh)
        out, acc = [], []
        for w in words:
            acc.append(str(w["w"]))
            out.append((float(w["t"]), " ".join(acc)))
        return out

    if text_path:
        with open(text_path) as fh:
            toks = fh.read().split()
        if not toks:
            return None
        # Spread words across voiced frames only. Crude, and much better than
        # spreading across the whole clip - trailing silence would otherwise
        # delay every partial and flatter the endpointer.
        voiced = [i for i, f in enumerate(clip.frames(FRAME_MS)) if _voiced(f)]
        if not voiced:
            return None
        out, acc = [], []
        for i, tok in enumerate(toks):
            frame = voiced[min(len(voiced) - 1, int(i / len(toks) * len(voiced)))]
            acc.append(tok)
            out.append(((frame + 1) * FRAME_MS, " ".join(acc)))
        return out

    return None


def _voiced(frame: bytes, threshold: int = 500) -> bool:
    from .endpoint import _rms
    return _rms(frame) >= threshold


def partial_at(partials, t_ms: float) -> str:
    text = ""
    if not partials:
        return text
    for at, p in partials:
        if at <= t_ms:
            text = p
    return text


def run(clip: au.Clip, ep, partials) -> Fire:
    ep.reset()
    t = 0.0
    for frame in clip.frames(FRAME_MS):
        if partials is not None and hasattr(ep, "update_partial"):
            ep.update_partial(partial_at(partials, t))
        if ep.push(frame):
            return Fire(t + FRAME_MS, ep.decision.reason if ep.decision else "", list(ep.trace))
        t += FRAME_MS
    return Fire(None, "never fired", list(ep.trace))


def gaps(clip: au.Clip) -> list[tuple[float, float]]:
    """Silence runs of at least GAP_MS between two speech-bearing frames."""
    frames = list(clip.frames(FRAME_MS))
    out, run_start, seen_speech = [], None, False
    for i, f in enumerate(frames):
        t = i * FRAME_MS
        if _voiced(f):
            if run_start is not None and seen_speech and (t - run_start) >= GAP_MS:
                out.append((run_start, t))
            run_start, seen_speech = None, True
        elif run_start is None:
            run_start = t
    return out
