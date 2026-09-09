"""WAV loading, ground-truth speech-end detection, and a synthetic fallback.

The single most important thing in this file is `speech_end_sample`. Every
latency number in the report is measured from the real end of the caller's
speech, not from the end of the file - if you measure from end-of-file you are
measuring your own trailing silence and the numbers will look wonderful.
"""
from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16_000


@dataclass
class Clip:
    pcm: bytes            # 16-bit LE mono at SAMPLE_RATE
    sample_rate: int
    speech_end_sample: int
    name: str

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate

    @property
    def speech_end_s(self) -> float:
        return self.speech_end_sample / self.sample_rate

    def frames(self, frame_ms: int):
        step = int(self.sample_rate * frame_ms / 1000) * 2
        for i in range(0, len(self.pcm) - step + 1, step):
            yield self.pcm[i : i + step]


def load_wav(path: str | Path) -> Clip:
    p = Path(path)
    with wave.open(str(p), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{p.name}: need 16-bit mono WAV. Convert with:\n"
                             f"  ffmpeg -i {p.name} -ac 1 -ar 16000 -sample_fmt s16 out.wav")
        if w.getframerate() != SAMPLE_RATE:
            raise SystemExit(f"{p.name}: need {SAMPLE_RATE} Hz, got {w.getframerate()}. "
                             f"ffmpeg -i {p.name} -ar 16000 out.wav")
        pcm = w.readframes(w.getnframes())
    return Clip(pcm, SAMPLE_RATE, speech_end_sample(pcm), p.stem)


def speech_end_sample(pcm: bytes, win_ms: int = 20, floor_db: float = -42.0) -> int:
    """Last window whose RMS is above an absolute floor.

    Absolute rather than relative to peak: a clip with one loud word and quiet
    trailing speech would otherwise have its tail trimmed as silence.
    """
    step = int(SAMPLE_RATE * win_ms / 1000)
    thresh = 32768.0 * (10.0 ** (floor_db / 20.0))
    last = 0
    total = len(pcm) // 2
    for start in range(0, total - step + 1, step):
        chunk = pcm[start * 2 : (start + step) * 2]
        acc = 0
        for i in range(0, len(chunk), 2):
            s = struct.unpack_from("<h", chunk, i)[0]
            acc += s * s
        if math.sqrt(acc / step) > thresh:
            last = start + step
    return last or total


def synthetic(speech_s: float = 2.6, trailing_silence_s: float = 1.4) -> Clip:
    """A tone burst plus trailing silence. Enough to exercise VAD and pacing
    end to end without needing a recording, and the ground truth is exact."""
    n_speech = int(SAMPLE_RATE * speech_s)
    n_sil = int(SAMPLE_RATE * trailing_silence_s)
    out = bytearray()
    for i in range(n_speech):
        env = 0.5 + 0.5 * math.sin(2 * math.pi * 3.1 * i / SAMPLE_RATE)   # syllable-ish
        v = int(9000 * env * math.sin(2 * math.pi * 165 * i / SAMPLE_RATE))
        out += struct.pack("<h", v)
    out += b"\x00\x00" * n_sil
    return Clip(bytes(out), SAMPLE_RATE, n_speech, "synthetic")


def write_wav(path: str | Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
