"""One-shot: WAV -> word timings, so the endpointer can be evaluated offline
a hundred times without paying for STT each run.

    python -m spike.transcribe audio/caller.wav

Writes audio/caller.words.json alongside the audio.
VERIFY: https://developers.deepgram.com/docs/pre-recorded-audio
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

URL = ("https://api.deepgram.com/v1/listen"
       "?model=nova-3&language=multi&punctuate=false&smart_format=false")


async def main_async(path: Path) -> int:
    try:
        import aiohttp
    except ImportError:
        raise SystemExit("pip install aiohttp")

    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise SystemExit("DEEPGRAM_API_KEY not set. See .env.example")

    data = path.read_bytes()
    async with aiohttp.ClientSession() as s:
        async with s.post(URL, data=data, headers={
            "Authorization": f"Token {key}",
            "Content-Type": "audio/wav",
        }) as r:
            if r.status != 200:
                raise SystemExit(f"deepgram {r.status}: {(await r.text())[:300]}")
            body = await r.json()

    alt = body["results"]["channels"][0]["alternatives"][0]
    words = [{"w": w["word"], "t": round(float(w["end"]) * 1000)} for w in alt.get("words", [])]
    if not words:
        raise SystemExit("no words returned — check the audio is 16-bit mono 16 kHz speech")

    out = path.with_suffix(".words.json")
    out.write_text(json.dumps(words, indent=1))
    print(f"{len(words)} words -> {out}")
    print(f"transcript: {alt.get('transcript','')[:160]}")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m spike.transcribe audio/caller.wav")
    return asyncio.run(main_async(Path(sys.argv[1])))


if __name__ == "__main__":
    raise SystemExit(main())
