"""Render the waterfall from a room-runs.jsonl produced by room_agent.py.

    python -m spike.summarise_room room-runs.jsonl
"""
from __future__ import annotations

import json
import sys

from .clock import Timeline
from . import report


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "room-runs.jsonl"
    runs: list[Timeline] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tl = Timeline(label=row.get("label", ""), meta=row.get("meta", {}))
            # Rebuild marks from hop durations so report.py can treat these
            # exactly like live runs.
            t = 0.0
            tl.marks["speech_end"] = 0.0
            for hop, end in (("endpoint", "vad_endpoint"), ("stt_final", "stt_final"),
                             ("llm_ttft", "llm_first_token"), ("tts_ttfb", "tts_first_byte")):
                ms = row.get("hops_ms", {}).get(hop)
                if ms is None:
                    break
                t += ms / 1000.0
                tl.marks[end] = t
            runs.append(tl)

    print(report.render(runs, network_ms=0, colour=sys.stdout.isatty()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
