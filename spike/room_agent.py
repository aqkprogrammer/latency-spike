"""Phase 2: the same measurement, inside a real LiveKit room.

Phase 1 tells you what the providers cost. This tells you what the transport,
the framework and the network add on top - which on a real PSTN call is
routinely another 100-200 ms nobody budgeted for.

Run it only after the bench numbers look sane. Debugging a room and a provider
at the same time is how a two-day spike becomes a two-week one.

    export LIVEKIT_URL=wss://<project>.livekit.cloud
    export LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
    python -m spike.room_agent dev            # then join the room from a browser

VERIFY: the livekit-agents API has moved more than once. This file targets the
1.x AgentSession surface. If import fails, check the installed version with
`pip show livekit-agents` and reconcile against
https://docs.livekit.io/agents/ - the measurement logic below is the part worth
keeping; the framework glue is the part that drifts.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from .clock import Timeline
from . import report

OUT = Path(os.environ.get("SPIKE_OUT", "room-runs.jsonl"))


class TurnRecorder:
    """One Timeline per turn, written to disk as it completes.

    Marks are driven by framework events rather than by our own polling, so the
    numbers include everything the framework does and not just the provider legs.
    """

    def __init__(self) -> None:
        self.current: Timeline | None = None
        self.turns: list[Timeline] = []

    def user_started(self) -> None:
        self.current = Timeline(label=f"turn-{len(self.turns) + 1}")

    def user_stopped(self) -> None:
        # Best available proxy for true end-of-speech inside a live room: the
        # framework's own end-of-utterance signal. It already contains some of
        # the endpointer's cost, so `endpoint` here reads lower than the bench
        # figure - compare bench-to-bench and room-to-room, never across.
        if self.current:
            self.current.mark("speech_end")

    def committed(self) -> None:
        if self.current:
            self.current.mark("vad_endpoint")

    def transcript(self, text: str) -> None:
        if self.current:
            self.current.mark("stt_final")
            self.current.meta["transcript"] = text[:120]

    def first_token(self) -> None:
        if self.current:
            self.current.mark("llm_first_token")

    def first_audio(self) -> None:
        if not self.current:
            return
        self.current.mark("tts_first_byte")
        self.turns.append(self.current)
        total = self.current.measured_total_ms()
        print(f"[spike] {self.current.label}: {total:,.0f} ms  {self.current.hops_ms()}", file=sys.stderr)
        with OUT.open("a") as fh:
            fh.write(json.dumps({
                "at": time.time(),
                "label": self.current.label,
                "hops_ms": {k: round(v, 1) for k, v in self.current.hops_ms().items()},
                "total_ms": round(total or 0, 1),
                "meta": self.current.meta,
            }) + "\n")
        self.current = None

    def summary(self) -> str:
        return report.render(self.turns, network_ms=0, colour=False)


def build_entrypoint():
    """Imported lazily so `python -m spike.bench` never needs livekit installed."""
    try:
        from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
        from livekit.plugins import deepgram, elevenlabs, openai, silero
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "LiveKit agent deps missing. Install:\n"
            "  pip install 'livekit-agents[deepgram,elevenlabs,openai,silero]'\n"
            f"({exc})"
        ) from exc

    rec = TurnRecorder()

    async def entrypoint(ctx: JobContext) -> None:
        await ctx.connect()

        session = AgentSession(
            vad=silero.VAD.load(min_silence_duration=0.5),
            stt=deepgram.STT(model="nova-3", language="multi"),
            llm=openai.LLM(model=os.environ.get("LLM_MODEL", "gpt-4o-mini")),
            tts=elevenlabs.TTS(model=os.environ.get("ELEVENLABS_MODEL", "eleven_flash_v2_5")),
        )

        # Event names are the part most likely to have moved. Each handler is
        # defensive so a renamed event degrades to an incomplete run rather than
        # a crashed worker - and report.render() will tell you which mark is missing.
        def _on(name: str, fn) -> None:
            try:
                session.on(name, fn)
            except Exception:  # noqa: BLE001
                print(f"[spike] event '{name}' not available on this version", file=sys.stderr)

        _on("user_started_speaking", lambda *_: rec.user_started())
        _on("user_stopped_speaking", lambda *_: rec.user_stopped())
        _on("user_input_transcribed", lambda ev, *_: (rec.committed(), rec.transcript(getattr(ev, "transcript", ""))))
        _on("agent_started_speaking", lambda *_: (rec.first_token(), rec.first_audio()))

        await session.start(
            agent=Agent(
                instructions=(
                    "You are a delivery support agent for a logistics company. "
                    "Answer in one short sentence. Never promise a date you were not given."
                )
            ),
            room=ctx.room,
        )

    return entrypoint, cli, WorkerOptions


def main() -> int:
    entrypoint, cli, WorkerOptions = build_entrypoint()
    print(f"[spike] per-turn results append to {OUT.resolve()}", file=sys.stderr)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
