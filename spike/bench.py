"""Phase 1: the offline component benchmark.

Feeds a WAV through VAD -> STT -> LLM -> TTS at realtime pace and records where
every millisecond went between the caller finishing and the first byte of audio
leaving. No LiveKit, no room, no SIP - so it runs today, on your laptop, with
nothing but API keys.

    python -m spike.bench --mock -n 7            # verify the harness, no keys
    python -m spike.bench --mock --semantic-vad  # what fused endpointing buys
    python -m spike.bench --real -n 7 --wav audio/caller.wav
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import AsyncIterator

from . import audio as au
from . import report
from .clock import Paced, Timeline


async def run_once(clip: au.Clip, providers, label: str) -> Timeline:
    vad, stt, llm, tts = providers
    tl = Timeline(label=label, meta={"clip": clip.name, "clip_s": round(clip.duration_s, 2)})
    vad.reset()

    # The fused endpointer reads the transcript as it arrives; a fixed timeout
    # ignores this and the no-op keeps both paths on one code path.
    if hasattr(stt, "set_partial_handler") and hasattr(vad, "update_partial"):
        stt.set_partial_handler(vad.update_partial)

    await stt.open()
    try:
        pace = Paced(vad.frame_ms / 1000.0)
        speech_end_wall = pace.start + clip.speech_end_s
        endpoint_seen = False

        # --- stream the clip at realtime pace ----------------------------------
        for idx, frame in enumerate(clip.frames(vad.frame_ms)):
            target = pace.elapsed_at_frame(idx)
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

            await stt.push(frame)

            if not endpoint_seen and vad.push(frame):
                endpoint_seen = True
                tl.mark("speech_end", at=speech_end_wall)
                tl.mark("vad_endpoint")
                break

        if not endpoint_seen:
            # VAD never committed inside the clip. Real condition, not a crash:
            # usually a hang time longer than the trailing silence in the file.
            tl.mark("speech_end", at=speech_end_wall)
            tl.meta["vad"] = "no endpoint within clip"
            await stt.close()
            return tl

        # --- transcript --------------------------------------------------------
        transcript = await stt.finalise()
        tl.mark("stt_final")
        tl.meta["transcript"] = transcript[:120]
        if getattr(vad, "decision", None) is not None:
            d = vad.decision
            tl.meta["endpoint_reason"] = f"{d.semantic.value}: {d.reason}"
            tl.meta["endpoint_threshold_ms"] = round(d.threshold_ms)

        # --- model, first token ------------------------------------------------
        async def deltas() -> AsyncIterator[str]:
            first = True
            async for piece in llm.stream(transcript):
                if first:
                    tl.mark("llm_first_token")
                    first = False
                yield piece

        # --- synthesis, first byte ---------------------------------------------
        async for _audio in tts.stream(deltas()):
            tl.mark("tts_first_byte")
            break  # TTFB is all we need; stop before paying for the whole utterance

    finally:
        await stt.close()

    return tl


async def main_async(args: argparse.Namespace) -> int:
    clip = au.load_wav(args.wav) if args.wav else au.synthetic()
    print(f"clip: {clip.name}  {clip.duration_s:.2f}s  speech ends at {clip.speech_end_s:.2f}s", file=sys.stderr)

    runs: list[Timeline] = []
    for i in range(args.n):
        if args.mock:
            from .providers import mock

            providers = mock.build(semantic_vad=args.semantic_vad, seed=i * 17)
        else:
            from .providers import real

            providers = real.build(vad_kind=args.vad, hang_ms=args.hang_ms)

        if args.endpointer == "fused":
            from .endpoint import Context, FusedEndpointer

            providers = (
                FusedEndpointer(context=Context(args.context), frame_ms=20),
                providers[1], providers[2], providers[3],
            )
        elif args.endpointer == "fixed":
            from .endpoint import FixedEndpointer

            providers = (
                FixedEndpointer(hang_ms=args.hang_ms, frame_ms=20),
                providers[1], providers[2], providers[3],
            )

        try:
            tl = await run_once(clip, providers, label=f"run-{i + 1}")
        except Exception as exc:  # noqa: BLE001 - one bad run must not lose the rest
            tl = Timeline(label=f"run-{i + 1}", meta={"error": f"{type(exc).__name__}: {exc}"})
            print(f"  run {i + 1}: {type(exc).__name__}: {exc}", file=sys.stderr)
        runs.append(tl)

        done = "ok" if tl.complete else "incomplete"
        total = tl.measured_total_ms()
        print(f"  run {i + 1}/{args.n}: {done}" + (f"  {total:,.0f} ms" if total else ""), file=sys.stderr)

        if i < args.n - 1:
            await asyncio.sleep(args.gap)

    print(report.render(runs, network_ms=args.network, colour=sys.stdout.isatty()))
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(report.to_json(runs, args.network))
        print(f"wrote {args.json}", file=sys.stderr)

    stats, p50, _, n = report.summarise(runs, args.network)
    if n == 0:
        return 2
    return 0 if p50 <= 850 else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Speech-end to first-audio latency spike.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="synthetic providers, no keys, no network")
    mode.add_argument("--real", action="store_true", help="live providers from spike/providers/real.py")
    p.add_argument("-n", type=int, default=5, help="runs (five is the floor for a real reading)")
    p.add_argument("--wav", help="16-bit mono 16 kHz WAV; omit for a synthetic clip")
    p.add_argument("--network", type=int, default=80, help="assumed egress + jitter, ms")
    p.add_argument("--gap", type=float, default=0.6, help="seconds between runs")
    p.add_argument("--vad", choices=["silero", "mock"], default="silero", help="real mode only")
    p.add_argument("--hang-ms", type=int, default=500, help="silence hang time for the baseline VAD")
    p.add_argument("--semantic-vad", action="store_true", help="mock mode: model a fused endpointer")
    p.add_argument("--endpointer", choices=["provider", "fixed", "fused"], default="provider",
                   help="'fused' uses the real endpointer in spike/endpoint.py")
    p.add_argument("--context", default="open",
                   choices=["yes_no", "selection", "open", "digits", "address", "spelling"],
                   help="what the dialogue layer expects next; sets the base threshold")
    p.add_argument("--json", help="also write the full result set here")
    args = p.parse_args()

    if not args.mock and not args.real:
        args.mock = True
        print("no mode given, defaulting to --mock", file=sys.stderr)

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
