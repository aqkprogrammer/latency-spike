"""Score the endpointers against the cases. This is the number the spike exists for.

Two metrics, and they trade against each other. Report both or the result is
propaganda:

  latency     ms from the true end of speech to the endpoint decision
  false cut   fired during a mid-utterance pause, before the caller finished

A fixed timeout can always win on latency by being short, and always wins on
false cuts by being long. The only interesting question is what a single
configuration does across all six cases at once.

    python -m spike.endpoint_eval
    python -m spike.endpoint_eval --trace dangling
    python -m spike.endpoint_eval --wav audio/caller.wav --words audio/caller.words.json
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from .cases import CASES, Case, by_name
from .endpoint import FixedEndpointer, FusedEndpointer

FRAME_MS = 20
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, RED, AMBER = "\033[32m", "\033[31m", "\033[33m"


@dataclass
class Result:
    case: str
    fired_ms: float | None
    latency_ms: float | None
    false_cut: bool
    never_fired: bool
    reason: str

    @property
    def ok(self) -> bool:
        return not self.false_cut and not self.never_fired


def run(case: Case, endpointer, feed_partials: bool) -> tuple[Result, list[str]]:
    endpointer.reset()
    t = 0.0
    fired_at: float | None = None

    for frame in case.frames(FRAME_MS):
        if feed_partials:
            endpointer.update_partial(case.partial_at(t))
        if endpointer.push(frame):
            fired_at = t + FRAME_MS
            break
        t += FRAME_MS

    if fired_at is None:
        return Result(case.name, None, None, False, True, "never fired"), list(endpointer.trace)

    latency = fired_at - case.true_end_ms
    false_cut = fired_at < case.true_end_ms
    reason = endpointer.decision.reason if endpointer.decision else ""
    return Result(case.name, fired_at, latency, false_cut, False, reason), list(endpointer.trace)


def evaluate(feed_partials: bool, endpointer_factory) -> list[Result]:
    out = []
    for case in CASES:
        ep = endpointer_factory(case)
        res, _ = run(case, ep, feed_partials)
        out.append(res)
    return out


def _cell(r: Result, colour: bool) -> str:
    g, rd, am, rs = (GREEN, RED, AMBER, RESET) if colour else ("", "", "", "")
    if r.never_fired:
        return f"{am}never fired{rs}"
    if r.false_cut:
        return f"{rd}CUT at {r.fired_ms:,.0f}ms{rs}"
    return f"{g}+{r.latency_ms:,.0f}ms{rs}"


def run_clip(args, colour: bool) -> int:
    from . import audio as au
    from . import clip_eval as ce
    from .endpoint import Context

    b, rs, dm = (BOLD, RESET, DIM) if colour else ("", "", "")
    clip = au.load_wav(args.wav)
    partials = ce.load_partials(clip, args.words, args.text)

    print()
    print(f"{b}{clip.name}{rs}  {clip.duration_s:.2f}s  "
          f"{dm}speech ends at {clip.speech_end_s*1000:,.0f} ms (ground truth){rs}")
    if partials:
        src = args.words or args.text
        print(f"{dm}partials: {len(partials)} from {src}{rs}")
    else:
        print(f"{AMBER if colour else ''}no transcript — semantic gate inert, so the fused "
              f"endpointer is running on two signals of three.{rs}")
        print(f"{dm}generate one: python -m spike.transcribe {args.wav}{rs}")
    print()

    true_end = clip.speech_end_s * 1000.0
    fixed = ce.run(clip, FixedEndpointer(hang_ms=args.hang_ms, frame_ms=ce.FRAME_MS), None)
    fused = ce.run(clip, FusedEndpointer(context=Context(args.context), frame_ms=ce.FRAME_MS), partials)

    def line(label, f):
        if f.at_ms is None:
            return f"{label:<14}{'never fired':>14}{'':>14}   {AMBER if colour else ''}no endpoint in clip{rs}"
        delta = f.at_ms - true_end
        cut = f.at_ms < true_end
        mark = (f"{RED if colour else ''}CUT{rs}" if cut else f"{GREEN if colour else ''}ok{rs}")
        return (f"{label:<14}{f.at_ms:>11,.0f} ms{delta:>+11,.0f} ms   {mark}  {dm}{f.reason}{rs}")

    print(f"{dm}{'endpointer':<14}{'fired at':>14}{'vs true end':>14}{rs}")
    print("─" * 78)
    print(line(f"fixed {args.hang_ms}", fixed))
    print(line("fused", fused))
    print("─" * 78)

    if fixed.at_ms and fused.at_ms:
        d = fixed.at_ms - fused.at_ms
        cuts = ("fixed" if fixed.at_ms < true_end else "") + ("+fused" if fused.at_ms < true_end else "")
        if d > 0 and not cuts:
            print(f"{GREEN if colour else ''}fused saved {d:,.0f} ms with no cut.{rs}")
        elif cuts:
            print(f"{RED if colour else ''}false cut: {cuts.strip('+')}{rs}")
        else:
            print(f"{dm}fused was {-d:,.0f} ms slower on this clip.{rs}")

    g = ce.gaps(clip)
    print()
    print(f"{b}Mid-utterance pauses{rs}  {dm}{len(g)} gap(s) over {ce.GAP_MS} ms — these are where a cut happens{rs}")
    if not g:
        print(f"{dm}  none. A single-run utterance only tests speed, not restraint —{rs}")
        print(f"{dm}  find a clip where the caller pauses to think.{rs}")
    for a, bnd in g:
        who = []
        if fixed.at_ms is not None and a <= fixed.at_ms <= bnd: who.append(f"{RED if colour else ''}fixed CUT{rs}")
        if fused.at_ms is not None and a <= fused.at_ms <= bnd: who.append(f"{RED if colour else ''}fused CUT{rs}")
        verdict = " · ".join(who) if who else f"{GREEN if colour else ''}both held{rs}"
        said = ce.partial_at(partials, a) if partials else ""
        tail = f'  {dm}"…{said[-46:]}"{rs}' if said else ""
        print(f"  {a:>6,.0f}–{bnd:<7,.0f} ms  {bnd-a:>5,.0f} ms   {verdict}{tail}")

    print()
    print(f"{b}fused trace{rs}")
    for t in fused.trace:
        print(f"  {dm}{t}{rs}")
    print()
    return 0 if (fused.at_ms is not None and fused.at_ms >= true_end) else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Fixed versus fused endpointing, across the case set.")
    p.add_argument("--hang-ms", type=int, default=500, help="baseline fixed timeout")
    p.add_argument("--trace", help="print the decision trace for one case and exit")
    p.add_argument("--wav", help="run against a real recording instead of the case set")
    p.add_argument("--words", help="word timings JSON from spike.transcribe")
    p.add_argument("--text", help="plain transcript, if you have no timings")
    p.add_argument("--context", default="open",
                   choices=["yes_no","selection","open","digits","address","spelling"])
    args = p.parse_args()

    if args.wav:
        return run_clip(args, colour=sys.stdout.isatty())
    colour = sys.stdout.isatty()
    b, rs, dm = (BOLD, RESET, DIM) if colour else ("", "", "")

    if args.trace:
        case = by_name(args.trace)
        print(f"\n{b}{case.name}{rs}  {dm}{case.note}{rs}")
        print(f"true end of speech: {case.true_end_ms} ms\n")
        for label, ep, partials in (
            ("fixed", FixedEndpointer(hang_ms=args.hang_ms, frame_ms=FRAME_MS), False),
            ("fused", FusedEndpointer(context=case.context, frame_ms=FRAME_MS,
                                      expect_digits=case.expect_digits), True),
        ):
            res, trace = run(case, ep, partials)
            print(f"{b}{label}{rs}  {_cell(res, colour)}  {dm}{res.reason}{rs}")
            for line in trace:
                print(f"   {dm}{line}{rs}")
            print()
        return 0

    fixed = evaluate(False, lambda c: FixedEndpointer(hang_ms=args.hang_ms, frame_ms=FRAME_MS))
    fused = evaluate(True, lambda c: FusedEndpointer(context=c.context, frame_ms=FRAME_MS,
                                                     expect_digits=c.expect_digits))

    print()
    print(f"{b}Endpointing: fixed {args.hang_ms} ms timeout versus fused{rs}")
    print("─" * 78)
    print(f"{dm}{'case':<20}{'fixed':>18}{'fused':>18}   note{rs}")
    for f, u in zip(fixed, fused):
        case = by_name(f.case)
        note = case.note.split(".")[0]
        print(f"{f.case:<20}{_cell(f, colour):>{18 + (9 if colour else 0)}}"
              f"{_cell(u, colour):>{18 + (9 if colour else 0)}}   {dm}{note[:34]}{rs}")
    print("─" * 78)

    def tally(rows: list[Result]) -> tuple[int, int, float]:
        cuts = sum(1 for r in rows if r.false_cut)
        never = sum(1 for r in rows if r.never_fired)
        good = [r.latency_ms for r in rows if r.ok and r.latency_ms is not None]
        return cuts, never, (sum(good) / len(good)) if good else 0.0

    fc, fn, fl = tally(fixed)
    uc, un, ul = tally(fused)
    print(f"{'false cuts':<20}{fc:>18}{uc:>18}")
    print(f"{'never fired':<20}{fn:>18}{un:>18}")
    print(f"{'mean latency (clean)':<20}{fl:>17,.0f}m{ul:>17,.0f}m")
    print()

    if uc == 0 and ul < fl:
        print(f"{GREEN if colour else ''}Fused wins on both axes: "
              f"{fc - uc} fewer cuts and {fl - ul:,.0f} ms faster on the turns it gets right.{rs}")
    elif uc < fc:
        print(f"{GREEN if colour else ''}Fused cuts {fc - uc} fewer callers off.{rs}")
    else:
        print(f"{AMBER if colour else ''}No improvement. Check the lexicons and the partial schedule.{rs}")
    print()
    return 0 if uc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
