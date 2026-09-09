"""Turn a set of Timelines into the only report that matters: a waterfall
against the budget, with P50 and P95, and a verdict.

Never report a single run. Five is the floor - see the noise-band argument in
the evaluation-harness spec. The report refuses to print a verdict below three.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass

from .clock import BUDGET_MS, HOPS, TARGET_TOTAL_MS, Timeline

BAR_W = 34
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, AMBER, RED, CYAN = "\033[32m", "\033[33m", "\033[31m", "\033[36m"


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class HopStat:
    name: str
    p50: float
    p95: float
    budget: int | None
    measured: bool
    n: int

    @property
    def over_ms(self) -> float:
        return 0.0 if self.budget is None else max(0.0, self.p50 - self.budget)


def summarise(runs: list[Timeline], network_ms: int) -> tuple[list[HopStat], float, float, int]:
    ok = [r for r in runs if r.complete]
    stats: list[HopStat] = []

    for hop, _, _ in HOPS:
        vals = [v for v in (r.hop_ms(hop) for r in ok) if v is not None]
        stats.append(
            HopStat(hop, _pct(vals, 0.50), _pct(vals, 0.95), BUDGET_MS.get(hop), True, len(vals))
        )

    stats.append(HopStat("network", float(network_ms), float(network_ms), BUDGET_MS["network"], False, len(ok)))

    totals = [t for t in (r.measured_total_ms() for r in ok) if t is not None]
    p50 = _pct(totals, 0.50) + network_ms
    p95 = _pct(totals, 0.95) + network_ms
    return stats, p50, p95, len(ok)


def _bar(ms: float, scale: float, colour: str) -> str:
    filled = 0 if scale <= 0 else max(1, min(BAR_W, round(BAR_W * ms / scale)))
    return f"{colour}{'█' * filled}{RESET}{DIM}{'·' * (BAR_W - filled)}{RESET}"


def render(runs: list[Timeline], network_ms: int, colour: bool = True) -> str:
    global RESET, BOLD, DIM, GREEN, AMBER, RED, CYAN
    if not colour:
        RESET = BOLD = DIM = GREEN = AMBER = RED = CYAN = ""

    stats, p50, p95, n = summarise(runs, network_ms)
    incomplete = [r for r in runs if not r.complete]
    scale = max([s.p50 for s in stats] + [1.0])

    lines: list[str] = []
    lines.append("")
    lines.append(f"{BOLD}Speech-end to first audio out{RESET}   {DIM}{n} complete run(s){RESET}")
    lines.append("─" * 78)
    lines.append(f"{DIM}{'hop':<12}{'':<{BAR_W}}  {'p50':>7} {'p95':>7}  {'budget':>7}   verdict{RESET}")

    for s in stats:
        if s.budget is None:
            colr, verdict = CYAN, ""
        elif s.measured and s.p50 > s.budget:
            colr, verdict = RED, f"{RED}over by {s.p50 - s.budget:,.0f}{RESET}"
        elif not s.measured:
            colr, verdict = CYAN, f"{DIM}assumed{RESET}"
        else:
            colr, verdict = GREEN, f"{GREEN}ok{RESET}"
        budget = f"{s.budget:,}" if s.budget is not None else "—"
        lines.append(
            f"{s.name:<12}{_bar(s.p50, scale, colr)}  {s.p50:>6,.0f}m {s.p95:>6,.0f}m  {budget:>7}   {verdict}"
        )

    lines.append("─" * 78)
    tcol = GREEN if p50 <= TARGET_TOTAL_MS else RED
    delta = p50 - TARGET_TOTAL_MS
    tail = f"{GREEN}inside budget by {-delta:,.0f} ms{RESET}" if delta <= 0 else f"{RED}OVER by {delta:,.0f} ms{RESET}"
    lines.append(f"{BOLD}{'total':<12}{'':<{BAR_W}}  {p50:>6,.0f}m {p95:>6,.0f}m  {TARGET_TOTAL_MS:>7}   {tail}{RESET}")
    lines.append("")

    if n < 3:
        lines.append(f"{AMBER}Fewer than three complete runs. Treat this as a smoke test, not a measurement.{RESET}")
    if incomplete:
        why: dict[str, int] = {}
        for r in incomplete:
            for m in r.missing():
                why[m] = why.get(m, 0) + 1
        detail = ", ".join(f"{k} missing in {v}" for k, v in sorted(why.items()))
        lines.append(f"{AMBER}{len(incomplete)} incomplete run(s): {detail}{RESET}")

    worst = max((s for s in stats if s.measured), key=lambda s: s.over_ms, default=None)
    if worst is not None and worst.over_ms > 0:
        lines.append("")
        lines.append(f"{BOLD}Biggest single win:{RESET} {worst.name} is {worst.over_ms:,.0f} ms over budget.")
        lines.append(f"{DIM}{HINTS.get(worst.name, '')}{RESET}")

    lines.append("")
    return "\n".join(lines)


HINTS = {
    "endpoint": "Fixed silence timeout? Replace it with semantic endpointing - biggest lever in the stack.",
    "stt_final": "Ask the provider for an immediate finalisation on endpoint rather than waiting for its own VAD.",
    "llm_ttft": "Cache the stable prompt prefix, shorten the system prompt, try a smaller model for the first sentence.",
    "tts_ttfb": "Stream from the first sentence, not the full completion. Check you are on the low-latency model.",
}


def to_json(runs: list[Timeline], network_ms: int) -> str:
    stats, p50, p95, n = summarise(runs, network_ms)
    return json.dumps(
        {
            "complete_runs": n,
            "target_total_ms": TARGET_TOTAL_MS,
            "total_p50_ms": round(p50, 1),
            "total_p95_ms": round(p95, 1),
            "within_budget": p50 <= TARGET_TOTAL_MS,
            "hops": [
                {
                    "name": s.name,
                    "p50_ms": round(s.p50, 1),
                    "p95_ms": round(s.p95, 1),
                    "budget_ms": s.budget,
                    "measured": s.measured,
                }
                for s in stats
            ],
            "runs": [
                {"label": r.label, "hops_ms": {k: round(v, 1) for k, v in r.hops_ms().items()},
                 "total_ms": round(r.measured_total_ms() or 0, 1), "complete": r.complete, "meta": r.meta}
                for r in runs
            ],
        },
        indent=2,
    )
