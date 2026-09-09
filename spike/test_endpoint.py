"""Cases for the endpointer. Run with: python -m spike.test_endpoint

No pytest dependency - this has to be runnable on a bare machine during a spike.
Every lexicon entry that ever caused a false cut in production should get a case
here before the fix ships, per the regression rule in the eval-harness spec.
"""
from __future__ import annotations

import sys

from .cases import CASES, by_name
from .endpoint import (
    Context,
    FixedEndpointer,
    FusedEndpointer,
    Semantic,
    SemanticGate,
    count_digits,
)
from .endpoint_eval import run

FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")


def test_english_danglers() -> None:
    g = SemanticGate(Context.OPEN)
    for text in ("i want to move it to", "can you check the", "it is",
                 "please send it and", "my order was", "i need to um"):
        sem, why = g.classify(text)
        check(f"en dangler {text!r}", sem, Semantic.VETO)


def test_english_complete() -> None:
    g = SemanticGate(Context.OPEN)
    for text in ("move it to thursday", "my order never arrived",
                 "can you check my delivery"):
        sem, _ = g.classify(text)
        check(f"en complete {text!r}", sem, Semantic.NEUTRAL)


def test_hindi_auxiliary_is_not_a_dangler() -> None:
    """The failure this whole lexicon split exists to prevent."""
    g = SemanticGate(Context.OPEN)
    for text in ("mera order kahan hai", "delivery kal aayi thi",
                 "mujhe kal chahiye"):
        sem, why = g.classify(text)
        check(f"hi complete {text!r} ({why})", sem, Semantic.NEUTRAL)


def test_hindi_postposition_dangles() -> None:
    g = SemanticGate(Context.OPEN)
    for text in ("mera order abhi tak nahi aaya usko",
                 "yeh package mere ghar ke",
                 "delivery kal karni hai lekin"):
        sem, _ = g.classify(text)
        check(f"hi dangler {text!r}", sem, Semantic.VETO)


def test_digits_context() -> None:
    g = SemanticGate(Context.DIGITS, expect_digits=10)
    check("3 digits", g.classify("nine eight seven")[0], Semantic.VETO)
    check("10 digits", g.classify("nine eight seven six five four three two one zero")[0],
          Semantic.ACCELERATE)
    check("count numeric", count_digits("9876543210"), 10)
    check("count words", count_digits("nine eight seven"), 3)
    check("count mixed", count_digits("98765 four three two one zero"), 10)


def test_short_answers_accelerate() -> None:
    g = SemanticGate(Context.YES_NO)
    for text in ("yes", "haan", "nahi", "theek hai"):
        sem, _ = g.classify(text)
        check(f"accelerate {text!r}", sem, Semantic.ACCELERATE)


def test_empty_partial_vetoes() -> None:
    """No transcript means no evidence. Wait, never guess."""
    check("empty", SemanticGate(Context.OPEN).classify("")[0], Semantic.VETO)


def test_no_false_cuts_across_case_set() -> None:
    """The gate that matters. A cut here blocks the change, always."""
    for case in CASES:
        ep = FusedEndpointer(context=case.context, frame_ms=20,
                             expect_digits=case.expect_digits)
        res, _ = run(case, ep, feed_partials=True)
        check(f"no cut on {case.name}", res.false_cut, False)
        check(f"fired on {case.name}", res.never_fired, False)


def test_fused_beats_fixed_on_latency() -> None:
    clean_fixed, clean_fused = [], []
    for case in CASES:
        f, _ = run(case, FixedEndpointer(hang_ms=500, frame_ms=20), False)
        u, _ = run(case, FusedEndpointer(context=case.context, frame_ms=20,
                                         expect_digits=case.expect_digits), True)
        if f.ok:
            clean_fixed.append(f.latency_ms)
        if u.ok:
            clean_fused.append(u.latency_ms)
    mf = sum(clean_fixed) / max(1, len(clean_fixed))
    mu = sum(clean_fused) / max(1, len(clean_fused))
    if mu >= mf:
        FAILURES.append(f"fused not faster: {mu:.0f}ms vs fixed {mf:.0f}ms")


def test_ceiling_holds() -> None:
    """A permanent veto must still hand over rather than wait forever."""
    ep = FusedEndpointer(context=Context.OPEN, frame_ms=20)
    ep.update_partial("i want to move it to")   # veto, forever
    silence = b"\x00\x00" * 160
    fired = False
    for _ in range(400):                        # 8 s of silence
        if ep.push(silence):
            fired = True
            break
    check("ceiling never reached without speech", fired, False)

    ep.reset()
    ep.update_partial("i want to move it to")
    loud = b"\x40\x10" * 160
    for _ in range(10):
        ep.push(loud)
    fired = any(ep.push(silence) for _ in range(400))
    check("ceiling fires after speech", fired, True)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            FAILURES.append(f"{t.__name__} raised {type(exc).__name__}: {exc}")

    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"  ✗ {f}")
        return 1
    print(f"\n{len(tests)} test groups passed")
    return 0




def test_degrades_without_partials() -> None:
    """No transcript source at all must not hang the turn."""
    ep = FusedEndpointer(context=Context.OPEN, frame_ms=20)
    loud, quiet = b"\x40\x10" * 160, b"\x00\x00" * 160
    for _ in range(30):
        ep.push(loud)
    fired = False
    for i in range(60):          # 1.2 s of silence — well under the 4 s ceiling
        if ep.push(quiet):
            fired = True
            break
    check("fires without partials", fired, True)
    if ep.decision:
        check("marked degraded", ep.decision.reason, "semantic unavailable")


def test_still_vetoes_once_partials_flow() -> None:
    """Degraded mode must not swallow the dangling veto when STT is working."""
    ep = FusedEndpointer(context=Context.OPEN, frame_ms=20)
    ep.update_partial("i want to move it to")
    loud, quiet = b"\x40\x10" * 160, b"\x00\x00" * 160
    for _ in range(30):
        ep.push(loud)
    fired = any(ep.push(quiet) for _ in range(60))
    check("veto survives", fired, False)


if __name__ == "__main__":
    sys.exit(main())
