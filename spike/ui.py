"""Local dashboard for the harness. Standard library only.

    python3 -m spike.ui            # then open the URL it prints

The terminal output is fine for one run. This is for the case where you have
changed a lexicon entry and want to see, at a glance, which of six utterances
moved and whether anything now gets cut off — which is a picture, not a table.

Binds to 127.0.0.1 only. There is nothing here worth exposing and the harness
executes local code paths on request.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import audio as au
from . import cases as C
from . import clip_eval as ce
from .endpoint import BASE_MS, Context, FixedEndpointer, FusedEndpointer
from .endpoint_eval import FRAME_MS, run as run_case

HERE = Path(__file__).parent


def case_payload(hang_ms: int = 500) -> dict:
    out = []
    for case in C.CASES:
        fixed, ftrace = run_case(case, FixedEndpointer(hang_ms=hang_ms, frame_ms=FRAME_MS), False)
        fused, utrace = run_case(
            case,
            FusedEndpointer(context=case.context, frame_ms=FRAME_MS, expect_digits=case.expect_digits),
            True,
        )
        out.append({
            "name": case.name,
            "note": case.note,
            "context": case.context.value,
            "base_ms": BASE_MS[case.context],
            "segments": case.segments,
            "true_end": case.true_end_ms,
            "total": case.total_ms,
            "partials": case.partials,
            "fixed": _fire(fixed, case.true_end_ms),
            "fused": _fire(fused, case.true_end_ms),
            "fixed_trace": ftrace,
            "fused_trace": utrace,
        })
    clean_fixed = [c["fixed"]["latency"] for c in out if c["fixed"]["ok"]]
    clean_fused = [c["fused"]["latency"] for c in out if c["fused"]["ok"]]
    return {
        "hang_ms": hang_ms,
        "cases": out,
        "summary": {
            "fixed_cuts": sum(1 for c in out if c["fixed"]["cut"]),
            "fused_cuts": sum(1 for c in out if c["fused"]["cut"]),
            "fixed_mean": round(sum(clean_fixed) / len(clean_fixed)) if clean_fixed else None,
            "fused_mean": round(sum(clean_fused) / len(clean_fused)) if clean_fused else None,
        },
    }


def _fire(res, true_end: int) -> dict:
    return {
        "at": res.fired_ms,
        "latency": res.latency_ms,
        "cut": bool(res.false_cut),
        "never": bool(res.never_fired),
        "ok": bool(res.ok),
        "reason": res.reason,
    }


def clip_payload(wav: str, words: str | None, text: str | None, context: str, hang_ms: int) -> dict:
    clip = au.load_wav(wav)
    partials = ce.load_partials(clip, words, text)
    fixed = ce.run(clip, FixedEndpointer(hang_ms=hang_ms, frame_ms=ce.FRAME_MS), None)
    fused = ce.run(clip, FusedEndpointer(context=Context(context), frame_ms=ce.FRAME_MS), partials)
    true_end = clip.speech_end_s * 1000.0

    def f(x):
        return {
            "at": x.at_ms,
            "latency": None if x.at_ms is None else x.at_ms - true_end,
            "cut": x.at_ms is not None and x.at_ms < true_end,
            "never": x.at_ms is None,
            "reason": x.reason,
        }

    voiced, run_start = [], None
    for i, fr in enumerate(clip.frames(ce.FRAME_MS)):
        t = i * ce.FRAME_MS
        if ce._voiced(fr):
            if run_start is None:
                run_start = t
        elif run_start is not None:
            voiced.append([run_start, t])
            run_start = None
    if run_start is not None:
        voiced.append([run_start, clip.duration_s * 1000])

    return {
        "name": clip.name,
        "has_partials": partials is not None,
        "n_partials": len(partials or []),
        "segments": voiced,
        "true_end": true_end,
        "total": clip.duration_s * 1000,
        "gaps": ce.gaps(clip),
        "fixed": f(fixed),
        "fused": f(fused),
        "fused_trace": fused.trace,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                self._send(200, (HERE / "ui.html").read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/cases":
                self._json(case_payload(int(q.get("hang_ms", 500))))
            elif u.path == "/api/clip":
                wav = q.get("wav") or "audio/caller.wav"
                if not Path(wav).exists():
                    self._json({"error": f"no such file: {wav}"}, 404)
                    return
                self._json(clip_payload(wav, q.get("words") or None, q.get("text") or None,
                                        q.get("context", "open"), int(q.get("hang_ms", 500))))
            elif u.path == "/api/clips":
                found = sorted(str(p) for p in Path("audio").glob("*.wav"))
                self._json({"wavs": found})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001 - a bad request must not kill the server
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def log_message(self, *a) -> None:
        return  # the terminal is for the harness, not for access logs


def main() -> int:
    p = argparse.ArgumentParser(description="Local dashboard for the latency spike.")
    p.add_argument("--port", type=int, default=8420)
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"dashboard: {url}\nctrl-c to stop")
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
