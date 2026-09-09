#!/usr/bin/env bash
# Convenience wrapper. Everything here is just `python -m spike.<module>`.
set -euo pipefail
[ -f .env ] && set -a && . ./.env && set +a
case "${1:-mock}" in
  mock)     python3 -m spike.bench --mock -n "${2:-7}" ;;
  fused)    python3 -m spike.bench --mock --semantic-vad -n "${2:-7}" ;;
  real)     python3 -m spike.bench --real -n "${2:-7}" --wav "${WAV:-audio/caller.wav}" --json results.json ;;
  endpoint) python3 -m spike.endpoint_eval ;;
  test)     python3 -m spike.test_endpoint ;;
  clip)     python3 -m spike.endpoint_eval --wav "${2:-audio/caller.wav}" --words "${3:-audio/caller.words.json}" ;;
  room)     python3 -m spike.room_agent dev ;;
  summary)  python3 -m spike.summarise_room "${2:-room-runs.jsonl}" ;;
  *) echo "usage: ./run.sh {mock|fused|endpoint|clip|test|real|room|summary} [n]"; exit 2 ;;
esac
