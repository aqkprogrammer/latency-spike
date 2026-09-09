# Latency spike

Measures one number: **end of the caller's speech → first byte of audio out**,
broken down by hop, against the 850 ms budget.

Nothing here is an agent. There is no intelligence on the critical path by
design — the point is to find out whether the *transport and the providers* can
reach the budget before anything is built on top of them. Two days of work here
either de-risks the whole plan or tells you the stack choices are wrong, which
is the cheapest bad news you will ever buy.

## Run it now, no keys

```bash
python3 -m spike.bench --mock -n 7
```

Synthetic providers with realistic lognormal latency. If these numbers look
wrong, the bug is in the harness, not in a vendor — which is a distinction
worth being able to make in ten seconds.

Then see what the whole argument is about:

```bash
python3 -m spike.bench --mock --semantic-vad -n 7
```

Same pipeline, fused endpointer instead of a fixed silence timeout. That single
swap is the difference between missing budget by 200 ms and clearing it by 100.

## Run it for real

```bash
cp .env.example .env      # fill in the keys
pip install -r requirements.txt
# drop a 16-bit mono 16 kHz recording at audio/caller.wav — see audio/README.md
./run.sh real 7
```

Use a recording from the actual queue you are targeting. Narrowband PSTN audio
with background noise is what the endpointer has to survive, and a clean studio
read will flatter every number in the report.

## The endpointer

The largest hop is endpointing, so `spike/endpoint.py` implements the fused
endpointer from the turn-taking spec: acoustic silence, an energy-decay prosody
cue, and a semantic gate reading STT partials. The semantic signal moves the
*threshold* rather than casting a vote, so every decision is one explainable
line rather than an opaque score.

```bash
python3 -m spike.endpoint_eval              # fixed vs fused, across six cases
python3 -m spike.endpoint_eval --trace dangling   # why it decided what it decided
python3 -m spike.test_endpoint              # lexicon and no-false-cut gates
```

### On a real recording

```bash
python3 -m spike.transcribe audio/caller.wav        # once — writes caller.words.json
python3 -m spike.endpoint_eval --wav audio/caller.wav \
                               --words audio/caller.words.json
```

Ground truth for "the caller finished here" is the last speech-bearing window in
the file, so any fire before it is a false cut with nothing to annotate by hand.
The report ends with a **mid-utterance pause list** — every silence gap over
250 ms and what each endpointer did there. That is the whole game on a real
clip: a single-run utterance only tests speed, never restraint.

Transcribe once, then iterate on the endpointer for free. Without a transcript
the semantic gate has nothing to read and the fused endpointer runs degraded on
two signals of three — the report says so, and on a pausing caller it cuts them
off just like the fixed timeout does.

```
case                             fixed             fused
complete                        +440ms            +320ms
dangling                CUT at 1,860ms            +360ms
digits                          +500ms            +140ms
short_answer                    +510ms            +150ms
hinglish_dangling       CUT at 2,000ms            +360ms
hinglish_complete               +500ms            +340ms
────────────────────────────────────────────────────────
false cuts                           2                 0
mean latency (clean)              488m              278m
```

Two metrics, always reported together. A fixed timeout can always win on
latency by being short, and always wins on false cuts by being long; the only
interesting question is what one configuration does across all six cases at once.

Use it in the bench with `--endpointer fused --context <ctx>`:

| context | endpoint p50 | total p50 |
|---|---|---|
| `yes_no` | 121 ms | 668 ms |
| `selection` | 201 ms | 747 ms |
| `open` | 361 ms | 908 ms |

`open` is above the 220 ms line and should be — the 220 in the budget table is a
blend, and an open-ended question is inherently the slow case. Optimise the mix
by making the dialogue layer declare a narrower context wherever it honestly can,
not by shortening the open threshold.

### The two things not to tune

`dangling_veto` and *when uncertain, wait*. Both encode the same asymmetry:
cutting a caller off costs far more than a moment of hesitation. Everything else
in `BASE_MS` is a starting point measured on Indian PSTN traffic and should be
re-tuned per market.

### Hindi is not English with different words

English is SVO, so a trailing auxiliary — "it is", "I was" — is mid-clause and
dangles. Hindi is verb-final, so a trailing auxiliary — "mera order kahan **hai**"
— is how a sentence *ends*. Merging the two lexicons makes an endpointer that
waits forever on completed Hindi and cuts off unfinished Hindi, which is why
`EN_DANGLERS` and `HI_DANGLERS` are separate and why `hinglish_complete` is in
the case set. The matrix-language sniff counts Hindi *function* words only —
"order" and "delivery" are loanwords, not a language switch.

## The two phases

**Phase 1 — `spike/bench.py`.** Offline. Feeds a WAV through VAD → STT → LLM →
TTS at realtime pace and records the four hops. Runs on a laptop with nothing
but API keys. Start here.

**Phase 2 — `spike/room_agent.py`.** The same measurement inside a live LiveKit
room, so the figure includes transport, framework and network. Routinely another
100–200 ms that nobody budgeted for. Only start this once phase 1 looks sane —
debugging a room and a provider simultaneously is how a two-day spike becomes a
two-week one.

```bash
pip install 'livekit-agents[deepgram,elevenlabs,openai,silero]'
python3 -m spike.room_agent dev          # join the room from a browser client
python3 -m spike.summarise_room          # waterfall from room-runs.jsonl
```

## Reading the report

```
hop                                  p50     p95   budget   verdict
endpoint    ████████████████████    501m    565m      220   over by 281
stt_final   █████···············     71m     85m       70   over by 1
llm_ttft    ████████████████·····   234m    518m      300   ok
tts_ttfb    ████████·············    118m    152m      120   ok
network     █████················     80m     80m       80   assumed
total                              1,048m  1,239m      850   OVER by 198 ms
```

- **endpoint** — true end of speech → the endpointer commits. Measured from
  ground truth in the WAV, not from end of file. Almost always the largest hop
  and almost always the cheapest to fix.
- **stt_final** — endpoint → final transcript. We disable the recogniser's own
  endpointing so this measures your pipeline rather than their silence timer.
- **llm_ttft** — transcript → first token.
- **tts_ttfb** — first token → first audio byte.
- **network** — assumed, not measured offline. Phase 2 replaces the assumption.

Watch p95, not just p50. Callers forgive a slow median far more readily than an
unpredictable one, and p95 is what a supervisor hears when they spot-check.

Exit code is 0 inside budget, 1 over, 2 if nothing completed — so this drops
straight into CI as a gate once you have a number worth defending.

## What to do with the result

| Hop over budget | First thing to try |
|---|---|
| endpoint | Semantic endpointing. Fixed silence timeouts are the default in every stack and the single biggest lever available. |
| stt_final | Ask the provider to finalise on your signal instead of waiting out its own VAD. |
| llm_ttft | Cache the stable prompt prefix; shorten the system prompt; consider a smaller model for the first sentence only. |
| tts_ttfb | Stream from the first sentence rather than the full completion; confirm you are on the flash/turbo model. |
| network | Co-locate everything in one region per geography. A cross-region hop costs more than any optimisation elsewhere. |

## Layout

```
spike/clock.py            Timeline, hop definitions, drift-free realtime pacing
spike/report.py           Percentiles, waterfall, verdict, JSON export
spike/audio.py            WAV loading and ground-truth speech-end detection
spike/endpoint.py         Fused endpointer: semantic gate, prosody, thresholds
spike/cases.py            Six endpointing cases with exact ground truth
spike/endpoint_eval.py    Fixed vs fused scoring
spike/test_endpoint.py    Lexicon cases and the no-false-cut gate
spike/providers/          Four-method protocols; mock.py and real.py
spike/bench.py            Phase 1 entry point
spike/room_agent.py       Phase 2 entry point (LiveKit)
spike/summarise_room.py   Waterfall from room-runs.jsonl
```

## Two things to verify before trusting a real number

1. **Provider wire details** in `spike/providers/real.py`. Each class carries a
   `VERIFY:` comment with its doc URL. A wrong parameter usually shows up as a
   plausible-but-wrong latency rather than an error.
2. **The LiveKit event names** in `spike/room_agent.py`. That API has moved more
   than once; the handlers degrade to an incomplete run rather than crashing,
   and the report names whichever mark went missing.
