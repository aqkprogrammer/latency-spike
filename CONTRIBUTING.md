# Contributing

The most useful contributions to this project are new languages and new failure
cases. Both are small, both are testable, and both come from people who have
heard a real queue rather than read about one.

## The one thing to read before changing the endpointer

**Cutting a caller off costs far more than a moment of hesitation.**

Everything in `spike/endpoint.py` follows from that asymmetry, and two
behaviours encode it directly. Neither is a tuning parameter:

- **The dangling veto.** `SemanticGate.classify` returns `Semantic.VETO` for a
  clause ending on a preposition, conjunction, postposition or filler, and
  `FusedEndpointer.threshold_ms` turns that into `MAX_HOLD_MS` — the endpoint
  waits regardless of how long the silence runs.
- **Waiting when uncertain.** `MAX_HOLD_MS` is a ceiling that hands over, not a
  timeout that fires an answer, and `SemanticGate` returns `VETO` rather than
  `NEUTRAL` for anything it cannot read.

Neither is behind a flag, deliberately. A flag invites being turned off.

A pull request that makes the endpointer faster by weakening either of those is
not an optimisation, it is a regression that the latency numbers will flatter.
`spike/endpoint_eval.py` exists to catch exactly that, and it is a merge gate.

Everything else in `BASE_MS` is a starting point measured on Indian PSTN traffic
and should be re-tuned per market. Tune those freely.

## Running it

No dependencies for anything gated in CI — it is all standard library.

```bash
python3 -m spike.test_endpoint      # 12 groups, including the no-false-cut gate
python3 -m spike.endpoint_eval      # fixed vs fused across the case set
python3 -m spike.bench --mock -n 7  # the latency harness, synthetic providers
```

`aiohttp` is needed only for the live providers in `spike/providers/real.py`,
which CI never calls. Please keep it that way — a contribution that pulls a
dependency into the gated path will be asked to move it behind the provider
boundary.

## Adding a language

The highest-value contribution, and the one most easily got wrong.

**Do not merge a new language into an existing lexicon.** English is SVO, so a
trailing auxiliary — "it is", "I was" — is mid-clause and dangles. Hindi is
verb-final, so a trailing auxiliary — "mera order kahan **hai**" — is how a
sentence ends. Merging the two produces an endpointer that waits forever on
completed Hindi and cuts off unfinished Hindi. Every language needs its own set
and its own answer to that question.

To add one:

1. **Create the dangler set** in `spike/endpoint.py`, alongside `EN_DANGLERS`
   and `HI_DANGLERS`. Include prepositions or postpositions, conjunctions,
   determiners — and auxiliaries **only if** the language is not verb-final.
2. **Extend the matrix-language sniff** in `_looks_hindi`, or add an equivalent.
   Count *function* words only. Content-word loans are not a language switch:
   "Bhai mera order cancel karo" is Hindi with four English nouns in it, and a
   token-counting detector gets this exactly backwards.
3. **Add accelerators** — the language's short affirmatives and negatives.
4. **Add two cases** to `spike/cases.py`: one utterance that pauses on a
   dangling word, and one that legitimately ends on whatever an English-tuned
   lexicon would wrongly treat as dangling. The second is the important one —
   `hinglish_complete` exists for precisely that reason.
5. **Add lexicon tests** to `spike/test_endpoint.py`, following
   `test_hindi_auxiliary_is_not_a_dangler`.
6. **Run `python3 -m spike.endpoint_eval`.** False cuts must be zero. Not
   "lower than before" — zero.

If you have real audio in the language, run it through
`spike.endpoint_eval --wav` with a transcript sidecar and put the numbers in the
pull request. That is worth more than the lexicon itself.

## Adding a failure case

If you find an utterance where the endpointer cuts someone off or hangs:

1. **Add the case first, before the fix.** A case that fails on `main` and
   passes on your branch is the strongest possible pull request.
2. Put it in `spike/cases.py` with a `note` explaining what it tests and an
   `origin` describing where you saw it.
3. Fix it.
4. Confirm the other cases still pass. A fix that breaks `hinglish_complete`
   is a common way to fix a dangler.

**Every bug becomes a permanent case.** That is the only mechanism that stops
this relearning the same failure, and it is not optional here.

## Measurement rules

These are easy to break by accident and they silently produce flattering
numbers.

- **Measure from the true end of speech**, not from end of file, or you measure
  your own trailing silence. `spike/audio.speech_end_sample` is ground truth.
- **Sleep to absolute deadlines**, never per-frame. Per-frame sleeps drift over
  a six-second utterance and turn a latency measurement into a throughput one.
- **Never report a single run.** Five is the floor. `spike/report.py` refuses a
  verdict below three and says why.
- **Report false cuts alongside latency, always.** A fixed timeout can always
  win on latency by being short. One number proves nothing.
- **Disable the recogniser's own endpointing.** Otherwise you are benchmarking
  their silence timer rather than your pipeline.

## Pull requests

The three CI gates are the checklist:

- [ ] `python3 -m spike.test_endpoint` passes
- [ ] `python3 -m spike.endpoint_eval` reports **zero** false cuts
- [ ] `python3 -m spike.endpoint_eval --wav … --words …` still runs
- [ ] Any behaviour change has a case that fails without it
- [ ] No new dependency in the gated path

Keep the diff small enough to review in one sitting. If a change touches both
the endpointer and the harness, two pull requests are easier for everyone.

## Commit messages

Explain **why**, not what — the diff already says what. The existing history is
the model: what the change does, the reasoning that is not obvious from the
code, and the measurement that justifies it. A one-line "fix endpointing" tells
the next person nothing, and on this codebase the reasoning is the valuable part.

## Vendor code

`spike/providers/real.py` and `spike/room_agent.py` talk to third-party APIs and
carry `VERIFY:` comments with doc links. If you update one because a wire format
moved, say so in the message and note the date you checked. Nobody can tell a
stale endpoint from a wrong one by reading it.

## Licence

Contributions are accepted under [Apache-2.0](LICENSE), the licence this project
uses. By opening a pull request you confirm you have the right to submit the
work under it.

## Conduct

Be straightforward and assume good faith. Critique the code, not the person.
Disagreements about the endpointer get settled by adding a case and running the
eval, which is a better argument than either of us can make in a comment thread.
