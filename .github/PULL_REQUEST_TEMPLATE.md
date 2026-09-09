## What this changes

<!-- Why, not what. The diff already says what. -->

## Numbers

<!-- Any change touching the endpointer needs these. Paste the output. -->

```
python3 -m spike.endpoint_eval
```

## Checklist

- [ ] `python3 -m spike.test_endpoint` passes
- [ ] `python3 -m spike.endpoint_eval` reports **zero** false cuts
- [ ] Behaviour changes have a case that fails without this PR
- [ ] No new dependency in the CI-gated path
- [ ] New language? Its own dangler set, not merged into an existing one

## If this makes the endpointer faster

<!-- Say which of the two invariants you checked you have not weakened:
     the dangling veto, and waiting rather than firing when uncertain.
     A latency win that comes from weakening either is a regression. -->
