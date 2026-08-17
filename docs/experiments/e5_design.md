# E5 — human-readable corpus perturbations (outline only, not run)

**Outline. No E5 data exists.** `RETURN AFTER` a predictive state readout
exists (E4b), because perturbing a state we cannot read leaves nothing to
measure the perturbation against.

## 1. The question

> Which **human-readable properties** of earlier training move the
> developmental state, and does moving it change future learnability?

    corpus property  ->  ΔS  ->  ΔV(S, D_future)

Not "which corpus performs best" — that is the data main effect, already
measured and already the thing our failed selector learned.

## 2. Properties to perturb, one at a time

Each is a single-property variant of the source, holding token budget, sequence
length and answer-marginal fixed. The A/A′ construction is the template: two
streams matched on every low-order statistic, differing in one property, with
the match verified against a same-stream null before any run.

| property | manipulation |
|---|---|
| binding density | fraction of clauses carrying a retrievable binding |
| entity reuse vs disjointness | recurrence rate within a document |
| retrieval distance | separation between binding and query |
| compositional depth | one-hop vs two-hop retrieval |
| factual-association density | proportion of weight-storable associations |
| interference | contradictory bindings for the same entity |
| repetition / frequency | Zipf slope over entities |
| distributional diversity | number of distinct templates |

## 3. Measurement, in three linked steps

1. **ΔS** — the state readout from E4b, applied before and after the perturbed
   source phase.
2. **ΔV** — identical continuations from each perturbed state, giving
   `V(S,D)` on the frozen common yardstick.
3. **The link** — does the property→ΔS→ΔV chain hold, and does ΔS *mediate* the
   property's effect on ΔV, or merely accompany it?

Step 3 is the one that matters and the one most easily faked. Mediation
requires the property's effect on `ΔV` to shrink when conditioning on `ΔS`;
a correlation among all three is not enough.

## 4. Why this is deferred rather than run now

It needs a state readout that predicts value. Without one, `ΔS` is measured in
units that have not been shown to matter, and the chain reduces to "changing
the corpus changes outcomes" — which needs no interpretability at all.

## 5. What it would buy

The bridge from an abstract state coordinate to something actionable about
training data: not "this checkpoint has property X" but "adding this kind of
data moves the model toward states where the next corpus is worth more". That
is the first point at which the programme becomes a training science rather
than checkpoint analysis.
