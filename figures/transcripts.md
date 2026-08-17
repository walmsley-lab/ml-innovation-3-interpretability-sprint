### What the model actually sees

Every stream shares one template — `the <entity> is <value> .` — so no
stream is identifiable from surface form. Only the *relationship* differs.

**`BIND`** — the queried entity **appears earlier**; the answer must be retrieved from context

```
prompt : <bos> the e276 is v54 . the e414 is v22 . the e447 is v1 . the e364 is v9 . the e110 is v8 . the e170 is v19 . the e383 is v60 . the e276 is
target : v54
```

**`FACT`** — the queried entity does **not** appear earlier; the answer is a globally fixed association held in the weights

```
prompt : <bos> the e276 is v54 . the e414 is v22 . the e447 is v1 . the e364 is v9 . the e110 is v8 . the e170 is v19 . the e383 is v60 . the e58 is
target : v41
```

**`BINDT`** — as BIND, but the answer is a fixed permutation of the bound value — retrieval alone gives the wrong token

```
prompt : <bos> the e276 is v54 . the e414 is v22 . the e447 is v1 . the e364 is v9 . the e110 is v8 . the e170 is v19 . the e383 is v60 . the e276 is
target : v33
```

### The same prompt across training histories

One example is not evidence at these accuracies, so the table reports 256
BIND prompts. Neither model has had **any** target-phase training: this is
zero-shot.

| history | exact answer correct | prediction is a value from the context |
|---|---|---|
| `A` | 0.113 | 1.000 |
| `A_prime` | 0.008 | 0.133 |
| `BG` | 0.004 | 0.074 |
| *chance* | 0.016 | 0.109 |

The second column is the more mechanistic one. Getting the exact binding
right is hard; **restricting the answer to values that appear in the context**
is the retrieval behaviour itself, and it separates the histories much more
sharply than exact accuracy does.

A single illustrative prompt, chosen as the first of the sample (not for
outcome). The correct answer is the value bound to the queried entity
earlier in the same context:

```
prompt : ... the e276 is ___
correct: v29

context values available: v29 v55 v16 v52 v21 v36 v62

A        wrong    top-3: v55 0.17  v21 0.17  v16 0.13   (3/3 drawn from context)
A_prime  wrong    top-3: v46 0.02  v60 0.02  v32 0.02   (0/3 drawn from context)
BG       wrong    top-3: v50 0.02  v61 0.02  v53 0.02   (0/3 drawn from context)
```

On this example every model gets the exact value wrong. What differs is
*where the guesses come from*, which is what the table quantifies.

