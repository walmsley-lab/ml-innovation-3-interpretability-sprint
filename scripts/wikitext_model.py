"""Frozen feature and model ladder for the WikiText Stage-5 test.

One module so the ladder cannot differ between the development fit, the
confirmatory prediction freeze and the adaptive scoring. Everything is fixed
before any outcome exists.

Features are computed from **train documents only**, under the frozen family
assignment, and exist for any pair including one never run:

* ``cosine`` — symmetric TF-IDF centroid similarity;
* ``kl`` — ``KL(target || source)`` over unigram distributions, asymmetric,
  which is what lets ``i->j`` differ from ``j->i`` at all.

Ridge throughout, with the penalty chosen by inner leave-one-pair-out
**inside the development pool only**. The ladder stops at 15 parameters
against 26 development relationships. No composition-slot interactions, no
pair-identity terms: on this project a 7-parameter model fitted on 8 points
scored best by LOPO and worst prospectively.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

CORPUS = Path("artifacts/corpus_v2")
MODELS = ("global", "source_only", "target_only", "additive", "cosine", "relational")
RIDGE_GRID = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


@lru_cache(maxsize=1)
def families() -> tuple:
    return tuple(json.loads((CORPUS / "frozen_pools.json").read_text())["families_used"])


@lru_cache(maxsize=1)
def features() -> dict:
    """Directional and symmetric relationships, from train documents only."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_corpus_intake import load_wikitext
    from sklearn.feature_extraction.text import TfidfVectorizer

    from dsi.corpus import deduplicate, split_documents
    from dsi.natural import whitespace_tokens

    used = families()
    assignments = json.loads((CORPUS / "assignments_k8.json").read_text())
    dedup, _ = deduplicate(load_wikitext(Path("data/wikitext103_train.parquet")))
    splits = split_documents(dedup, seed=0, fractions=(0.7, 0.15, 0.15))

    texts = {f: [] for f in used}
    for d in splits["train"].documents:
        f = assignments["train"].get(d.doc_id)
        if f in texts:
            texts[f].append(d.text)

    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
    matrix = vectorizer.fit_transform([" ".join(texts[f]) for f in used])
    centroids = np.asarray(matrix.todense())
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    cosine = centroids @ centroids.T

    counts = {}
    for f in used:
        c = {}
        for text in texts[f]:
            for piece in whitespace_tokens(text):
                c[piece] = c.get(piece, 0) + 1
        counts[f] = c
    keys = sorted({k for c in counts.values() for k in c})
    probs = {}
    for f in used:
        v = np.array([counts[f].get(k, 0) for k in keys], dtype=float) + 1.0
        probs[f] = v / v.sum()

    out = {}
    for a, i in enumerate(used):
        for b, j in enumerate(used):
            if i == j:
                continue
            out[(i, j)] = {
                "cosine": float(cosine[a, b]),
                "kl": float(np.sum(probs[j] * np.log(probs[j] / probs[i]))),
            }
    return out


def design(pairs, kind: str) -> np.ndarray:
    used, feats, rows = families(), features(), []
    for source, target in pairs:
        f = feats[(source, target)]
        if kind == "global":
            rows.append([1.0])
        elif kind == "source_only":
            rows.append([1.0] + [1.0 if source == g else 0.0 for g in used[1:]])
        elif kind == "target_only":
            rows.append([1.0] + [1.0 if target == g else 0.0 for g in used[1:]])
        elif kind == "additive":
            rows.append([1.0]
                        + [1.0 if source == g else 0.0 for g in used[1:]]
                        + [1.0 if target == g else 0.0 for g in used[1:]])
        elif kind == "cosine":
            rows.append([1.0, f["cosine"]])
        elif kind == "relational":
            rows.append([1.0]
                        + [1.0 if source == g else 0.0 for g in used[1:]]
                        + [1.0 if target == g else 0.0 for g in used[1:]]
                        + [f["cosine"], f["kl"]])
        else:
            raise ValueError(kind)
    return np.array(rows, dtype=float)


def ridge_fit(X: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    """Ridge with the intercept column left unpenalized."""
    p = X.shape[1]
    reg = np.eye(p) * penalty
    reg[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def lopo_predictions(pairs, y: np.ndarray, kind: str, penalty: float) -> np.ndarray:
    X = design(pairs, kind)
    out = np.empty(len(pairs))
    for i in range(len(pairs)):
        keep = [j for j in range(len(pairs)) if j != i]
        out[i] = X[i] @ ridge_fit(X[keep], y[keep], penalty)
    return out


def select_penalty(pairs, y: np.ndarray, kind: str) -> tuple:
    """Choose the ridge penalty by inner LOPO, inside development only."""
    scores = {}
    for penalty in RIDGE_GRID:
        try:
            pred = lopo_predictions(pairs, y, kind, penalty)
        except np.linalg.LinAlgError:
            continue
        scores[penalty] = float(np.sqrt(np.mean((pred - y) ** 2)))
    best = min(scores, key=scores.get)
    return best, scores[best], scores


def fit_ladder(pairs, y: np.ndarray) -> dict:
    out = {}
    for kind in MODELS:
        penalty, rmse, grid = select_penalty(pairs, y, kind)
        X = design(pairs, kind)
        beta = ridge_fit(X, y, penalty)
        out[kind] = {"penalty": penalty, "lopo_rmse": rmse,
                     "n_parameters": int(X.shape[1]),
                     "coefficients": beta.tolist(), "penalty_grid": grid}
    return out
