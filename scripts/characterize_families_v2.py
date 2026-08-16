"""Is WikiText family 6 an intervention family or a generic residual cluster?

Outcome-blind: no transfer has been run on this substrate and the proposal is
not changed by anything here. The question is whether a cluster holding 60% of
the training documents is a coherent developmental family or the leftover bin
k-means produces when the rest of the space is carved off. If it is residual,
using it as a source measures "generic text" and using it inside a control
makes the control generic too.

Diagnostics, all computable without any outcome:

* size, tokens, document length;
* representative terms by centroid weight, and by **distinctiveness** —
  a term's weight in this family relative to its mean across families, which
  separates "this family is about X" from "X is common everywhere";
* centroid cohesion: mean cosine of member documents to their own centroid;
* separation: cosine between family centroids, and each family's margin
  between its own centroid and the nearest other centroid.

A residual cluster shows low cohesion, low distinctiveness, and a small
margin: its documents are not near each other, they are merely not near
anything else.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dsi.corpus import deduplicate, split_documents
from dsi.natural import whitespace_tokens

OUT = Path("artifacts/corpus_v2")


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_corpus_intake import load_wikitext
    from sklearn.feature_extraction.text import TfidfVectorizer

    assignments = json.loads((OUT / "assignments_k8.json").read_text())
    dedup, _ = deduplicate(load_wikitext(Path("data/wikitext103_train.parquet")))
    splits = split_documents(dedup, seed=0, fractions=(0.7, 0.15, 0.15))
    families = sorted({int(v) for v in assignments["train"].values()})

    docs = {f: [] for f in families}
    for d in splits["train"].documents:
        f = assignments["train"].get(d.doc_id)
        if f is not None:
            docs[f].append(d)

    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
    all_docs = [d for f in families for d in docs[f]]
    labels = np.array([f for f in families for _ in docs[f]])
    matrix = vectorizer.fit_transform([d.text for d in all_docs])
    terms = np.array(vectorizer.get_feature_names_out())

    centroids = np.vstack([np.asarray(matrix[labels == f].mean(axis=0)) for f in families])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    similarity = centroids @ centroids.T
    mean_weight = centroids.mean(axis=0) + 1e-12

    print(f"\n{'fam':>4s} {'docs':>6s} {'tokens':>10s} {'med len':>8s} {'cohesion':>9s} "
          f"{'margin':>7s} {'nearest':>8s}")
    rows = {}
    for i, f in enumerate(families):
        members = matrix[labels == f]
        norms = np.sqrt(members.multiply(members).sum(axis=1)).A.ravel() + 1e-12
        cohesion = float(np.mean((members @ centroids[i]).ravel() / norms))
        others = [similarity[i, j] for j in range(len(families)) if j != i]
        nearest = int(np.argmax([similarity[i, j] if j != i else -1
                                 for j in range(len(families))]))
        margin = float(1.0 - max(others))
        lengths = np.array([len(whitespace_tokens(d.text)) for d in docs[f]])
        rows[f] = {"documents": len(docs[f]), "tokens": int(lengths.sum()),
                   "median_length": float(np.median(lengths)), "cohesion": cohesion,
                   "margin_to_nearest": margin, "nearest_family": families[nearest],
                   "max_similarity": float(max(others))}
        print(f"{f:>4d} {len(docs[f]):>6d} {int(lengths.sum()):>10d} "
              f"{np.median(lengths):>8.0f} {cohesion:>9.3f} {margin:>7.3f} "
              f"{families[nearest]:>8d}")

    print("\nrepresentative terms (top by centroid weight | top by distinctiveness):")
    for i, f in enumerate(families):
        top = terms[np.argsort(-centroids[i])[:8]]
        distinct = terms[np.argsort(-(centroids[i] / mean_weight))[:8]]
        rows[f]["top_terms"] = list(top)
        rows[f]["distinctive_terms"] = list(distinct)
        print(f"  f{f}: {', '.join(top[:6])}")
        print(f"      distinctive: {', '.join(distinct[:6])}")

    cohesions = {f: rows[f]["cohesion"] for f in families}
    margins = {f: rows[f]["margin_to_nearest"] for f in families}
    suspect = min(cohesions, key=cohesions.get)
    print(f"\nlowest cohesion: family {suspect} ({cohesions[suspect]:.3f}); "
          f"median across families {np.median(list(cohesions.values())):.3f}")
    print(f"smallest margin: family {min(margins, key=margins.get)} "
          f"({min(margins.values()):.3f})")

    (OUT / "family_characterization_k8.json").write_text(json.dumps(
        {"per_family": rows,
         "centroid_similarity": similarity.tolist(),
         "lowest_cohesion_family": suspect}, indent=2, default=str) + "\n")
    print(f"\nwrote {OUT / 'family_characterization_k8.json'}")


if __name__ == "__main__":
    main()
