"""Arbitrary-corpus path: documents, splits, and family proposal.

The ordering here is the whole point. Documents are split into train,
validation and test **before** any family proposal happens, the proposer is
fitted on train only and then frozen, and held-out documents are assigned by
the frozen proposer. A proposer fitted on all documents has seen the
evaluation set, and every family it proposes is contaminated.

Nothing here claims arbitrary-corpus optimization. This is the minimum needed
to ask whether the frozen pipeline transfers to an unseen natural corpus,
which is Gate J and a later stage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Document", "Corpus", "load_jsonl", "load_text_dir", "deduplicate",
           "split_documents", "FamilyProposal", "TfidfLsaProposer",
           "token_budget_sample", "audit_proposal", "ProposalAudit"]


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    metadata: tuple = ()

    @staticmethod
    def of(text: str, metadata=()) -> "Document":
        # Content-addressed, so the same document has the same id in every run.
        return Document(hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest(),
                        text, tuple(sorted(metadata)))


@dataclass(frozen=True)
class Corpus:
    name: str
    version: str
    documents: tuple = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.documents)

    def subset(self, ids) -> "Corpus":
        keep = set(ids)
        return Corpus(self.name, self.version,
                      tuple(d for d in self.documents if d.doc_id in keep))


def load_jsonl(path, name: str, version: str, text_field: str = "text") -> Corpus:
    docs = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                docs.append(Document.of(json.loads(line)[text_field]))
    return Corpus(name, version, tuple(docs))


def load_text_dir(path, name: str, version: str, pattern: str = "*.txt") -> Corpus:
    return Corpus(name, version, tuple(
        Document.of(p.read_text(errors="replace")) for p in sorted(Path(path).glob(pattern))))


def deduplicate(corpus: Corpus) -> tuple:
    """Collapse content-identical documents BEFORE splitting.

    ``Document.of`` hashes text, so two identical documents share an id. If
    both survive into different splits, the "held-out" split contains a
    document the proposer was fitted on, and document-level isolation is
    silently broken. Deduplicating first makes the ids unique by construction.

    Returns the deduplicated corpus and the statistics, which are recorded
    rather than discarded: a corpus that is 40% duplicates is a different
    object from one that is 2%.
    """
    seen, unique = set(), []
    for document in corpus.documents:
        if document.doc_id not in seen:
            seen.add(document.doc_id)
            unique.append(document)
    stats = {"input_documents": len(corpus.documents), "unique_documents": len(unique),
             "duplicates_removed": len(corpus.documents) - len(unique)}
    stats["duplicate_fraction"] = (stats["duplicates_removed"] / len(corpus.documents)
                                   if corpus.documents else 0.0)
    return Corpus(corpus.name, f"{corpus.version}+dedup", tuple(unique)), stats


def split_documents(corpus: Corpus, *, seed: int = 0,
                    fractions=(0.7, 0.15, 0.15)) -> dict:
    """Document-level train/val/test split. Runs BEFORE family proposal.

    Splitting after proposal, or proposing on all documents, leaks the
    evaluation set into the family definitions themselves.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    ids = np.array([d.doc_id for d in corpus.documents])
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(
            "corpus contains content-duplicate documents; call deduplicate() "
            "before splitting or the same text can land in two splits")
    order = np.random.default_rng(seed).permutation(len(ids))
    n_train = int(round(fractions[0] * len(ids)))
    n_val = int(round(fractions[1] * len(ids)))
    return {"train": corpus.subset(ids[order[:n_train]]),
            "val": corpus.subset(ids[order[n_train:n_train + n_val]]),
            "test": corpus.subset(ids[order[n_train + n_val:]])}


@dataclass(frozen=True)
class FamilyProposal:
    """Frozen assignment of documents to proposed families."""
    proposer: str
    n_families: int
    assignments: dict          # doc_id -> family index
    fitted_on: str

    def family_of(self, doc_id: str):
        return self.assignments.get(doc_id)

    def sizes(self) -> dict:
        out = {}
        for f in self.assignments.values():
            out[f] = out.get(f, 0) + 1
        return out


class TfidfLsaProposer:
    """TF-IDF + LSA + k-means. One proposer, fitted once, then frozen."""

    def __init__(self, n_families: int = 6, n_components: int = 64, seed: int = 0):
        self.n_families, self.n_components, self.seed = n_families, n_components, seed
        self._pipeline = None

    def fit(self, corpus: Corpus) -> "TfidfLsaProposer":
        from sklearn.cluster import KMeans
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline

        vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
        matrix = vectorizer.fit_transform([d.text for d in corpus.documents])
        components = min(self.n_components, max(2, matrix.shape[1] - 1))
        svd = TruncatedSVD(n_components=components, random_state=self.seed)
        reduced = svd.fit_transform(matrix)
        kmeans = KMeans(n_clusters=self.n_families, random_state=self.seed, n_init=10)
        kmeans.fit(reduced)
        self._pipeline = (vectorizer, svd, kmeans)
        return self

    def assign(self, corpus: Corpus, fitted_on: str) -> FamilyProposal:
        """Assign documents using the frozen proposer. Never refits."""
        if self._pipeline is None:
            raise RuntimeError("proposer must be fitted on the train split first")
        vectorizer, svd, kmeans = self._pipeline
        labels = kmeans.predict(svd.transform(
            vectorizer.transform([d.text for d in corpus.documents])))
        return FamilyProposal(
            proposer=f"tfidf_lsa_kmeans_k{self.n_families}_seed{self.seed}",
            n_families=self.n_families,
            assignments={d.doc_id: int(l) for d, l in zip(corpus.documents, labels)},
            fitted_on=fitted_on,
        )


def token_budget_sample(corpus: Corpus, proposal: FamilyProposal, *,
                        token_budget: int, tokenize, family: int | None = None,
                        exclude_family: int | None = None, seed: int = 0) -> dict:
    """Draw documents up to an exact token budget.

    Matching document *counts* does not match training budgets: documents
    differ in length, so equal counts give unequal exposure and the transfer
    estimate becomes partly a measurement of how long the source documents
    happened to be. The budget is counted in tokenizer tokens and the final
    document is truncated so the total is exact.

    Used for both arms. ``family=i`` draws the source D_i;
    ``exclude_family=j`` draws the neutral control N, which must exclude the
    target — a background containing D_j would measure exposure to the
    target rather than a neutral prefix.
    """
    if (family is None) == (exclude_family is None):
        raise ValueError("pass exactly one of family or exclude_family")
    eligible = [d for d in corpus.documents
                if (proposal.family_of(d.doc_id) == family if family is not None
                    else proposal.family_of(d.doc_id) not in (None, exclude_family))]
    if not eligible:
        raise ValueError(f"no eligible documents (family={family}, "
                         f"exclude_family={exclude_family})")

    order = np.random.default_rng(seed).permutation(len(eligible))
    chosen, tokens, total = [], [], 0
    for i in order:
        document = eligible[i]
        piece = list(tokenize(document.text))
        if total + len(piece) >= token_budget:
            piece = piece[: token_budget - total]
            chosen.append(document); tokens.extend(piece); total = token_budget
            break
        chosen.append(document); tokens.extend(piece); total += len(piece)

    if total < token_budget:
        raise ValueError(
            f"eligible pool holds {total} tokens, short of the {token_budget} "
            "budget; the arms cannot be budget-matched from this corpus")
    return {"documents": tuple(chosen), "tokens": tokens, "n_tokens": total,
            "n_documents": len(chosen), "budget": token_budget}


# --- 3. frozen-proposer audit --------------------------------------------


@dataclass(frozen=True)
class ProposalAudit:
    """Per-family support under the frozen proposer, on every split."""
    per_family: dict
    usable_families: tuple
    unusable_families: tuple
    ok: bool

    def render(self) -> str:
        lines = [f"{'family':>7s} {'train docs':>11s} {'train tok':>10s} "
                 f"{'heldout docs':>13s} {'heldout tok':>12s}  usable"]
        for family, row in sorted(self.per_family.items()):
            lines.append(
                f"{family:>7d} {row['train_documents']:>11d} {row['train_tokens']:>10d} "
                f"{row['heldout_documents']:>13d} {row['heldout_tokens']:>12d}  "
                f"{'yes' if family in self.usable_families else 'NO: ' + row['reason']}")
        return "\n".join(lines)


def audit_proposal(train: Corpus, heldout: Corpus, train_proposal: FamilyProposal,
                   heldout_proposal: FamilyProposal, *, tokenize,
                   min_train_tokens: int, min_heldout_documents: int = 3) -> ProposalAudit:
    """Refuse to run transfer on families that cannot support it.

    A family with almost no training tokens cannot be a source, and one with
    almost no held-out documents cannot be a target: its acquisition signal
    would be measured on a handful of documents. Catching that here is much
    cheaper than discovering it in a transfer matrix full of noise.
    """
    per_family, usable, unusable = {}, [], []
    for family in range(train_proposal.n_families):
        train_docs = [d for d in train.documents
                      if train_proposal.family_of(d.doc_id) == family]
        held_docs = [d for d in heldout.documents
                     if heldout_proposal.family_of(d.doc_id) == family]
        train_tokens = sum(len(list(tokenize(d.text))) for d in train_docs)
        held_tokens = sum(len(list(tokenize(d.text))) for d in held_docs)
        reason = ""
        if train_tokens < min_train_tokens:
            reason = f"train tokens {train_tokens} < {min_train_tokens}"
        elif len(held_docs) < min_heldout_documents:
            reason = f"held-out docs {len(held_docs)} < {min_heldout_documents}"
        per_family[family] = {
            "train_documents": len(train_docs), "train_tokens": train_tokens,
            "heldout_documents": len(held_docs), "heldout_tokens": held_tokens,
            "reason": reason}
        (usable if not reason else unusable).append(family)
    return ProposalAudit(per_family, tuple(usable), tuple(unusable), not unusable)
