"""Source-disjoint learned prior over initial decision-sufficient states.

This small model is deliberately not an item recommender and it never reads a
frozen test trajectory at runtime.  It learns the *interaction shape* of a
legacy complex query (direct / one clarification / ambiguity / trade-off /
initial probe) from source-disjoint development examples.  DSUS may expose the
prior to a constrained language model as calibration evidence, while the
language model still owns the public retrieval frame and wording.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import Field
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from taira_mt.models import StrictModel
from taira_mt.provenance import content_sha256

DSUS_INITIAL_PRIOR_VERSION = "dsus_initial_state_prior_v1"
STATE_FAMILIES = ("direct", "progressive", "ambiguous", "revision", "conflict")


class DSUSPriorExample(StrictModel):
    """One development-only source query with a weak state-family label."""

    candidate_key: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    query: str = Field(min_length=1)
    state_family: str = Field(min_length=1)
    source_score: int = Field(ge=0, le=100)


class DSUSPriorCandidate(StrictModel):
    state_family: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)


class DSUSPriorPrediction(StrictModel):
    model_id: str = Field(min_length=1)
    top_state_family: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    candidates: list[DSUSPriorCandidate] = Field(min_length=1)


def _model_text(*, domain: str, query: str) -> str:
    """Keep domain visible but ensure the model is driven by public text."""

    return f"domain={domain.removeprefix('amazon_')} {query}"


@dataclass
class DSUSInitialStatePrior:
    """Word+character TF-IDF state-family classifier with transparent output."""

    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    model_id: str
    version: str = DSUS_INITIAL_PRIOR_VERSION

    @classmethod
    def fit(cls, examples: Iterable[DSUSPriorExample]) -> DSUSInitialStatePrior:
        materialized = list(examples)
        if len(materialized) < len(STATE_FAMILIES):
            raise ValueError("DSUS initial-state prior needs examples for all state families")
        labels = [item.state_family for item in materialized]
        if set(labels) != set(STATE_FAMILIES):
            raise ValueError("DSUS initial-state prior requires every configured state family")
        texts = [_model_text(domain=item.domain, query=item.query) for item in materialized]
        word = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
            max_features=20_000,
            dtype=np.float32,
        )
        char = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=30_000,
            dtype=np.float32,
        )
        features = hstack((word.fit_transform(texts), char.fit_transform(texts)), format="csr")
        classifier = LogisticRegression(
            C=1.25,
            class_weight="balanced",
            max_iter=2_000,
            random_state=17,
        )
        classifier.fit(features, labels)
        fingerprint = content_sha256(
            {
                "version": DSUS_INITIAL_PRIOR_VERSION,
                "examples": [item.model_dump(mode="json") for item in materialized],
                "classes": list(classifier.classes_),
            }
        )
        return cls(
            word_vectorizer=word,
            char_vectorizer=char,
            classifier=classifier,
            model_id=f"dsus_initial_tfidf_prior::{fingerprint[:16]}",
        )

    def _features(self, *, domain: str, query: str):
        text = _model_text(domain=domain, query=query)
        return hstack(
            (self.word_vectorizer.transform([text]), self.char_vectorizer.transform([text])),
            format="csr",
        )

    def predict(self, *, domain: str, query: str) -> DSUSPriorPrediction:
        probabilities = self.classifier.predict_proba(self._features(domain=domain, query=query))[0]
        candidates = sorted(
            (
                DSUSPriorCandidate(state_family=str(label), probability=float(probability))
                for label, probability in zip(self.classifier.classes_, probabilities, strict=True)
            ),
            key=lambda item: (-item.probability, item.state_family),
        )
        best = candidates[0]
        return DSUSPriorPrediction(
            model_id=self.model_id,
            top_state_family=best.state_family,
            confidence=best.probability,
            candidates=candidates,
        )

    def save(self, path: Path) -> None:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("saving a DSUS initial-state prior requires ML dependencies") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "version": self.version,
                "word_vectorizer": self.word_vectorizer,
                "char_vectorizer": self.char_vectorizer,
                "classifier": self.classifier,
                "model_id": self.model_id,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> DSUSInitialStatePrior:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("loading a DSUS initial-state prior requires ML dependencies") from exc
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("version") != DSUS_INITIAL_PRIOR_VERSION:
            raise ValueError(f"unsupported DSUS initial-state prior: {path}")
        classifier = payload.get("classifier")
        if not isinstance(classifier, LogisticRegression):
            raise ValueError("DSUS initial-state prior is missing its classifier")
        if set(str(item) for item in classifier.classes_) != set(STATE_FAMILIES):
            raise ValueError("DSUS initial-state prior has an unexpected label set")
        return cls(
            word_vectorizer=payload["word_vectorizer"],
            char_vectorizer=payload["char_vectorizer"],
            classifier=classifier,
            model_id=str(payload["model_id"]),
        )
