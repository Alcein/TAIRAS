"""Source-disjoint development state prototypes for DSUS calibration/retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import Field
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from taira_mt.models import StrictModel
from taira_mt.provenance import content_sha256

STATE_PROTOTYPE_BANK_VERSION = "dsus_state_prototype_bank_v1"


class StatePrototype(StrictModel):
    prototype_id: str = Field(min_length=1)
    candidate_key: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    query: str = Field(min_length=1)
    mechanism_label: str = Field(min_length=1)
    disclosure_status: str = Field(min_length=1)
    recommendation_readiness: str = Field(min_length=1)
    unresolved_axis: str = Field(default="")
    source_score: int = Field(ge=0, le=100)


@dataclass
class StatePrototypeBank:
    """Small lexical nearest-prototype bank built only from development sources."""

    prototypes: tuple[StatePrototype, ...]
    vectorizer: TfidfVectorizer
    matrix: csr_matrix
    bank_id: str
    version: str = STATE_PROTOTYPE_BANK_VERSION

    @classmethod
    def build(cls, prototypes: list[StatePrototype]) -> StatePrototypeBank:
        if not prototypes:
            raise ValueError("state prototype bank needs at least one prototype")
        ids = [item.prototype_id for item in prototypes]
        if len(ids) != len(set(ids)):
            raise ValueError("state prototype IDs must be unique")
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            dtype=np.float32,
        )
        matrix = vectorizer.fit_transform([item.query for item in prototypes]).tocsr()
        payload = [item.model_dump(mode="json") for item in prototypes]
        return cls(
            prototypes=tuple(prototypes),
            vectorizer=vectorizer,
            matrix=matrix,
            bank_id=f"state_prototypes::{content_sha256(payload)[:16]}",
        )

    def examples(self, *, domain: str, query: str, k: int = 3) -> list[dict[str, object]]:
        if k < 1:
            raise ValueError("prototype retrieval k must be positive")
        scores = (self.vectorizer.transform([query]) @ self.matrix.T).toarray().ravel()
        candidates = [
            (float(scores[index]), prototype)
            for index, prototype in enumerate(self.prototypes)
            if prototype.domain == domain
        ]
        candidates.sort(key=lambda item: (-item[0], item[1].prototype_id))
        # DSUS needs contrastive examples, not five near-duplicates of the
        # same inferred mechanism. Select the closest source-disjoint example
        # for distinct state labels first, then fill any remaining slots.
        selected: list[tuple[float, StatePrototype]] = []
        seen_labels: set[str] = set()
        for candidate in candidates:
            if candidate[1].mechanism_label not in seen_labels:
                selected.append(candidate)
                seen_labels.add(candidate[1].mechanism_label)
            if len(selected) == k:
                break
        if len(selected) < k:
            selected_ids = {prototype.prototype_id for _, prototype in selected}
            selected.extend(candidate for candidate in candidates if candidate[1].prototype_id not in selected_ids)
        return [
            {
                "query": prototype.query,
                "disclosure_status": prototype.disclosure_status,
                "recommendation_readiness": prototype.recommendation_readiness,
                "unresolved_axis": prototype.unresolved_axis,
                "similarity": round(score, 4),
            }
            for score, prototype in selected[:k]
        ]

    def save(self, path: Path) -> None:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("saving state prototype banks requires ML dependencies") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "version": self.version,
                "prototypes": [item.model_dump(mode="json") for item in self.prototypes],
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "bank_id": self.bank_id,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> StatePrototypeBank:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("loading state prototype banks requires ML dependencies") from exc
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("version") != STATE_PROTOTYPE_BANK_VERSION:
            raise ValueError(f"unsupported state prototype bank: {path}")
        raw = payload.get("prototypes")
        if not isinstance(raw, list) or not isinstance(payload.get("matrix"), csr_matrix):
            raise ValueError("state prototype bank is incomplete")
        prototypes = tuple(StatePrototype.model_validate(item) for item in raw)
        if not prototypes or len(prototypes) != payload["matrix"].shape[0]:
            raise ValueError("state prototype bank prototype/matrix mismatch")
        return cls(
            prototypes=prototypes,
            vectorizer=payload["vectorizer"],
            matrix=payload["matrix"],
            bank_id=str(payload["bank_id"]),
        )
