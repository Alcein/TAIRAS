"""Shared full-catalog lexical retrieval for TAIRAS and every baseline.

The frozen benchmark evaluates interaction quality, but its difficulty still
depends on converting a rich public request into catalogue-grounded product
attributes.  This module deliberately makes that conversion explicit through
``SearchSpec`` and gives every compared system the same full-domain retrieval
surface.  It contains no benchmark private state, target item, or evaluator
label.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field, field_validator
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from taira_mt.models import StrictModel
from taira_mt.provenance import content_sha256
from taira_mt.recommender.catalog import CandidatePool

LEXICAL_RETRIEVER_VERSION = "mt_agent_full_catalog_tfidf_v1"
_SPACE_RE = re.compile(r"\s+")


def _clean_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", html.unescape(str(value or "")).strip())


def _normalized_terms(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        text = _clean_text(value)
        folded = text.casefold()
        if text and folded not in seen:
            seen.add(folded)
            normalized.append(text)
    return normalized


class SearchSpec(StrictModel):
    """A public, auditable conversion from need language to catalog search."""

    query: str = Field(min_length=1)
    target_type: str = ""
    must_terms: list[str] = Field(default_factory=list)
    preferred_terms: list[str] = Field(default_factory=list)
    avoid_terms: list[str] = Field(default_factory=list)
    excluded_item_ids: list[str] = Field(default_factory=list)
    candidate_k: int = Field(default=100, ge=10, le=1000)
    ranking_k: int = Field(default=10, ge=1, le=50)

    @field_validator("query", "target_type", mode="before")
    @classmethod
    def clean_scalar_text(cls, value: Any) -> str:
        return _clean_text(value)

    @field_validator("must_terms", "preferred_terms", "avoid_terms", "excluded_item_ids", mode="before")
    @classmethod
    def clean_term_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("search terms must be a list")
        return _normalized_terms(str(item) for item in value)

    @classmethod
    def from_public_query(cls, query: str, *, ranking_k: int = 10) -> SearchSpec:
        """Create the equal-tool no-planning baseline search request."""

        return cls(query=query, ranking_k=ranking_k)

    def rendered_query(self) -> str:
        """Weight Agent-derived constraints without hiding the original request."""

        terms: list[str] = [self.query]
        if self.target_type:
            terms.extend([self.target_type, self.target_type])
        for term in self.must_terms:
            terms.extend([term, term, term])
        for term in self.preferred_terms:
            terms.extend([term, term])
        return " ".join(terms)


class CatalogRetrievalItem(StrictModel):
    """One public ranked result with its observable matching evidence."""

    item_id: str
    title: str
    category: str = ""
    score: float
    matched_must_terms: list[str] = Field(default_factory=list)
    matched_preferred_terms: list[str] = Field(default_factory=list)
    matched_avoid_terms: list[str] = Field(default_factory=list)


class CatalogRetrievalResult(StrictModel):
    """A reproducible public retrieval result for one SearchSpec."""

    retriever_id: str
    domain: str
    pool_id: str
    search_spec: SearchSpec
    results: list[CatalogRetrievalItem]


@dataclass
class LexicalCatalogRetriever:
    """Field-weighted TF-IDF retrieval over one complete public catalog.

    The retriever is intentionally lightweight and deterministic.  It is the
    common floor for the one-shot baseline, direct tool Agent, ReAct Agent,
    static TAIRA adaptation, and TAIRAS.  A later semantic reranker can be
    inserted after this stage only as a shared post-retrieval component.
    """

    domain: str
    pool_id: str
    item_ids: tuple[str, ...]
    titles: tuple[str, ...]
    categories: tuple[str, ...]
    field_texts: tuple[str, ...]
    vectorizer: TfidfVectorizer
    matrix: csr_matrix
    retriever_id: str
    version: str = LEXICAL_RETRIEVER_VERSION

    @classmethod
    def build(
        cls,
        candidate_pool: CandidatePool,
        *,
        max_features: int = 120_000,
        min_df: int | float = 2,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> LexicalCatalogRetriever:
        """Build an index from public product title/category/description fields."""

        if max_features < 1:
            raise ValueError("max_features must be positive")
        item_ids = candidate_pool.item_ids
        titles: list[str] = []
        categories: list[str] = []
        field_texts: list[str] = []
        for item_id in item_ids:
            item = candidate_pool.index.get(item_id)
            title = _clean_text(item.title)
            category = _clean_text(item.attributes.get("category", ""))
            description = _clean_text(item.attributes.get("description", ""))
            # Title/category repetition is a transparent field weighting
            # choice, rather than an evaluator-derived item score.
            field_text = " ".join(
                value
                for value in (title, title, title, category, category, description)
                if value
            )
            titles.append(title)
            categories.append(category)
            field_texts.append(field_text or "catalog item")
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        matrix = vectorizer.fit_transform(field_texts).tocsr()
        identity = content_sha256(
            {
                "version": LEXICAL_RETRIEVER_VERSION,
                "domain": candidate_pool.index.domain,
                "pool_id": candidate_pool.pool_id,
                "max_features": max_features,
                "min_df": min_df,
                "ngram_range": list(ngram_range),
            }
        )
        return cls(
            domain=candidate_pool.index.domain,
            pool_id=candidate_pool.pool_id,
            item_ids=tuple(item_ids),
            titles=tuple(titles),
            categories=tuple(categories),
            field_texts=tuple(field_texts),
            vectorizer=vectorizer,
            matrix=matrix,
            retriever_id=f"lexical_catalog::{identity}",
        )

    @staticmethod
    def _term_matches(text: str, terms: Iterable[str]) -> list[str]:
        folded = text.casefold()
        return [term for term in terms if term.casefold() in folded]

    def _scores(self, spec: SearchSpec) -> np.ndarray:
        query_vector = self.vectorizer.transform([spec.rendered_query()])
        # SciPy keeps the 1 x N multiplication sparse; make only this score
        # row dense (roughly 2.5 MB even for the largest current catalog).
        scores = (query_vector @ self.matrix.T).toarray().ravel().astype(np.float64, copy=False)
        for index, text in enumerate(self.field_texts):
            must_matches = self._term_matches(text, spec.must_terms)
            preferred_matches = self._term_matches(text, spec.preferred_terms)
            avoid_matches = self._term_matches(text, spec.avoid_terms)
            if must_matches:
                scores[index] += 0.12 * len(must_matches)
            if preferred_matches:
                scores[index] += 0.04 * len(preferred_matches)
            if avoid_matches:
                scores[index] -= 0.20 * len(avoid_matches)
        return scores

    def search(self, spec: SearchSpec) -> CatalogRetrievalResult:
        """Rank public items and return the Top-k evidence surface."""

        scores = self._scores(spec)
        excluded_item_ids = set(spec.excluded_item_ids)
        if excluded_item_ids:
            scores[
                np.fromiter(
                    (item_id in excluded_item_ids for item_id in self.item_ids),
                    dtype=bool,
                    count=len(self.item_ids),
                )
            ] = -np.inf
        available_count = len(self.item_ids) - sum(item_id in excluded_item_ids for item_id in self.item_ids)
        candidate_count = min(spec.candidate_k, available_count)
        if candidate_count < 1:
            return CatalogRetrievalResult(
                retriever_id=self.retriever_id,
                domain=self.domain,
                pool_id=self.pool_id,
                search_spec=spec,
                results=[],
            )
        candidate_indices = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
        ordered_indices = sorted(
            candidate_indices.tolist(),
            key=lambda index: (-float(scores[index]), self.item_ids[index]),
        )[: spec.ranking_k]
        results: list[CatalogRetrievalItem] = []
        for index in ordered_indices:
            text = self.field_texts[index]
            results.append(
                CatalogRetrievalItem(
                    item_id=self.item_ids[index],
                    title=self.titles[index],
                    category=self.categories[index],
                    score=float(scores[index]),
                    matched_must_terms=self._term_matches(text, spec.must_terms),
                    matched_preferred_terms=self._term_matches(text, spec.preferred_terms),
                    matched_avoid_terms=self._term_matches(text, spec.avoid_terms),
                )
            )
        return CatalogRetrievalResult(
            retriever_id=self.retriever_id,
            domain=self.domain,
            pool_id=self.pool_id,
            search_spec=spec,
            results=results,
        )

    def save(self, path: Path) -> None:
        """Persist a reusable full-catalog index without benchmark labels."""

        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - dependency failure only
            raise RuntimeError("saving retrieval indexes requires the ML dependencies") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "version": self.version,
                "domain": self.domain,
                "pool_id": self.pool_id,
                "item_ids": self.item_ids,
                "titles": self.titles,
                "categories": self.categories,
                "field_texts": self.field_texts,
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "retriever_id": self.retriever_id,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> LexicalCatalogRetriever:
        """Load a cached index and reject incompatible/corrupt payloads."""

        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - dependency failure only
            raise RuntimeError("loading retrieval indexes requires the ML dependencies") from exc
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("version") != LEXICAL_RETRIEVER_VERSION:
            raise ValueError(f"unsupported lexical retriever artifact: {path}")
        required = {
            "domain",
            "pool_id",
            "item_ids",
            "titles",
            "categories",
            "field_texts",
            "vectorizer",
            "matrix",
            "retriever_id",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"lexical retriever artifact is incomplete: {missing}")
        if not isinstance(payload["matrix"], csr_matrix):
            raise ValueError("lexical retriever matrix must be CSR")
        item_ids = tuple(payload["item_ids"])
        titles = tuple(payload["titles"])
        categories = tuple(payload["categories"])
        field_texts = tuple(payload["field_texts"])
        if not item_ids or not (
            len(item_ids) == len(titles) == len(categories) == len(field_texts) == payload["matrix"].shape[0]
        ):
            raise ValueError("lexical retriever item metadata does not align with its matrix")
        return cls(
            domain=str(payload["domain"]),
            pool_id=str(payload["pool_id"]),
            item_ids=item_ids,
            titles=titles,
            categories=categories,
            field_texts=field_texts,
            vectorizer=payload["vectorizer"],
            matrix=payload["matrix"],
            retriever_id=str(payload["retriever_id"]),
        )
