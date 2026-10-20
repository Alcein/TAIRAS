"""Focused tests for the shared paper-canonical catalog retrieval layer."""

from __future__ import annotations

from pathlib import Path

from taira_mt.models import CatalogItem
from taira_mt.mt_agent.retrieval import LexicalCatalogRetriever, SearchSpec
from taira_mt.recommender.catalog import CatalogIndex


def _pool():
    index = CatalogIndex(
        [
            CatalogItem(
                item_id="fit-jeans",
                title="KingSize Men's Big & Tall Mid-Rise Jeans",
                attributes={
                    "category": "Jeans | Clothing | Men | Big & Tall",
                    "description": "Casual mid-rise denim jeans with a 40-inch waist option.",
                },
            ),
            CatalogItem(
                item_id="wrong-jeans",
                title="Women's Junior Skinny Jeans",
                attributes={
                    "category": "Jeans | Clothing | Women",
                    "description": "Fashion skinny jeans for juniors.",
                },
            ),
            CatalogItem(
                item_id="shirt",
                title="Men's Big & Tall Casual Shirt",
                attributes={
                    "category": "Shirts | Clothing | Men | Big & Tall",
                    "description": "A casual shirt, not jeans.",
                },
            ),
        ],
        domain="amazon_clothing",
    )
    return index.pool(recipe_id="test_full_catalog")


def test_search_spec_deduplicates_public_terms() -> None:
    spec = SearchSpec(
        query="Find big and tall jeans",
        must_terms=["Big & Tall", "big & tall", "Jeans"],
        preferred_terms=["mid-rise", "MID-RISE"],
    )
    assert spec.must_terms == ["Big & Tall", "Jeans"]
    assert spec.preferred_terms == ["mid-rise"]


def test_lexical_retriever_grounds_attribute_terms() -> None:
    retriever = LexicalCatalogRetriever.build(_pool(), min_df=1, max_features=1_000)
    result = retriever.search(
        SearchSpec(
            query="men's jeans for casual use",
            target_type="jeans",
            must_terms=["Big & Tall", "Jeans"],
            preferred_terms=["mid-rise", "40-inch waist"],
            avoid_terms=["women"],
            ranking_k=2,
        )
    )
    assert result.results[0].item_id == "fit-jeans"
    assert result.results[0].matched_must_terms == ["Big & Tall", "Jeans"]
    assert "mid-rise" in result.results[0].matched_preferred_terms


def test_retriever_round_trips_without_benchmark_metadata(tmp_path: Path) -> None:
    retriever = LexicalCatalogRetriever.build(_pool(), min_df=1, max_features=1_000)
    path = tmp_path / "lexical_index.joblib"
    retriever.save(path)
    loaded = LexicalCatalogRetriever.load(path)
    result = loaded.search(SearchSpec.from_public_query("big tall jeans"))
    assert loaded.retriever_id == retriever.retriever_id
    assert result.results[0].item_id == "fit-jeans"


def test_retriever_excludes_publicly_rejected_items() -> None:
    retriever = LexicalCatalogRetriever.build(_pool(), min_df=1, max_features=1_000)
    result = retriever.search(
        SearchSpec(
            query="men's big and tall jeans",
            excluded_item_ids=["fit-jeans"],
            ranking_k=3,
        )
    )
    assert "fit-jeans" not in [item.item_id for item in result.results]
