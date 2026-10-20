"""Construct and inspect a compact, rich multi-turn recommendation benchmark.

The old TAIRA query assets contain the important object this project must keep:
requests expressed through occasions, functions, style, and latent needs.  This
module turns those legacy queries into a reproducible benchmark whose main
challenge is not category/price filtering.  It uses full-catalog TF-IDF search
to build candidate evidence, then freezes a transparent semantic-fit proxy and
multi-turn disclosure contract.  A later one-time LLM judge can replace or
augment the proxy without changing the public case schema.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from taira_mt.provenance import content_sha256


RICH_BENCHMARK_VERSION = "rich_intent_benchmark_v8"
FAMILIES = ("direct", "progressive", "ambiguous", "revision", "conflict")
DOMAINS = ("amazon_clothing", "amazon_beauty", "amazon_music")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "can", "could", "do", "for", "from",
    "give", "have", "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "recommend",
    "show", "some", "that", "the", "this", "to", "want", "what", "with", "would", "you",
}
MUSIC_GENRES = (
    "rock", "pop", "jazz", "disco", "electronic", "folk", "country", "gospel", "blues", "classical",
    "rap", "hip hop", "hip-hop", "r&b", "soul",
)


@dataclass(frozen=True)
class AttributeConcept:
    name: str
    aliases: tuple[str, ...]
    kind: str


ATTRIBUTE_LEXICONS: dict[str, tuple[AttributeConcept, ...]] = {
    "amazon_clothing": (
        AttributeConcept("breathable", ("breathable", "ventilat", "airflow", "cool", "summer"), "functional"),
        AttributeConcept("lightweight", ("lightweight", "light", "thin", "not too thick"), "functional"),
        AttributeConcept("moisture_wicking", ("moisture", "sweat", "wicking", "dry", "running"), "functional"),
        AttributeConcept("durable", ("durable", "sturdy", "long lasting", "long-lasting"), "functional"),
        AttributeConcept("comfortable", ("comfortable", "comfort", "soft", "easy to wear"), "functional"),
        AttributeConcept("flexible_fit", ("flexible", "stretch", "fits", "petite", "big tall"), "fit"),
        AttributeConcept("casual_style", ("casual", "everyday", "playdate", "outing", "city walk"), "style"),
        AttributeConcept("formal_style", ("formal", "wedding", "interview", "ceremony", "family gathering"), "style"),
        AttributeConcept("weather_protection", ("outdoor", "hiking", "rain", "cold", "winter", "sun"), "functional"),
        AttributeConcept("performance_use", ("athletic", "sport", "exercise", "fitness", "activewear"), "functional"),
    ),
    "amazon_beauty": (
        AttributeConcept("gentle", ("gentle", "sensitive", "mild", "irritated"), "functional"),
        AttributeConcept("soothing", ("soothe", "soothing", "calm", "relief", "healing"), "functional"),
        AttributeConcept("hydrating", ("hydrating", "hydrate", "moisturiz", "dry skin"), "functional"),
        AttributeConcept("sun_recovery", ("sun exposure", "after sun", "sunburn", "uv"), "functional"),
        AttributeConcept("natural_organic", ("natural", "organic", "virgin", "cold pressed", "ecocert"), "ingredient"),
        AttributeConcept("anti_aging", ("anti-aging", "anti aging", "age", "youthful", "wrinkle"), "functional"),
        AttributeConcept("fast_convenient", ("fast", "quick", "easy", "mist", "spray", "travel"), "functional"),
        AttributeConcept("treatment_strength", ("acid", "peel", "treatment", "exfoliat", "repair"), "functional"),
        AttributeConcept("hair_application", ("hair", "coloring", "styling", "applicator"), "usage"),
        AttributeConcept("skin_health", ("skin care", "skin health", "appearance", "complexion"), "usage"),
        AttributeConcept("blemish_control", ("blemish", "pores", "oily skin", "acne", "spot treatment"), "functional"),
        AttributeConcept("style_expression", ("vintage", "elegance", "elegant", "costume", "themed"), "style"),
        AttributeConcept("precise_application", ("mist", "spray", "applicator", "coloring"), "usage"),
    ),
    "amazon_music": (
        AttributeConcept("energetic", ("energetic", "energy", "lively", "upbeat", "vibrant"), "mood"),
        AttributeConcept("danceable", ("dance", "party", "rhythm", "beats"), "mood"),
        AttributeConcept("relaxing", ("relax", "unwind", "after a long day", "calm", "easy listening"), "mood"),
        AttributeConcept("immersive", ("immersive", "complex arrangement", "full listen", "album-oriented"), "listening"),
        AttributeConcept("nostalgic", ("nostalgic", "heritage", "classic", "era", "early 80"), "style"),
        AttributeConcept("soulful", ("soulful", "authentic", "emotional", "blues"), "mood"),
        AttributeConcept("modern_polish", ("modern", "fresh", "polish", "contemporary"), "style"),
        AttributeConcept("social_context", ("barbecue", "gathering", "background", "event", "playlist"), "usage"),
        AttributeConcept("cultural_identity", ("british", "cultural", "identity", "heritage"), "usage"),
        AttributeConcept("genre_alignment", ("rock", "pop", "jazz", "disco", "electronic", "folk", "country"), "genre"),
        AttributeConcept("catchy_hook", ("catchy", "hook", "singalong"), "mood"),
        AttributeConcept("explicit_content", ("explicit", "uncensored"), "content"),
        AttributeConcept("physical_format", ("vinyl", "cd", "box set"), "format"),
    ),
}

DEFAULT_ATTRIBUTES = {
    "amazon_clothing": ("comfortable", "durable", "casual_style"),
    "amazon_beauty": ("gentle", "hydrating", "skin_health"),
    "amazon_music": ("immersive", "genre_alignment", "social_context"),
}

# Product type is a first-class constraint rather than a broad domain/category
# overlap.  The old v3 proxy treated e.g. ``conditioner`` and ``cleanser`` as
# similar merely because both occurred under Beauty > Skin Care.  These compact
# signatures keep the benchmark tied to the actual product requested while
# allowing ordinary lexical variants in Amazon titles and category paths.
TYPE_SIGNATURES: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "amazon_clothing": (
        ("tie clip", ("tie clip",)),
        ("bodysuit", ("bodysuit",)),
        ("cover-up", ("cover up", "cover-up", "caftan")),
        ("nightgown", ("nightgown", "sleepwear", "pajama")),
        ("costume", ("costume",)),
        ("t-shirt", ("t-shirt", "tee shirt", "t shirt", "tee")),
        ("tank top", ("tank top",)),
        ("blouse", ("blouse",)),
        ("shorts", ("shorts",)),
        ("jeans", ("jeans",)),
        ("jeggings", ("jeggings",)),
        ("dress", ("dress", "gown")),
        ("flats", ("flat", "flats", "ballerina")),
        ("boots", ("boot", "boots")),
        ("sandal", ("sandal", "sandals")),
        ("shoes", ("shoe", "shoes", "footwear")),
        ("skirt", ("skirt",)),
        ("leggings", ("legging", "leggings")),
    ),
    "amazon_beauty": (
        ("fragrance", ("fragrance", "eau de parfum", "perfume", "rollerball")),
        ("blemish treatment", ("blemish treatment", "spot treatment", "acne treatment")),
        ("perfume oil", ("perfume oil", "fragrance oil")),
        ("makeup brush", ("makeup brush", "brush head")),
        ("face mask", ("face mask", "facial mask")),
        ("face cream", ("face cream", "facial cream", "facial moisturizer", "face moisturizer")),
        ("body lotion", ("body lotion", "body moisturizer", "body moisturiser", "body cream")),
        ("conditioner", ("conditioner",)),
        ("shampoo", ("shampoo",)),
        ("concealer", ("concealer",)),
        ("wash cloth", ("wash cloth", "washcloth")),
        ("bath gloves", ("bath glove", "bath gloves", "exfoliating glove", "exfoliating gloves")),
        ("body butter", ("body butter", "cocoa butter", "shea butter")),
        ("face treatment", ("face treatment", "treatment serum", "facial treatment")),
        ("moisturizer", ("moisturizer", "moisturiser", "moisturizing lotion", "moisturising lotion")),
        ("cleanser", ("cleanser", "face wash", "facial wash")),
        ("oil", ("oil",)),
    ),
    "amazon_music": (
        ("vinyl", ("vinyl",)),
        ("box set", ("box set",)),
        ("compilation", ("compilation",)),
        ("collection", ("collection",)),
        ("album", ("album",)),
        ("track", ("track", "single")),
        ("tune", ("tune", "song")),
        ("music", ("music",)),
    ),
}


@dataclass(frozen=True)
class RichBenchmarkConfig:
    cases_per_family_domain: int = 20
    candidate_pool_size: int = 120
    retrieval_top_k: int = 90
    source_seed: int = 20260730
    version: str = RICH_BENCHMARK_VERSION


@dataclass(frozen=True)
class CatalogRow:
    item_id: str
    title: str
    category: str
    description: str
    price: str
    quality: str

    @property
    def text(self) -> str:
        return " ".join(value for value in (self.title, self.description, self.category) if value)


class CatalogSearchIndex:
    """In-memory full-catalog TF-IDF index shared by every case in one domain."""

    def __init__(self, rows: list[CatalogRow]) -> None:
        self.rows = rows
        self.position_by_item_id = {row.item_id: position for position, row in enumerate(rows)}
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            dtype=np.float32,
            min_df=2,
            max_features=140_000,
        )
        self.matrix = self.vectorizer.fit_transform([row.text for row in rows])

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        vector = self.vectorizer.transform([query])
        scores = (self.matrix @ vector.T).toarray().ravel()
        if not np.any(scores):
            return []
        top_k = min(top_k, len(scores))
        indices = np.argpartition(scores, -top_k)[-top_k:]
        ordered = indices[np.argsort(scores[indices])[::-1]]
        return [(int(index), float(scores[index])) for index in ordered if scores[index] > 0.0]


def _stable_key(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.casefold()) if len(token) > 2 and token not in STOP_WORDS}


def _clean_type(value: str, category: str) -> str:
    first = value.split("|")[0].strip() if value else ""
    first = re.sub(r"\b(that|which|for|with|and|the|a|an)\b.*", "", first, flags=re.IGNORECASE).strip()
    if len(first) >= 3 and len(first.split()) <= 9:
        return first
    pieces = [piece.strip() for piece in category.split("|") if piece.strip()]
    return pieces[-1] if pieces else "product"


def _concepts_for(domain: str) -> dict[str, AttributeConcept]:
    return {concept.name: concept for concept in ATTRIBUTE_LEXICONS[domain]}


def _matching_concepts(domain: str, text: str) -> list[str]:
    lowered = text.casefold()

    def matches(alias: str) -> bool:
        escaped = re.escape(alias.casefold())
        return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", lowered) is not None

    matched = [
        concept.name
        for concept in ATTRIBUTE_LEXICONS[domain]
        if any(matches(alias) for alias in concept.aliases)
    ]
    return matched


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase.casefold())}(?![a-z0-9])", text.casefold()) is not None


def _target_signature(domain: str, target_type: str, source_query: str, category: str) -> tuple[str, tuple[str, ...]]:
    """Return a compact product-type signature inferred from the user request.

    Category text is deliberately a fallback only.  It is useful when legacy
    Music rows lack an explicit target, but must not turn every Beauty product
    into a matching ``skin care`` product.
    """

    target_text = f"{target_type} {source_query}".casefold()
    for name, aliases in TYPE_SIGNATURES[domain]:
        if any(_contains_phrase(target_text, alias) for alias in aliases):
            return name, aliases
    fallback = tuple(sorted(_tokens(target_type)))
    if fallback:
        return "lexical_fallback", fallback
    category_tokens = tuple(sorted(_tokens(category)))
    return "category_fallback", category_tokens[:3] or ("product",)


def _type_match(signature: tuple[str, tuple[str, ...]], item: CatalogRow) -> tuple[float, list[str]]:
    """Score type match from exact requested-type evidence, not domain ancestry."""

    name, aliases = signature
    text = item.text.casefold()
    hits = [alias for alias in aliases if _contains_phrase(text, alias)]
    if hits:
        return 1.0, hits
    if name == "lexical_fallback":
        overlap = len(set(aliases).intersection(_tokens(text)))
        return round(overlap / max(1, min(2, len(aliases))), 4), []
    return 0.0, []


def _derive_need(domain: str, source: dict[str, str], family: str) -> dict[str, Any]:
    query = source["query"]
    # Rich need facts must originate in the user request.  Anchor title/category
    # remain retrieval evidence only; otherwise title words such as "vintage"
    # can silently create an unrelated evaluator attribute.
    source_text = query
    explicit_concepts = _matching_concepts(domain, source_text)
    concepts = list(explicit_concepts)
    defaults = list(DEFAULT_ATTRIBUTES[domain])
    for value in defaults:
        if len(concepts) >= 5:
            break
        if value not in concepts:
            concepts.append(value)
    concepts = concepts[:6]
    if len(concepts) < 3:
        concepts = (concepts + defaults)[:3]

    concept_map = _concepts_for(domain)
    hard = concepts[:2]
    soft = concepts[2:]
    initial_known = list(explicit_concepts) or concepts[:1]
    latent = [name for name in concepts if name not in initial_known]
    unresolved = [] if family == "direct" else latent
    target_type = _clean_type(source.get("targets", ""), source["category"])
    signature_name, signature_aliases = _target_signature(domain, target_type, source["query"], source["category"])
    hard = list(explicit_concepts[:2]) or concepts[:1]
    soft = [name for name in concepts if name not in hard]

    scenario = _scenario_surface(domain, concepts, source["query"])
    return {
        "target_type": target_type,
        "target_signature": {"name": signature_name, "aliases": list(signature_aliases)},
        "source_query": source["query"],
        "genre_constraints": _genre_constraints(source["query"]) if domain == "amazon_music" else [],
        "usage_scenario": scenario,
        "attributes": [
            {
                "name": name,
                "kind": concept_map[name].kind,
                "aliases": list(concept_map[name].aliases),
            }
            for name in concepts
        ],
        "hard_attributes": hard,
        "soft_attributes": soft,
        "initial_known_attributes": initial_known,
        "unresolved_attributes": unresolved,
        "budget": _parse_budget(source.get("price", ""), family),
        "family": family,
    }


def _scenario_surface(domain: str, concepts: list[str], query: str) -> str:
    concept_set = set(concepts)
    if domain == "amazon_clothing":
        if "performance_use" in concept_set or "moisture_wicking" in concept_set:
            return "exercise, running, or an active outdoor routine"
        if "formal_style" in concept_set:
            return "a formal gathering, interview, or family occasion"
        if "weather_protection" in concept_set:
            return "extended time outdoors in changing weather"
        if "casual_style" in concept_set:
            return "daily wear and relaxed social outings"
        return "a practical occasion with comfort and fit requirements"
    if domain == "amazon_beauty":
        if "sun_recovery" in concept_set:
            return "skin recovery after time in the sun"
        if "hair_application" in concept_set:
            return "a hair coloring or styling routine"
        if "gentle" in concept_set or "soothing" in concept_set:
            return "a sensitive-skin care routine"
        return "a targeted daily beauty-care routine"
    if "danceable" in concept_set or "energetic" in concept_set:
        return "a lively social gathering or party playlist"
    if "relaxing" in concept_set:
        return "unwinding after a long day"
    if "cultural_identity" in concept_set:
        return "a cultural identity or heritage event"
    if "nostalgic" in concept_set:
        return "a nostalgic social occasion"
    return "a focused listening context with a distinct mood"


def _parse_budget(value: str, family: str) -> dict[str, Any] | None:
    try:
        price = float(value.replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 0.0:
        return None
    multiplier = 1.35 if family != "conflict" else 0.9
    return {"operator": "lte", "value": round(price * multiplier, 2), "source": "anchor_price"}


def _initial_query(domain: str, need: dict[str, Any], family: str) -> str:
    item_type = need["target_type"]
    scenario = need["usage_scenario"]
    attributes = need["hard_attributes"] + need["soft_attributes"]
    if family == "direct":
        rendered = ", ".join(value.replace("_", " ") for value in attributes[:3])
        return f"I need {item_type} for {scenario}, ideally {rendered}."
    if family == "progressive":
        return f"I am looking for {item_type} for {scenario}, but I may need help narrowing down the right features."
    if family == "ambiguous":
        return f"I need {item_type} for {scenario}; several interpretations seem plausible to me."
    if family == "revision":
        return f"Could you suggest {item_type} for {scenario}? I have a few preferences but may refine them after seeing options."
    return f"I need {item_type} for {scenario}, but my requirements may force a trade-off."


def _source_rows(data_root: Path, domain: str) -> list[dict[str, str]]:
    path = data_root / domain / "query_data1.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        if not row.get("id") or not row.get("query") or not row.get("title"):
            continue
        if len(row["query"].split()) < 8:
            continue
        result.append(row)
    return result


def _catalog_rows(data_root: Path, domain: str) -> list[CatalogRow]:
    path = data_root / domain / "metadata.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    rows = []
    for row in records:
        item_id = (row.get("id") or "").strip()
        if not item_id:
            continue
        rows.append(
            CatalogRow(
                item_id=item_id,
                title=(row.get("title") or "").strip(),
                category=(row.get("category") or "").strip(),
                description=(row.get("description") or "").strip(),
                price=(row.get("price") or "").strip(),
                quality=(row.get("rating") or row.get("ranking") or "").strip(),
            )
        )
    return rows


def _select_sources(data_root: Path, config: RichBenchmarkConfig) -> list[tuple[str, str, dict[str, str]]]:
    selected: list[tuple[str, str, dict[str, str]]] = []
    for domain in DOMAINS:
        rows = _source_rows(data_root, domain)
        used: set[str] = set()
        for family in FAMILIES:
            ranked = sorted(
                rows,
                key=lambda row: _stable_key(config.source_seed, domain, family, row["id"], row["query"]),
            )
            family_rows = []
            for row in ranked:
                key = f"{row['id']}::{row.get('user_id', '')}::{row['query']}"
                if key in used:
                    continue
                family_rows.append(row)
                used.add(key)
                if len(family_rows) == config.cases_per_family_domain:
                    break
            if len(family_rows) != config.cases_per_family_domain:
                raise ValueError(f"insufficient rich sources for {domain}/{family}")
            selected.extend((domain, family, row) for row in family_rows)
    return selected


def _attribute_evidence(need: dict[str, Any], item: CatalogRow) -> dict[str, list[str]]:
    text = item.text.casefold()
    evidence: dict[str, list[str]] = {}
    for attribute in need["attributes"]:
        aliases = attribute["aliases"]
        # Match the genres named in this particular request, rather than any
        # genre word (e.g. a Rock album cannot satisfy a Pop request by itself).
        if attribute["name"] == "genre_alignment" and need.get("genre_constraints"):
            aliases = need["genre_constraints"]
        hits = [
            alias
            for alias in aliases
            if re.search(rf"(?<![a-z0-9]){re.escape(alias.casefold())}(?![a-z0-9])", text) is not None
        ]
        if hits:
            evidence[attribute["name"]] = hits
    return evidence


def _genre_constraints(query: str) -> list[str]:
    return [genre for genre in MUSIC_GENRES if _contains_phrase(query, genre)]


def _normalized_scores(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    maximum = max(values.values())
    if maximum <= 0.0:
        return {key: 0.0 for key in values}
    return {key: value / maximum for key, value in values.items()}


def _candidate_pool(
    *,
    domain: str,
    index: CatalogSearchIndex,
    source: dict[str, str],
    need: dict[str, Any],
    config: RichBenchmarkConfig,
) -> list[dict[str, Any]]:
    attributes = " ".join(value["name"].replace("_", " ") for value in need["attributes"])
    signature = need["target_signature"]
    type_query = " ".join(signature["aliases"]) or need["target_type"]
    pattern_query = f"{need['source_query']} {attributes}"
    queries = {
        "raw": need["source_query"],
        "type": type_query,
        "pattern": pattern_query,
    }
    scores_by_kind: dict[str, dict[int, float]] = {}
    selected: set[int] = set()
    for kind, query in queries.items():
        values = dict(index.search(query, config.retrieval_top_k))
        scores_by_kind[kind] = _normalized_scores(values)
        selected.update(values)

    source_position = index.position_by_item_id.get(source["id"])
    if source_position is not None:
        selected.add(source_position)

    # The source item's historical interaction is a traceable relevance anchor,
    # not a leaked answer: the public Agent still searches the entire catalog
    # and never receives its id.  Keeping it in private evidence guarantees that
    # each frozen case has one real, query-grounded positive alongside retrieved
    # semantic alternatives.
    excluded_ids = set(value for value in (source.get("remaining_interaction_string") or "").split("|") if value)
    excluded_ids.discard(source["id"])
    selected = {item for item in selected if index.rows[item].item_id not in excluded_ids}
    if len(selected) < config.candidate_pool_size:
        remaining = [
            position
            for position, row in enumerate(index.rows)
            if row.item_id not in excluded_ids and position not in selected
        ]
        remaining.sort(key=lambda position: _stable_key(source["id"], "fill", index.rows[position].item_id))
        selected.update(remaining[: config.candidate_pool_size - len(selected)])

    target_signature = (signature["name"], tuple(signature["aliases"]))
    candidates: list[dict[str, Any]] = []
    for position in selected:
        item = index.rows[position]
        evidence = _attribute_evidence(need, item)
        attribute_coverage = len(evidence) / max(1, len(need["attributes"]))
        hard_coverage = len(set(evidence).intersection(need["hard_attributes"])) / max(1, len(need["hard_attributes"]))
        type_overlap, type_evidence = _type_match(target_signature, item)
        raw_score = scores_by_kind["raw"].get(position, 0.0)
        pattern_score = scores_by_kind["pattern"].get(position, 0.0)
        source_grounded = item.item_id == source["id"]
        semantic_fit = 0.38 * type_overlap + 0.34 * attribute_coverage + 0.20 * pattern_score + 0.08 * raw_score
        if source_grounded:
            semantic_fit += 0.10
        candidates.append(
            {
                "item_id": item.item_id,
                "title": item.title,
                "category": item.category,
                "description": item.description[:360],
                "price": item.price,
                "quality": item.quality,
                "attribute_evidence": evidence,
                "type_evidence": type_evidence,
                "attribute_coverage": round(attribute_coverage, 4),
                "hard_attribute_coverage": round(hard_coverage, 4),
                "type_match": round(type_overlap, 4),
                "source_grounded": source_grounded,
                "raw_retrieval_score": round(raw_score, 5),
                "pattern_retrieval_score": round(pattern_score, 5),
                "semantic_fit": round(semantic_fit, 5),
            }
        )
    candidates.sort(key=lambda item: (-item["semantic_fit"], -item["pattern_retrieval_score"], item["item_id"]))
    ranked_candidates = candidates
    candidates = ranked_candidates[: config.candidate_pool_size]
    if not any(item["source_grounded"] for item in candidates):
        # The source item can have a terse title and therefore fall below the
        # lexical top-k.  Recover it from the pre-truncation union deterministically.
        source_candidate = next((item for item in ranked_candidates if item["source_grounded"]), None)
        if source_candidate is not None:
            candidates = candidates[:-1] + [source_candidate]
            candidates.sort(key=lambda item: (-item["semantic_fit"], -item["pattern_retrieval_score"], item["item_id"]))
    source_positive = [item for item in candidates if item["source_grounded"]]
    strict = [item for item in candidates if item["type_match"] >= 1.0 and item["hard_attribute_coverage"] >= 1.0]
    supportive = [item for item in candidates if item["type_match"] >= 1.0 and item["hard_attribute_coverage"] >= 0.5]
    ordered_positive = source_positive + [item for item in strict if item not in source_positive] + [
        item for item in supportive if item not in source_positive and item not in strict
    ]
    positive_item_ids = {item["item_id"] for item in ordered_positive[:12]}
    strict_ids = {item["item_id"] for item in source_positive + strict}
    for item in candidates:
        if item["item_id"] in positive_item_ids:
            item["relevance_tier"] = "excellent" if item["item_id"] in strict_ids else "acceptable"
        elif item["raw_retrieval_score"] > item["pattern_retrieval_score"] and item["attribute_coverage"] < 0.34:
            item["relevance_tier"] = "lexical_hard_negative"
        elif item["attribute_coverage"] < 0.17:
            item["relevance_tier"] = "attribute_hard_negative"
        else:
            item["relevance_tier"] = "plausible_alternative"
    return candidates


def _disclosure_plan(need: dict[str, Any], family: str) -> dict[str, Any]:
    attributes = [item["name"] for item in need["attributes"]]
    known = list(dict.fromkeys(need.get("initial_known_attributes", []) or attributes[:1]))
    hidden = [attribute for attribute in attributes if attribute not in known]
    if family == "direct":
        return {"initial": attributes, "steps": [], "revision": None, "tradeoff": None}
    if family == "progressive":
        return {
            "initial": known,
            "steps": [
                {"ask": "usage_context", "reveal": hidden[:2]},
                {"ask": "priority", "reveal": hidden[2:]},
            ],
            "revision": None,
            "tradeoff": None,
        }
    if family == "ambiguous":
        return {
            "initial": known,
            "steps": [{"ask": "occasion_interpretation", "reveal": hidden[:2]}],
            "revision": None,
            "tradeoff": {"options": (hidden or attributes)[:2]},
        }
    if family == "revision":
        # Revision cases start from a viable inferred interpretation and then
        # change one preference after a concrete option is observed.
        revision = attributes[-1:]
        return {
            "initial": attributes[:-1],
            "steps": [],
            "revision": {"trigger": "first_recommendation", "reveal": revision},
            "tradeoff": None,
        }
    return {
        "initial": known,
        "steps": [{"ask": "tradeoff", "reveal": hidden}],
        "revision": None,
        "tradeoff": {"conflict": _conflict_pair(need["target_type"], attributes)},
    }


def _conflict_pair(target_type: str, attributes: list[str]) -> list[str]:
    pairs = (
        ("lightweight", "weather_protection"),
        ("gentle", "treatment_strength"),
        ("relaxing", "energetic"),
        ("formal_style", "performance_use"),
    )
    attribute_set = set(attributes)
    for pair in pairs:
        if set(pair).issubset(attribute_set):
            return list(pair)
    return attributes[:2]


def _build_case(
    domain: str,
    family: str,
    source: dict[str, str],
    index: CatalogSearchIndex,
    config: RichBenchmarkConfig,
) -> dict[str, Any]:
    need = _derive_need(domain, source, family)
    case_id = f"rich-{domain.replace('amazon_', '')}-{family}-{_stable_key(source['id'], source['query'])[:10]}"
    candidates = _candidate_pool(domain=domain, index=index, source=source, need=need, config=config)
    disclosure = _disclosure_plan(need, family)
    return {
        "case_id": case_id,
        "domain": domain,
        "family": family,
        "public": {
            # The initial user utterance is an immutable legacy natural-language
            # request.  State transitions disclose latent preferences separately;
            # the benchmark never replaces the request with a synthetic slot list.
            "initial_query": need["source_query"],
            "target_type": need["target_type"],
            "query_provenance": "legacy_rich_natural_language_query",
            "turn_budget": 5 if family in {"revision", "conflict"} else 4,
            "search_contract": "full_catalog_bm25_tfidf_then_top10_rerank",
        },
        "private": {
            "source_anchor": {
                "item_id": source["id"],
                "title": source["title"],
                "category": source["category"],
                "legacy_query": source["query"],
            },
            "need_card": need,
            "disclosure_plan": disclosure,
            "candidate_pool_size": len(candidates),
            "acceptable_item_ids": [
                item["item_id"] for item in candidates if item["relevance_tier"] in {"excellent", "acceptable"}
            ],
        },
        "candidates": candidates,
    }


def _candidate_for_strategy(case: dict[str, Any], strategy: str, active_attributes: Iterable[str]) -> dict[str, Any]:
    attributes = set(active_attributes)
    candidates = case["candidates"]
    if strategy == "naive":
        return max(candidates, key=lambda item: (item["raw_retrieval_score"], item["item_id"]))

    def score(item: dict[str, Any]) -> tuple[float, float, str]:
        evidence = set(item["attribute_evidence"])
        coverage = len(evidence.intersection(attributes)) / max(1, len(attributes))
        public_fit = (
            0.35 * item["type_match"]
            + 0.40 * coverage
            + 0.20 * item["pattern_retrieval_score"]
            + 0.05 * item["raw_retrieval_score"]
        )
        return public_fit, item["semantic_fit"], item["item_id"]

    return max(candidates, key=score)


def _render_attribute_list(values: Iterable[str]) -> str:
    return ", ".join(value.replace("_", " ") for value in values)


def simulate_strategy(case: dict[str, Any], strategy: str) -> dict[str, Any]:
    """Run a transparent baseline or Pattern-aware policy against frozen case rules."""

    plan = case["private"]["disclosure_plan"]
    need = case["private"]["need_card"]
    family = case["family"]
    active = list(plan["initial"])
    transcript = [{"speaker": "user", "text": case["public"]["initial_query"]}]
    turns = 0
    success = False
    violation = 0

    def recommend(stage: str) -> dict[str, Any]:
        nonlocal turns, violation, success
        item = _candidate_for_strategy(case, strategy, active)
        evidenced_attributes = [attribute for attribute in active if attribute in item["attribute_evidence"]]
        evidence_reason = _render_attribute_list(evidenced_attributes) if evidenced_attributes else "the requested product type"
        turns += 1
        transcript.append(
            {
                "speaker": "system",
                "action": "recommend",
                "stage": stage,
                "item_id": item["item_id"],
                "title": item["title"],
                "derived_attributes": list(active),
                "text": f"I recommend {item['title']} because catalog evidence supports {evidence_reason}.",
            }
        )
        if item["relevance_tier"] in {"excellent", "acceptable"} and set(need["hard_attributes"]).issubset(set(active)):
            success = True
            transcript.append({"speaker": "user", "act": "accept", "text": "This matches the need well."})
        else:
            violation += 1
            transcript.append(
                {
                    "speaker": "user",
                    "act": "reject",
                    "text": "That misses an important functional or situational requirement.",
                }
            )
        return item

    if strategy == "naive":
        recommend("initial")
        if family in {"progressive", "ambiguous", "conflict"} and not success:
            transcript.append({"speaker": "user", "act": "clarify", "text": "I need a more specific fit for the situation."})
            active = [item["name"] for item in need["attributes"]]
            recommend("late_repair")
        elif family == "revision" and not success:
            active = [item["name"] for item in need["attributes"]]
            transcript.append({"speaker": "user", "act": "revise", "text": "After seeing it, I need to revise one preference."})
            recommend("late_revision")
    else:
        for step in plan["steps"]:
            if not step["reveal"]:
                continue
            turns += 1
            transcript.append(
                {
                    "speaker": "system",
                    "action": "ask",
                    "target": step["ask"],
                    "text": f"Which matters more for this situation: {_render_attribute_list(step['reveal'])}?",
                }
            )
            active.extend(value for value in step["reveal"] if value not in active)
            transcript.append(
                {
                    "speaker": "user",
                    "act": "inform",
                    "text": f"Please prioritize {_render_attribute_list(step['reveal'])}.",
                }
            )
        if family == "conflict":
            turns += 1
            conflict = (plan.get("tradeoff") or {}).get("conflict", [])
            transcript.append(
                {
                    "speaker": "system",
                    "action": "negotiate",
                    "text": f"This request contains competing preferences. Which should take priority: {_render_attribute_list(conflict)}?",
                }
            )
            transcript.append({"speaker": "user", "act": "tradeoff", "text": "Keep the requirement most tied to my use case."})
        recommend("pattern_ready")
        if family == "revision":
            revealed = (plan.get("revision") or {}).get("reveal", [])
            turns += 1
            transcript.append(
                {
                    "speaker": "user",
                    "act": "revise",
                    "text": f"One more thing after seeing the option: prioritize {_render_attribute_list(revealed)}.",
                }
            )
            active.extend(value for value in revealed if value not in active)
            success = False
            recommend("replanned")
    return {
        "strategy": strategy,
        "success": success,
        "turns": turns,
        "violations": violation,
        "transcript": transcript,
    }


def _case_difficulty(case: dict[str, Any]) -> dict[str, Any]:
    need = case["private"]["need_card"]
    candidates = case["candidates"]
    plan = case["private"]["disclosure_plan"]
    positive = sum(item["relevance_tier"] in {"excellent", "acceptable"} for item in candidates)
    hard_negative = sum("hard_negative" in item["relevance_tier"] for item in candidates)
    initial = len(plan["initial"])
    total = len(need["attributes"])
    return {
        "semantic_attribute_count": total,
        "initially_revealed_attribute_count": initial,
        "hidden_attribute_count": total - initial,
        "candidate_pool_size": len(candidates),
        "acceptable_count": positive,
        "hard_negative_count": hard_negative,
        "hard_negative_rate": round(hard_negative / max(1, len(candidates)), 4),
        "transition_required": case["family"] != "direct",
        "requires_repair_or_negotiation": case["family"] in {"revision", "conflict"},
    }


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(case["family"] for case in cases)
    domains = Counter(case["domain"] for case in cases)
    difficulty = [_case_difficulty(case) for case in cases]
    comparisons = []
    for case in cases:
        naive = simulate_strategy(case, "naive")
        pattern = simulate_strategy(case, "pattern")
        comparisons.append({"case_id": case["case_id"], "naive": naive, "pattern": pattern})

    def mean(key: str) -> float:
        return round(sum(float(item[key]) for item in difficulty) / max(1, len(difficulty)), 3)

    return {
        "version": RICH_BENCHMARK_VERSION,
        "case_count": len(cases),
        "domain_counts": dict(sorted(domains.items())),
        "family_counts": dict(sorted(counts.items())),
        "mean_semantic_attribute_count": mean("semantic_attribute_count"),
        "mean_hidden_attribute_count": mean("hidden_attribute_count"),
        "mean_candidate_pool_size": mean("candidate_pool_size"),
        "mean_acceptable_count": mean("acceptable_count"),
        "mean_hard_negative_rate": mean("hard_negative_rate"),
        "transition_cases": sum(item["transition_required"] for item in difficulty),
        "repair_or_negotiation_cases": sum(item["requires_repair_or_negotiation"] for item in difficulty),
        "naive_success_rate": round(sum(item["naive"]["success"] for item in comparisons) / len(comparisons), 4),
        "pattern_success_rate": round(sum(item["pattern"]["success"] for item in comparisons) / len(comparisons), 4),
        "naive_mean_turns": round(sum(item["naive"]["turns"] for item in comparisons) / len(comparisons), 3),
        "pattern_mean_turns": round(sum(item["pattern"]["turns"] for item in comparisons) / len(comparisons), 3),
        "naive_violations": sum(item["naive"]["violations"] for item in comparisons),
        "pattern_violations": sum(item["pattern"]["violations"] for item in comparisons),
        "difficulty_records": difficulty,
        "strategy_comparisons": comparisons,
    }


# A paper/demo needs cases that are both varied and legible.  These slots cover
# the ordinary one-shot control plus all state-transition mechanisms, without
# hand-picking a result after a later experiment.
SHOWCASE_SLOTS = (
    ("amazon_clothing", "direct"),
    ("amazon_clothing", "progressive"),
    ("amazon_clothing", "revision"),
    ("amazon_beauty", "ambiguous"),
    ("amazon_beauty", "conflict"),
    ("amazon_music", "progressive"),
    ("amazon_music", "revision"),
    ("amazon_music", "conflict"),
)


def _last_recommendation(comparison: dict[str, Any]) -> str | None:
    for turn in reversed(comparison["pattern"]["transcript"]):
        if turn.get("action") == "recommend":
            return turn["item_id"]
    return None


def _select_showcases(cases: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {case["case_id"]: case for case in cases}
    comparison_by_id = {comparison["case_id"]: comparison for comparison in comparisons}
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for domain, family in SHOWCASE_SLOTS:
        eligible = [case for case in cases if case["domain"] == domain and case["family"] == family and case["case_id"] not in used]

        def score(case: dict[str, Any]) -> tuple[float, ...]:
            comparison = comparison_by_id[case["case_id"]]
            item_id = _last_recommendation(comparison)
            item = next((candidate for candidate in case["candidates"] if candidate["item_id"] == item_id), {})
            return (
                float(comparison["pattern"]["success"]),
                float(not comparison["naive"]["success"]),
                float(item.get("type_match", 0.0)),
                float(item.get("attribute_coverage", 0.0)),
                float(item.get("semantic_fit", 0.0)),
            )

        if eligible:
            chosen = max(eligible, key=score)
            selected.append({
                "case_id": chosen["case_id"],
                "domain": domain,
                "family": family,
                "selection_rule": "fixed_domain_family_slot_then_pattern_success_naive_gap_type_attribute_fit",
            })
            used.add(chosen["case_id"])
    return selected


def _showcase_markdown(result: dict[str, Any]) -> str:
    cases = {case["case_id"]: case for case in result["cases"]}
    comparisons = {row["case_id"]: row for row in result["analysis"]["strategy_comparisons"]}
    lines = [
        "# Curated rich-intent benchmark showcase",
        "",
        "These cases are selected by a fixed domain/family coverage rule, then by the frozen Pattern-vs-naive gap and evidence quality. "
        "They are illustrative cases, not held-out experimental results.",
        "",
    ]
    for slot in result["showcases"]:
        case = cases[slot["case_id"]]
        need = case["private"]["need_card"]
        plan = case["private"]["disclosure_plan"]
        comparison = comparisons[case["case_id"]]
        final_item_id = _last_recommendation(comparison)
        final_item = next(item for item in case["candidates"] if item["item_id"] == final_item_id)
        negative = next(
            (item for item in case["candidates"] if "hard_negative" in item["relevance_tier"]),
            None,
        )
        lines.extend([
            f"## {case['case_id']} — {case['domain']} / {case['family']}",
            "",
            f"**Fixed initial query.** {case['public']['initial_query']}",
            f"**Requested product type.** {need['target_type']} (signature: {need['target_signature']['name']})",
            f"**State.** known={plan['initial']}; later disclosure={plan['steps']}; revision={plan['revision']}; trade-off={plan['tradeoff']}",
            f"**Search setting.** The Agent searches the full domain catalog; the frozen evaluator evidence contains {len(case['candidates'])} unioned candidates, with {len(case['private']['acceptable_item_ids'])} acceptable alternatives.",
            f"**Evidence-backed final option.** {final_item['title']} — type evidence={final_item['type_evidence']}; attribute evidence={final_item['attribute_evidence']}; tier={final_item['relevance_tier']}.",
        ])
        if negative:
            lines.append(
                f"**A hard negative.** {negative['title']} — type evidence={negative['type_evidence']}; "
                f"attribute evidence={negative['attribute_evidence']}; tier={negative['relevance_tier']}."
            )
        lines.extend([
            f"**Transparent policy diagnostic.** literal-query naive: success={comparison['naive']['success']}, "
            f"turns={comparison['naive']['turns']}, violations={comparison['naive']['violations']}; "
            f"attribute-aware Pattern policy: success={comparison['pattern']['success']}, "
            f"turns={comparison['pattern']['turns']}, violations={comparison['pattern']['violations']}.",
            "**Pattern path.**",
        ])
        for turn in comparison["pattern"]["transcript"]:
            if turn.get("action") in {"ask", "negotiate", "recommend"} or turn.get("act") in {"inform", "revise", "tradeoff"}:
                lines.append(f"- {turn['speaker']}: {turn['text']}")
        lines.append("")
    return "\n".join(lines)


def build_rich_benchmark(data_root: Path, config: RichBenchmarkConfig | None = None) -> dict[str, Any]:
    config = config or RichBenchmarkConfig()
    selected = _select_sources(data_root, config)
    indexes: dict[str, CatalogSearchIndex] = {}
    cases: list[dict[str, Any]] = []
    for domain in DOMAINS:
        print(f"阶段进度：正在建立 {domain} 全量商品文本索引。", flush=True)
        indexes[domain] = CatalogSearchIndex(_catalog_rows(data_root, domain))
        domain_cases = [(family, source) for candidate_domain, family, source in selected if candidate_domain == domain]
        for position, (family, source) in enumerate(domain_cases, start=1):
            if position % 20 == 0 or position == len(domain_cases):
                print(f"阶段进度：{domain} 已构造 {position}/{len(domain_cases)} 个富语义案例。", flush=True)
            cases.append(_build_case(domain, family, source, indexes[domain], config))
    cases.sort(key=lambda item: item["case_id"])
    analysis = _aggregate(cases)
    showcases = _select_showcases(cases, analysis["strategy_comparisons"])
    public_cases = [
        {
            "case_id": case["case_id"],
            "domain": case["domain"],
            "family": case["family"],
            **case["public"],
        }
        for case in cases
    ]
    private_cases = [
        {
            "case_id": case["case_id"],
            "domain": case["domain"],
            "family": case["family"],
            **case["private"],
        }
        for case in cases
    ]
    identity = {
        "version": config.version,
        "config": asdict(config),
        "case_ids": [case["case_id"] for case in cases],
        "public_sha256": content_sha256(public_cases),
        "private_sha256": content_sha256(private_cases),
    }
    return {
        "identity": identity,
        "cases": cases,
        "public_cases": public_cases,
        "private_cases": private_cases,
        "analysis": analysis,
        "showcases": showcases,
    }


def _jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _analysis_markdown(result: dict[str, Any]) -> str:
    analysis = result["analysis"]
    lines = [
        "# Rich-intent benchmark analysis",
        "",
        "## What changed",
        "",
        "This benchmark evaluates scenario-to-attribute conversion and multi-turn planning, not category/price filtering. "
        "Every case was sourced from a legacy rich natural-language query, retrieved against the full local Amazon domain catalog, "
        "and frozen with semantic attributes, explicit product-type evidence, disclosure/revision rules, relevance tiers, and hard negatives.",
        "",
        "## Coverage",
        "",
        f"- Cases: {analysis['case_count']}",
        f"- Domains: {analysis['domain_counts']}",
        f"- Families: {analysis['family_counts']}",
        f"- Mean semantic attributes per case: {analysis['mean_semantic_attribute_count']}",
        f"- Mean initially hidden attributes: {analysis['mean_hidden_attribute_count']}",
        f"- Mean full-catalog-derived candidate evidence pool: {analysis['mean_candidate_pool_size']}",
        f"- Mean acceptable alternatives: {analysis['mean_acceptable_count']}",
        f"- Mean hard-negative rate: {analysis['mean_hard_negative_rate']:.1%}",
        f"- Cases requiring a state transition: {analysis['transition_cases']}",
        f"- Cases requiring repair/negotiation: {analysis['repair_or_negotiation_cases']}",
        "",
        "## Why the task is difficult",
        "",
        "1. The initial request often expresses an occasion or desired outcome rather than catalog fields. The Agent must infer attributes such as breathable, gentle, immersive, or danceable before a useful search can be formed.",
        "2. Progressive, ambiguous, revision, and conflict families deliberately withhold or change attributes. A correct item at one point can become insufficient after feedback.",
        "3. Candidate evidence is retrieved from the entire domain catalog. The frozen pool contains lexical and attribute hard negatives: items that match surface words but miss the requested product type or functional evidence.",
        "4. The Agent is charged for premature recommendation, redundant questioning, and failure to negotiate a conflict; retrieval alone cannot solve the dialogue.",
        "",
        "## How TAIRAS is expected to solve it",
        "",
        "1. DSUS records which scenario facts and attributes are known, unresolved, revised, or conflicting.",
        "2. A matched Thought Pattern supplies an SOP for clarifying the decisive attribute, compiling a SearchSpec, inspecting evidence, and repairing after feedback.",
        "3. The full-catalog search uses the derived attributes rather than only the literal initial query.",
        "4. CIV compares asking, retrieving, recommending, and negotiating at branch points; it should avoid both early recommendation and needless questions.",
        "",
        "## Deterministic sanity comparison",
        "",
        "The following is not a paper baseline table. It checks that a literal-query naive policy and an attribute-aware Pattern policy experience different outcomes under the same frozen simulator. "
        "Relevance tiers are a transparent lexical/semantic proxy and must be calibrated with the planned human and LLM evaluation before any paper claim.",
        "",
        "| Policy | Success | Mean turns | Violations |",
        "|---|---:|---:|---:|",
        f"| Literal-query naive | {analysis['naive_success_rate']:.1%} | {analysis['naive_mean_turns']:.2f} | {analysis['naive_violations']} |",
        f"| Attribute-aware Pattern policy | {analysis['pattern_success_rate']:.1%} | {analysis['pattern_mean_turns']:.2f} | {analysis['pattern_violations']} |",
        "",
    ]
    return "\n".join(lines)


def write_rich_benchmark(result: dict[str, Any], output_root: Path) -> Path:
    run_id = f"rich-intent-benchmark-{content_sha256(result['identity'])[:16]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "cases.public.jsonl").write_text(_jsonl(result["public_cases"]), encoding="utf-8")
    (run_dir / "cases.private.jsonl").write_text(_jsonl(result["private_cases"]), encoding="utf-8")
    (run_dir / "candidate_evidence.private.jsonl").write_text(
        _jsonl({"case_id": case["case_id"], "candidates": case["candidates"]} for case in result["cases"]),
        encoding="utf-8",
    )
    (run_dir / "strategy_comparisons.private.jsonl").write_text(
        _jsonl(result["analysis"]["strategy_comparisons"]), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps({key: value for key, value in result["analysis"].items() if key not in {"difficulty_records", "strategy_comparisons"}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "analysis.md").write_text(_analysis_markdown(result), encoding="utf-8")
    (run_dir / "showcase_case_ids.json").write_text(
        json.dumps(result["showcases"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "showcase_cases.md").write_text(_showcase_markdown(result), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(result["identity"], ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def inspect_rich_benchmark(run_dir: Path, *, case_count: int = 6) -> str:
    public = [json.loads(line) for line in (run_dir / "cases.public.jsonl").read_text(encoding="utf-8").splitlines()]
    private = [json.loads(line) for line in (run_dir / "cases.private.jsonl").read_text(encoding="utf-8").splitlines()]
    evidence = {
        row["case_id"]: row["candidates"]
        for row in (json.loads(line) for line in (run_dir / "candidate_evidence.private.jsonl").read_text(encoding="utf-8").splitlines())
    }
    comparisons = {
        row["case_id"]: row
        for row in (json.loads(line) for line in (run_dir / "strategy_comparisons.private.jsonl").read_text(encoding="utf-8").splitlines())
    }
    by_id = {row["case_id"]: row for row in private}
    showcase_path = run_dir / "showcase_case_ids.json"
    if showcase_path.exists():
        showcase_ids = [row["case_id"] for row in json.loads(showcase_path.read_text(encoding="utf-8"))]
        public_by_id = {row["case_id"]: row for row in public}
        selected = [public_by_id[case_id] for case_id in showcase_ids[:case_count] if case_id in public_by_id]
    else:
        preferred = []
        for domain in DOMAINS:
            for family in ("progressive", "ambiguous", "revision", "conflict", "direct"):
                found = next((row for row in public if row["domain"] == domain and row["family"] == family), None)
                if found is not None:
                    preferred.append(found)
        selected = preferred[:case_count]
    blocks = []
    for case in selected:
        private_case = by_id[case["case_id"]]
        need = private_case["need_card"]
        comparison = comparisons[case["case_id"]]
        final_id = _last_recommendation(comparison)
        final_item = next(item for item in evidence[case["case_id"]] if item["item_id"] == final_id)
        hard_negative = next((item for item in evidence[case["case_id"]] if "hard_negative" in item["relevance_tier"]), None)
        blocks.extend(
            [
                f"## {case['case_id']} ({case['domain']} / {case['family']})",
                "",
                f"Initial request: {case['initial_query']}",
                f"Target type: {case['target_type']}",
                f"Need card: scenario={need['usage_scenario']}; type-signature={need['target_signature']['name']}; hard={need['hard_attributes']}; soft={need['soft_attributes']}; unresolved={need['unresolved_attributes']}",
                f"Candidate evidence: {private_case['candidate_pool_size']} items, {len(private_case['acceptable_item_ids'])} acceptable items.",
                f"Pattern-selected evidence: {final_item['title']} (type={final_item['type_evidence']}; attributes={final_item['attribute_evidence']}; tier={final_item['relevance_tier']}).",
                f"Naive: success={comparison['naive']['success']}, turns={comparison['naive']['turns']}, violations={comparison['naive']['violations']}",
                f"Pattern-aware: success={comparison['pattern']['success']}, turns={comparison['pattern']['turns']}, violations={comparison['pattern']['violations']}",
                "Pattern transcript:",
            ]
        )
        for turn in comparison["pattern"]["transcript"]:
            blocks.append(f"- {turn['speaker']}: {turn['text']}")
        if hard_negative:
            blocks.append(
                f"Hard negative: {hard_negative['title']} (type={hard_negative['type_evidence']}; "
                f"attributes={hard_negative['attribute_evidence']}; tier={hard_negative['relevance_tier']})."
            )
        blocks.append("")
    return "\n".join(blocks)
