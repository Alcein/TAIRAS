"""Build target-free, set-valued pilot cases from frozen catalog metadata."""

from __future__ import annotations

import csv
import hashlib
import heapq
import html
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from taira_mt.data.legacy_catalog import _sha256
from taira_mt.models import (
    AttributeValue,
    BeliefState,
    CatalogItem,
    Constraint,
    ConstraintImportance,
    HiddenGoal,
    UserPersona,
)
from taira_mt.pilot.schemas import (
    CandidatePoolRecord,
    ConstructionProvenance,
    PilotScenario,
    PrivateCaseSpec,
    PublicCaseSpec,
    SourceFileRecord,
)
from taira_mt.recommender import CandidatePool, CatalogIndex
from taira_mt.state import DomainSlotOntology, SlotDefinition, initialize_belief

BUILDER_VERSION = "category_set_v1"
POOL_RECIPE_ID = "category_5pos_15near_10far_v1"


@dataclass(frozen=True)
class AnchorSpec:
    case_id: str
    domain: str
    scenario: PilotScenario
    anchor_item_id: str
    visible_tag: str
    hidden_tag: str


DEFAULT_ANCHOR_SPECS = (
    AnchorSpec("beauty-clarify-001", "amazon_beauty", PilotScenario.CLARIFY, "B006GACVYQ", "face", "acids & peels"),
    AnchorSpec(
        "beauty-ready-001",
        "amazon_beauty",
        PilotScenario.READY,
        "B0058KSD56",
        "eyelash tools",
        "fake eyelashes & adhesives",
    ),
    AnchorSpec(
        "beauty-repair-001",
        "amazon_beauty",
        PilotScenario.REPAIR,
        "B0013VBMJ8",
        "hair care",
        "hair extensions",
    ),
    AnchorSpec(
        "clothing-clarify-001",
        "amazon_clothing",
        PilotScenario.CLARIFY,
        "B005QNUHA4",
        "running",
        "trail running",
    ),
    AnchorSpec(
        "clothing-ready-001",
        "amazon_clothing",
        PilotScenario.READY,
        "B00HEGRWEK",
        "women",
        "blouses & button-down shirts",
    ),
    AnchorSpec(
        "clothing-repair-001",
        "amazon_clothing",
        PilotScenario.REPAIR,
        "B000ML7OYW",
        "boots",
        "rain boots",
    ),
    AnchorSpec(
        "music-clarify-001",
        "amazon_music",
        PilotScenario.CLARIFY,
        "B00000AFQ0",
        "dance & electronic",
        "house",
    ),
    AnchorSpec(
        "music-ready-001",
        "amazon_music",
        PilotScenario.READY,
        "B000002WT1",
        "jazz",
        "acid jazz",
    ),
    AnchorSpec(
        "music-repair-001",
        "amazon_music",
        PilotScenario.REPAIR,
        "B00006AW2G",
        "classic rock",
        "album-oriented rock (aor)",
    ),
)


@dataclass(frozen=True)
class CatalogRow:
    item_id: str
    title: str
    category_tags: tuple[str, ...]


@dataclass(frozen=True)
class BuiltPilotCase:
    public_case: PublicCaseSpec
    private_case: PrivateCaseSpec
    ontology: DomainSlotOntology
    initial_belief: BeliefState
    catalog_index: CatalogIndex
    candidate_pool: CandidatePool
    candidate_pool_record: CandidatePoolRecord


def _normalize_text(value: str) -> str:
    value = html.unescape(value)
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_category_tag(value: str) -> str:
    return _normalize_text(value).casefold()


def parse_category_tags(raw_value: str) -> tuple[str, ...]:
    tags = {normalize_category_tag(part) for part in raw_value.split("|") if normalize_category_tag(part)}
    return tuple(sorted(tags))


def _metadata_path(legacy_root: Path, domain: str) -> Path:
    return legacy_root / "data" / domain / "metadata.csv"


def _query_path(legacy_root: Path, domain: str) -> Path:
    return (
        legacy_root
        / "data"
        / "query_construction_v2"
        / "artifacts"
        / "runs"
        / f"{domain}_sample"
        / "queries.csv"
    )


def load_catalog_rows(path: Path) -> list[CatalogRow]:
    rows: list[CatalogRow] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "title", "category"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"metadata is missing required columns: {sorted(missing)}")
        for raw in reader:
            item_id = (raw.get("id") or "").strip()
            if not item_id:
                continue
            if item_id in seen:
                raise ValueError(f"duplicate item ID in metadata: {item_id}")
            seen.add(item_id)
            tags = parse_category_tags(raw.get("category") or "")
            if not tags:
                continue
            rows.append(CatalogRow(item_id, _normalize_text(raw.get("title") or ""), tags))
    return rows


def _source_query_row(path: Path, anchor_item_id: str) -> dict[str, str]:
    matches: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("target_item_id") or "").strip() == anchor_item_id:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one V2 source row for {anchor_item_id}, found {len(matches)}")
    return matches[0]


def _stable_take(rows: list[CatalogRow], count: int, salt: str) -> list[CatalogRow]:
    def key(row: CatalogRow) -> tuple[str, str]:
        digest = hashlib.sha256(f"{salt}::{row.item_id}".encode()).hexdigest()
        return digest, row.item_id

    return heapq.nsmallest(count, rows, key=key)


def _domain_noun(domain: str) -> str:
    return {
        "amazon_beauty": "beauty product",
        "amazon_clothing": "clothing item",
        "amazon_music": "music release",
    }[domain]


def _initial_request(spec: AnchorSpec) -> str:
    noun = _domain_noun(spec.domain)
    if spec.scenario == PilotScenario.READY:
        return f"I'm looking for a {noun} in {spec.visible_tag}, specifically {spec.hidden_tag}."
    return f"I'm looking for a {noun} in {spec.visible_tag}."


def _pseudo_user_id(raw_user_id: str) -> str:
    return f"pilot-user-{hashlib.sha256(raw_user_id.encode()).hexdigest()[:12]}"


def _as_catalog_item(row: CatalogRow) -> CatalogItem:
    tags = list(row.category_tags)
    return CatalogItem(
        item_id=row.item_id,
        title=row.title,
        attributes={
            "broad_category": tags,
            "specific_category": tags,
            "category_tags": tags,
        },
    )


def build_case(
    spec: AnchorSpec,
    rows: list[CatalogRow],
    legacy_root: Path,
    metadata_sha256: str,
) -> BuiltPilotCase:
    by_id = {row.item_id: row for row in rows}
    try:
        anchor = by_id[spec.anchor_item_id]
    except KeyError as exc:
        raise KeyError(f"anchor is missing from metadata: {spec.anchor_item_id}") from exc
    query_path = _query_path(legacy_root, spec.domain)
    query_row = _source_query_row(query_path, spec.anchor_item_id)
    raw_history = (query_row.get("interaction_string") or "").split("|")
    excluded_ids = {item_id.strip() for item_id in raw_history if item_id.strip()}
    excluded_ids.add(spec.anchor_item_id)
    visible = normalize_category_tag(spec.visible_tag)
    hidden = normalize_category_tag(spec.hidden_tag)

    eligible = [row for row in rows if row.item_id not in excluded_ids]
    positives = [row for row in eligible if visible in row.category_tags and hidden in row.category_tags]
    near_negatives = [row for row in eligible if visible in row.category_tags and hidden not in row.category_tags]
    far_negatives = [row for row in eligible if visible not in row.category_tags]
    if len(positives) < 5 or len(near_negatives) < 15 or len(far_negatives) < 10:
        raise ValueError(f"insufficient deterministic candidate buckets for {spec.case_id}")

    salt = f"{spec.domain}::{spec.anchor_item_id}::{visible}::{hidden}"
    selected_positive = _stable_take(positives, 5, f"{salt}::positive")
    selected_near = _stable_take(near_negatives, 15, f"{salt}::near")
    selected_far = _stable_take(far_negatives, 10, f"{salt}::far")
    if spec.scenario in {PilotScenario.CLARIFY, PilotScenario.REPAIR}:
        ordered_rows = [*selected_near, *selected_positive, *selected_far]
    else:
        ordered_rows = [*selected_positive, *selected_near, *selected_far]
    items = [_as_catalog_item(row) for row in ordered_rows]
    recipe_config = {
        "builder_version": BUILDER_VERSION,
        "scenario": spec.scenario.value,
        "positive_count": 5,
        "near_negative_count": 15,
        "far_negative_count": 10,
        "anchor_and_history_excluded": True,
    }
    catalog_index = CatalogIndex(items, domain=spec.domain, source_sha256=metadata_sha256)
    candidate_pool = catalog_index.pool(recipe_id=POOL_RECIPE_ID, recipe_config=recipe_config)
    pool_record = CandidatePoolRecord(pool_id=candidate_pool.pool_id, **candidate_pool.identity_payload)

    initial_request = _initial_request(spec)
    specific_is_public = spec.scenario == PilotScenario.READY
    constraints = [
        Constraint(
            slot="broad_category",
            value=visible,
            importance=ConstraintImportance.HARD,
            initially_revealed=True,
        ),
        Constraint(
            slot="specific_category",
            value=hidden,
            importance=ConstraintImportance.HARD,
            initially_revealed=specific_is_public,
        ),
    ]
    goal = HiddenGoal(
        goal_id=f"goal-{spec.case_id}",
        user_id=_pseudo_user_id(query_row.get("user_id") or "unknown"),
        domain=spec.domain,
        initial_request=initial_request,
        constraints=constraints,
        target_item_ids=[],
    )
    persona = UserPersona(
        persona_id=f"persona-{spec.case_id}",
        patience_turns=6,
        cooperativeness=1.0,
        acceptance_threshold=0.8,
        question_cost=0.04,
    )
    ontology = DomainSlotOntology(
        domain=spec.domain,
        slots=[
            SlotDefinition(name="broad_category", askable=False, priority=10),
            SlotDefinition(name="specific_category", aliases=["subtype"], askable=True, priority=20),
        ],
    )
    observed: dict[str, AttributeValue] = {"broad_category": visible}
    if specific_is_public:
        observed["specific_category"] = hidden
    initial_belief = initialize_belief(
        observed_constraints=observed,
        ontology=ontology,
        candidate_pool=candidate_pool,
        patience_prior=5,
    )

    query_relative = str(query_path.relative_to(legacy_root))
    provenance = ConstructionProvenance(
        builder_version=BUILDER_VERSION,
        anchor_item_id=spec.anchor_item_id,
        anchor_title=anchor.title,
        raw_user_id=query_row.get("user_id") or "",
        source_query_relative_path=query_relative,
        source_query_sha256=_sha256(query_path),
        excluded_history_item_ids=sorted(excluded_ids),
        positive_count_in_source=len(positives),
        near_negative_count_in_source=len(near_negatives),
        far_negative_count_in_source=len(far_negatives),
    )
    public_case = PublicCaseSpec(
        case_id=spec.case_id,
        domain=spec.domain,
        scenario=spec.scenario,
        initial_request=initial_request,
        observed_constraints=observed,
        ontology_id=f"{spec.domain}-category-v1",
        candidate_pool_id=candidate_pool.pool_id,
    )
    private_case = PrivateCaseSpec(
        case_id=spec.case_id,
        hidden_goal=goal,
        persona=persona,
        acceptable_item_ids=[row.item_id for row in selected_positive],
        provenance=provenance,
    )
    return BuiltPilotCase(
        public_case=public_case,
        private_case=private_case,
        ontology=ontology,
        initial_belief=initial_belief,
        catalog_index=catalog_index,
        candidate_pool=candidate_pool,
        candidate_pool_record=pool_record,
    )


def build_domain_cases(
    legacy_root: Path,
    domain: str,
    specs: tuple[AnchorSpec, ...] = DEFAULT_ANCHOR_SPECS,
) -> tuple[list[BuiltPilotCase], list[SourceFileRecord]]:
    metadata_path = _metadata_path(legacy_root, domain)
    query_path = _query_path(legacy_root, domain)
    rows = load_catalog_rows(metadata_path)
    metadata_hash = _sha256(metadata_path)
    selected_specs = [spec for spec in specs if spec.domain == domain]
    built = [build_case(spec, rows, legacy_root, metadata_hash) for spec in selected_specs]
    sources = [
        SourceFileRecord(
            domain=domain,
            kind="metadata",
            relative_path=str(metadata_path.relative_to(legacy_root)),
            sha256=metadata_hash,
            row_count=len(rows),
        ),
        SourceFileRecord(
            domain=domain,
            kind="v2_query_anchor",
            relative_path=str(query_path.relative_to(legacy_root)),
            sha256=_sha256(query_path),
            row_count=sum(1 for _ in query_path.open(encoding="utf-8")) - 1,
        ),
    ]
    return built, sources
