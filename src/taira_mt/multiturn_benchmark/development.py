"""Build and independently quality-gate the 30-case multi-turn development set.

This module intentionally treats legacy TAIRA queries as source utterances, not
as pre-existing dialogues.  A fixed blueprint must state why a particular
multi-turn transition is decision-critical; a strong-model author and a
separate judge then create and admit the final trajectory contract.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from taira_mt.rich_benchmark.core import CatalogSearchIndex, _catalog_rows


DEVELOPMENT_VERSION = "mt_complex_intent_development_v2"
QUALITY_DIMENSIONS = (
    "source_fidelity",
    "transition_necessity",
    "state_consistency",
    "catalog_evaluability",
    "paper_case_value",
    "method_discriminativeness",
)
OBJECT_SIGNATURES = {
    "amazon_clothing": (
        ("bodysuit", ("bodysuit",)), ("sock", ("sock",)), ("jeans", ("jean", "jeans")),
        ("flat", ("flat", "flats", "ballerina")), ("boot", ("boot", "boots")),
        ("polo", ("polo",)), ("pump", ("pump", "pumps")), ("hat", ("hat", "hats")),
        ("shoe", ("shoe", "shoes", "footwear")),
    ),
    "amazon_beauty": (
        ("sunscreen", ("sunscreen",)), ("lotion", ("lotion", "moisturizer", "moisturiser")),
        ("blemish treatment", ("blemish", "acne", "spot treatment")), ("hair treatment", ("hair", "treatment")),
        ("face treatment", ("face treatment", "treatment serum", "serum")), ("conditioner", ("conditioner",)),
        ("scrub", ("scrub", "exfoliat")), ("self tanner", ("self-tanner", "self tanner", "bronzer")),
    ),
    "amazon_music": (
        ("vinyl", ("vinyl",)), ("album", ("album",)), ("collection", ("collection",)),
        ("track", ("track", "single")), ("tune", ("tune", "song")), ("music", ("music",)),
    ),
}


@dataclass(frozen=True)
class DevelopmentBlueprint:
    case_id: str
    domain: str
    family: str
    source_item_id: str
    intent_topology: str
    transition_rationale: str
    expected_state_path: tuple[str, ...]
    required_decision_axis: str


# These are intentionally curated rather than a random family assignment.  The
# source query contributes the complex-intent setting; the rationale fixes the
# exact decision event that a reviewer must be able to justify.
DEVELOPMENT_BLUEPRINTS: tuple[DevelopmentBlueprint, ...] = (
    DevelopmentBlueprint("mtdev-clothing-direct-sun-hat", "amazon_clothing", "direct", "B000ALL1ZW", "scenario_to_attribute", "Sun protection, outdoor use, and a sun-hat target already identify the target use; asking a generic extra question should be penalized.", ("InitialDemand", "RecommendationReady", "Success"), "sun protection and outdoor hat use"),
    DevelopmentBlueprint("mtdev-clothing-direct-petite-jeans", "amazon_clothing", "direct", "B001188FZC", "constraint_satisfaction", "Petite fit and practical comfort are explicit; this is a control against unnecessary elicitation.", ("InitialDemand", "RecommendationReady", "Success"), "petite fit and comfort"),
    DevelopmentBlueprint("mtdev-clothing-progressive-running-sock", "amazon_clothing", "progressive", "B004BKIPHK", "scenario_to_attribute", "The sports/outdoor running setting is explicit, but cushioning-versus-breathability tolerance is a stable decision-critical preference not stated in the request.", ("InitialDemand", "ConstraintMissing", "PreferenceClarification", "RecommendationReady", "Success"), "cushioning versus breathability"),
    DevelopmentBlueprint("mtdev-clothing-progressive-park-flats", "amazon_clothing", "progressive", "B00KM9NV0U", "occasion_conversion", "A park family gathering requires a meaningful choice between dressiness and all-day comfort; the user has not expressed which dominates.", ("InitialDemand", "AmbiguousIntent", "PreferenceClarification", "RecommendationReady", "Success"), "dressiness versus all-day comfort"),
    DevelopmentBlueprint("mtdev-clothing-ambiguous-bodysuit", "amazon_clothing", "ambiguous", "B000FRSOQU", "multi_occasion", "Family gatherings and playdates support two plausible interpretations: presentable occasion wear or rapid-change active wear.", ("InitialDemand", "AmbiguousIntent", "PreferenceClarification", "RecommendationReady", "Success"), "occasion priority: presentable versus easy-change"),
    DevelopmentBlueprint("mtdev-clothing-ambiguous-outdoor-formal-boot", "amazon_clothing", "ambiguous", "B006H1T3SQ", "multi_occasion", "The request combines rugged outdoor use and semi-formal gatherings, so the Agent must establish the primary occasion before retrieving.", ("InitialDemand", "AmbiguousIntent", "PreferenceClarification", "RecommendationReady", "Success"), "outdoor ruggedness versus semi-formal appearance"),
    DevelopmentBlueprint("mtdev-clothing-revision-layering-polo", "amazon_clothing", "revision", "B00AO4HUDC", "scenario_to_attribute", "A polo can satisfy the original family-gathering interpretation; after seeing an option, the user can coherently revise toward layerability under a blazer rather than merely restate the query.", ("InitialDemand", "RecommendationReady", "NegativeFeedback", "RepairReplan", "Success"), "layering bulk and collar structure"),
    DevelopmentBlueprint("mtdev-clothing-revision-wedding-pumps", "amazon_clothing", "revision", "B0085MI7PK", "occasion_conversion", "Formal wedding and professional-event suitability admits a first recommendation, after which long walking/travel comfort can become newly decisive.", ("InitialDemand", "RecommendationReady", "NegativeFeedback", "RepairReplan", "Success"), "formal style versus prolonged-wear comfort"),
    DevelopmentBlueprint("mtdev-clothing-conflict-warm-running-shoe", "amazon_clothing", "conflict", "B0087QYI02", "multi_occasion", "Outdoor running, casual outings, and warmth in cooler weather create a genuine lightweight-performance versus insulation/protection tension.", ("InitialDemand", "PreferenceConflict", "PreferenceClarification", "RecommendationReady", "Success"), "running agility versus warmth/protection"),
    DevelopmentBlueprint("mtdev-clothing-conflict-formal-casual-pumps", "amazon_clothing", "conflict", "B003UAS4D2", "multi_occasion", "The same pumps are requested for formal gatherings and casual family outings; the user must state which setting controls the compromise.", ("InitialDemand", "PreferenceConflict", "PreferenceClarification", "RecommendationReady", "Success"), "formal polish versus casual walkability"),
    DevelopmentBlueprint("mtdev-beauty-direct-aftersun-lotion", "amazon_beauty", "direct", "B000P9DRZQ", "scenario_to_attribute", "After-sun soothing and irritated-skin recovery are explicit, so the control trajectory rewards direct evidence-grounded recommendation.", ("InitialDemand", "RecommendationReady", "Success"), "after-sun soothing"),
    DevelopmentBlueprint("mtdev-beauty-direct-sensitive-sunscreen", "amazon_beauty", "direct", "B00C0Z5CYW", "constraint_satisfaction", "UV defense for sensitive skin is sufficiently specified; extra questions should not be rewarded absent a real safety uncertainty.", ("InitialDemand", "RecommendationReady", "Success"), "UV defense and sensitive-skin compatibility"),
    DevelopmentBlueprint("mtdev-beauty-progressive-blemish-gel", "amazon_beauty", "progressive", "B0001EKYTM", "scenario_to_attribute", "Unclogging pores for oily skin does not establish the user's tolerance for active treatment strength, which changes the suitable blemish-treatment evidence.", ("InitialDemand", "ConstraintMissing", "PreferenceClarification", "RecommendationReady", "Success"), "active treatment strength versus gentleness"),
    DevelopmentBlueprint("mtdev-beauty-progressive-hair-recovery", "amazon_beauty", "progressive", "B0037TM252", "multi_constraint", "Stress-related hair recovery can prioritize active regrowth support or a nourishing low-irritation routine; a targeted question is necessary.", ("InitialDemand", "ConstraintMissing", "PreferenceClarification", "RecommendationReady", "Success"), "regrowth focus versus nourishing routine"),
    DevelopmentBlueprint("mtdev-beauty-ambiguous-delicate-treatment", "amazon_beauty", "ambiguous", "B000WLXWUC", "ambiguity", "Elasticity treatment for delicate skin is ambiguous between a mild maintenance routine and a more intensive anti-aging intervention.", ("InitialDemand", "AmbiguousIntent", "PreferenceClarification", "RecommendationReady", "Success"), "maintenance gentleness versus intensive elasticity treatment"),
    DevelopmentBlueprint("mtdev-beauty-ambiguous-scalp-treatment", "amazon_beauty", "ambiguous", "B003UK84P0", "scenario_to_attribute", "Scalp irritation plus a brand preference leaves open whether immediate soothing or a leave-in treatment format matters more.", ("InitialDemand", "AmbiguousIntent", "PreferenceClarification", "RecommendationReady", "Success"), "immediate soothing versus leave-in routine compatibility"),
    DevelopmentBlueprint("mtdev-beauty-revision-aging-acne", "amazon_beauty", "revision", "B002WTC384", "multi_constraint", "A moisturizer can initially balance anti-aging and acne-prone skin; after a concrete option, the user may coherently elevate breakout control as the non-negotiable criterion.", ("InitialDemand", "RecommendationReady", "NegativeFeedback", "RepairReplan", "Success"), "anti-aging benefit versus acne control"),
    DevelopmentBlueprint("mtdev-beauty-revision-curly-conditioner", "amazon_beauty", "revision", "B00AZIZHU0", "scenario_to_attribute", "Moisture maintenance for naturally curly hair is explicit; after seeing a rich conditioner, the user can revise toward a non-weighing daily-use texture.", ("InitialDemand", "RecommendationReady", "NegativeFeedback", "RepairReplan", "Success"), "deep moisture versus lightweight daily use"),
    DevelopmentBlueprint("mtdev-beauty-conflict-gentle-scrub", "amazon_beauty", "conflict", "B0000632EN", "multi_constraint", "The request explicitly combines gentle sensitive-skin use and effective exfoliation, a real strength-versus-tolerance conflict.", ("InitialDemand", "PreferenceConflict", "PreferenceClarification", "RecommendationReady", "Success"), "exfoliation efficacy versus sensitive-skin gentleness"),
    DevelopmentBlueprint("mtdev-beauty-conflict-sensitive-selftanner", "amazon_beauty", "conflict", "B002HEW8EI", "multi_constraint", "A natural-looking self-tan and sensitive-skin safety compete when stronger color payoff may increase formulation concerns.", ("InitialDemand", "PreferenceConflict", "PreferenceClarification", "RecommendationReady", "Success"), "natural color payoff versus sensitive-skin safety"),
    DevelopmentBlueprint("mtdev-music-direct-heritage", "amazon_music", "direct", "B000I5X82O", "scenario_to_attribute", "British heritage for a cultural identity event provides a clear thematic target; it is a control against needless preference questioning.", ("InitialDemand", "RecommendationReady", "Success"), "British heritage evidence"),
    DevelopmentBlueprint("mtdev-music-direct-complex-listening", "amazon_music", "direct", "B0000C3I5V", "scenario_to_attribute", "Complex structures and experimental sounds for a thought-provoking evening define a sufficiently specific focused-listening need.", ("InitialDemand", "RecommendationReady", "Success"), "experimental complexity"),
    DevelopmentBlueprint("mtdev-music-progressive-vinyl-dance", "amazon_music", "progressive", "B09HVTKLPQ", "multi_constraint", "A catchy dance-party tune on vinyl leaves a meaningful primary decision between physical format fidelity and dance-floor energy.", ("InitialDemand", "ConstraintMissing", "PreferenceClarification", "RecommendationReady", "Success"), "vinyl format versus danceability"),
    DevelopmentBlueprint("mtdev-music-progressive-country-pop", "amazon_music", "progressive", "B00000DG1R", "ambiguity", "The current request is country hits while the profile mentions pop/dance; the Agent should ask whether to follow the event request or the historical listening preference.", ("InitialDemand", "AmbiguousIntent", "PreferenceClarification", "RecommendationReady", "Success"), "event genre versus historical preference"),
    DevelopmentBlueprint("mtdev-music-ambiguous-nostalgic-barbecue", "amazon_music", "ambiguous", "B0017LFKOC", "multi_occasion", "A nostalgic backyard barbecue may need unobtrusive background music or a focal singalong set; both are plausible without clarification.", ("InitialDemand", "AmbiguousIntent", "PreferenceClarification", "RecommendationReady", "Success"), "background listening versus focal nostalgic set"),
    DevelopmentBlueprint("mtdev-music-ambiguous-relaxing-european", "amazon_music", "ambiguous", "B00Z8CEGIC", "scenario_to_attribute", "A peaceful evening with European flair is ambiguous between a relaxation-first mood and culturally specific repertoire.", ("InitialDemand", "AmbiguousIntent", "PreferenceClarification", "RecommendationReady", "Success"), "relaxing mood versus European repertoire"),
    DevelopmentBlueprint("mtdev-music-revision-disco", "amazon_music", "revision", "B00004R7FV", "multi_constraint", "A first option can match disco/dance energy; after seeing it, the user can coherently revise toward an actual listening format for a planned party rather than broad genre similarity.", ("InitialDemand", "RecommendationReady", "NegativeFeedback", "RepairReplan", "Success"), "party-ready dance energy versus preferred listening format"),
    DevelopmentBlueprint("mtdev-music-revision-nostalgic-romantic", "amazon_music", "revision", "B000B5IPLK", "scenario_to_attribute", "A nostalgic romantic collection can be initially suitable; after a candidate is shown, the user may specify whether the evening needs intimate vocals or unobtrusive ambience.", ("InitialDemand", "RecommendationReady", "NegativeFeedback", "RepairReplan", "Success"), "intimate vocals versus unobtrusive ambience"),
    DevelopmentBlueprint("mtdev-music-conflict-uplifting-relaxing", "amazon_music", "conflict", "B0BCHDTLDK", "multi_constraint", "The query explicitly asks for uplifting and relaxing smooth jazz, requiring a priority decision instead of assuming one mood dominates.", ("InitialDemand", "PreferenceConflict", "PreferenceClarification", "RecommendationReady", "Success"), "uplifting energy versus relaxation"),
    DevelopmentBlueprint("mtdev-music-conflict-youth-nostalgic-vinyl", "amazon_music", "conflict", "B08XTS2HW1", "multi_constraint", "Youth-culture energy and nostalgic vinyl listening create a genuine temporal/style tension that must be negotiated.", ("InitialDemand", "PreferenceConflict", "PreferenceClarification", "RecommendationReady", "Success"), "youthful energy versus nostalgic vinyl character"),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _clean(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _source_maps(data_root: Path) -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    query_map: dict[tuple[str, str], dict[str, str]] = {}
    metadata_map: dict[tuple[str, str], dict[str, str]] = {}
    domains = sorted({blueprint.domain for blueprint in DEVELOPMENT_BLUEPRINTS})
    needed_ids = {(blueprint.domain, blueprint.source_item_id) for blueprint in DEVELOPMENT_BLUEPRINTS}
    for domain in domains:
        for row in _read_csv(data_root / domain / "query_data1.csv"):
            key = (domain, _clean(row.get("id")))
            if key in needed_ids and key not in query_map:
                query_map[key] = row
        for row in _read_csv(data_root / domain / "metadata.csv"):
            key = (domain, _clean(row.get("id")))
            if key in needed_ids and key not in metadata_map:
                metadata_map[key] = row
    return query_map, metadata_map


def make_case_inputs(data_root: Path) -> list[dict[str, Any]]:
    query_map, metadata_map = _source_maps(data_root)
    inputs: list[dict[str, Any]] = []
    for blueprint in DEVELOPMENT_BLUEPRINTS:
        key = (blueprint.domain, blueprint.source_item_id)
        source = query_map.get(key)
        if source is None:
            raise ValueError(f"Missing legacy source query for {blueprint.case_id}: {key}")
        anchor = metadata_map.get(key, {})
        inputs.append(
            {
                "version": DEVELOPMENT_VERSION,
                "case_id": blueprint.case_id,
                "domain": blueprint.domain,
                "family": blueprint.family,
                "intent_topology": blueprint.intent_topology,
                "transition_rationale": blueprint.transition_rationale,
                "expected_state_path": list(blueprint.expected_state_path),
                "required_decision_axis": blueprint.required_decision_axis,
                "legacy_source": {
                    "item_id": blueprint.source_item_id,
                    "query": _clean(source.get("query")),
                    "targets": _clean(source.get("targets")),
                    "category": _clean(source.get("category")),
                    "user_id": _clean(source.get("user_id")),
                },
                "anchor_item": {
                    "title": _clean(anchor.get("title")),
                    "category": _clean(anchor.get("category")),
                    "description": _clean(anchor.get("description"))[:700],
                    "price": _clean(anchor.get("price")),
                },
            }
        )
    return inputs


def attach_catalog_evidence(data_root: Path, case_inputs: list[dict[str, Any]], *, per_case: int = 12) -> list[dict[str, Any]]:
    """Attach a small, full-catalog-derived evidence menu to every case.

    The author may only cite ids and text spans from this menu.  This separates
    historical query provenance from verifiable catalogue evidence and prevents
    an LLM judge from approving invented properties such as an absent UPF claim.
    """

    indexes: dict[str, CatalogSearchIndex] = {}
    for domain in sorted({case_input["domain"] for case_input in case_inputs}):
        print(f"阶段进度：正在建立 {domain} 的开发集证据检索索引。", flush=True)
        indexes[domain] = CatalogSearchIndex(_catalog_rows(data_root, domain))
    enriched: list[dict[str, Any]] = []
    for case_input in case_inputs:
        source = case_input["legacy_source"]
        query_variants = (
            source["query"],
            f"{source['targets']} {case_input['required_decision_axis']}",
            case_input["required_decision_axis"],
        )
        index = indexes[case_input["domain"]]
        seen: dict[int, float] = {}
        for query in query_variants:
            for position, score in index.search(query, 24):
                seen[position] = max(seen.get(position, 0.0), score)
        anchor_position = index.position_by_item_id.get(source["item_id"])
        if anchor_position is not None:
            seen.setdefault(anchor_position, 0.0)
        ranked = sorted(seen, key=lambda position: (-seen[position], index.rows[position].item_id))
        # The historical anchor is evidence provenance rather than an automatic
        # positive.  It nevertheless must remain available to the author: it is
        # frequently the only catalog row that preserves an explicit legacy
        # demographic/category conjunction such as "men's big & tall".
        if anchor_position is not None:
            ranked = [anchor_position] + [position for position in ranked if position != anchor_position]
        ranked = ranked[:per_case]
        evidence = []
        for position in ranked:
            item = index.rows[position]
            evidence.append(
                {
                    "item_id": item.item_id,
                    "title": item.title,
                    "category": item.category,
                    "description": item.description[:700],
                    "source_anchor": item.item_id == source["item_id"],
                }
            )
        enriched.append({**case_input, "catalog_evidence_menu": evidence})
    return enriched


def _candidate_text(candidate: dict[str, Any]) -> str:
    return " ".join(str(candidate.get(field, "")) for field in ("title", "category", "description")).casefold()


def _has_term(text: str, term: str) -> bool:
    """Use word boundaries for demographic labels (``men`` is in ``women``)."""

    return re.search(rf"(?<![a-z]){re.escape(term.casefold())}(?![a-z])", text.casefold()) is not None


def _candidate_matches_gender(candidate: dict[str, Any], expected: str) -> bool:
    """Require unambiguous product evidence for a gender-restricted request.

    Amazon category paths can occasionally contain both gender labels after
    taxonomy aggregation.  A title/description conflict is decisive; otherwise
    a clean category label is admissible when the title itself is neutral.
    """

    aliases = ("men", "men's", "mens", "male") if expected == "men" else ("women", "women's", "womens", "female")
    opposite = ("women", "women's", "womens", "female") if expected == "men" else ("men", "men's", "mens", "male")
    title_text = " ".join(str(candidate.get(field, "")) for field in ("title", "description")).casefold()
    category_text = str(candidate.get("category", "")).casefold()
    if any(_has_term(title_text, alias) for alias in opposite):
        return False
    if any(_has_term(title_text, alias) for alias in aliases):
        return True
    return any(_has_term(category_text, alias) for alias in aliases) and not any(
        _has_term(category_text, alias) for alias in opposite
    )


def _candidate_matches_object(candidate: dict[str, Any], aliases: tuple[str, ...]) -> bool:
    text = _candidate_text(candidate)
    for alias in aliases:
        if _has_term(text, alias):
            return True
        # Catalog taxonomies commonly pluralize the product noun while legacy
        # queries use its singular form (sock/Socks, polo/Polos).
        if not alias.endswith("s") and _has_term(text, f"{alias}s"):
            return True
    return False


def _required_object_aliases(case_input: dict[str, Any]) -> tuple[str, ...]:
    source = case_input["legacy_source"]
    text = f"{source.get('targets', '')} {source.get('query', '')}".casefold()
    matches: list[tuple[int, tuple[str, ...]]] = []
    for _, aliases in OBJECT_SIGNATURES[case_input["domain"]]:
        if any(alias in text for alias in aliases):
            matches.append((max(len(alias) for alias in aliases), aliases))
    return max(matches, default=(0, ()), key=lambda value: value[0])[1]


def _gender_requirement(case_input: dict[str, Any]) -> str | None:
    text = f"{case_input['legacy_source'].get('targets', '')} {case_input['legacy_source'].get('query', '')}".casefold()
    if re.search(r"\b(men|men's|mens|male)\b", text) and not re.search(r"\b(women|women's|womens|female)\b", text):
        return "men"
    if re.search(r"\b(women|women's|womens|female)\b", text) and not re.search(r"\b(men|men's|mens|male)\b", text):
        return "women"
    return None


def _priority_terms(transition: dict[str, Any]) -> list[str]:
    resolved = transition.get("resolved_priority", {})
    if not isinstance(resolved, dict):
        return []
    terms = resolved.get("required_catalog_terms", [])
    return [str(term).casefold().strip() for term in terms if len(str(term).strip()) >= 3]


def local_contract_errors(case_input: dict[str, Any], trajectory: dict[str, Any]) -> list[str]:
    """Reject unsupported evidence before an LLM judge can be persuaded by prose."""

    errors: list[str] = []
    if trajectory.get("public_initial_query") != case_input["legacy_source"]["query"]:
        errors.append("public_initial_query must copy the legacy source query exactly")
    contract = trajectory.get("evaluation_contract", {})
    positives = contract.get("positive_evidence", [])
    negatives = contract.get("hard_negative_evidence", [])
    candidates = {candidate["item_id"]: candidate for candidate in case_input.get("catalog_evidence_menu", [])}
    positive_candidates: list[dict[str, Any]] = []
    positive_ids: set[str] = set()
    if not positives:
        errors.append("evaluation_contract requires at least one positive_evidence record")
    if not negatives:
        errors.append("evaluation_contract requires at least one hard_negative_evidence record")
    for label, values in (("positive_evidence", positives), ("hard_negative_evidence", negatives)):
        if not isinstance(values, list):
            errors.append(f"{label} must be a list")
            continue
        for value in values:
            if not isinstance(value, dict):
                errors.append(f"{label} entries must be objects")
                continue
            item_id = value.get("item_id", "")
            candidate = candidates.get(item_id)
            if candidate is None:
                errors.append(f"{label} cites item outside catalog_evidence_menu: {item_id}")
                continue
            if label == "positive_evidence":
                positive_candidates.append(candidate)
                positive_ids.add(item_id)
                spans = value.get("supporting_spans", [])
                if not isinstance(spans, list) or not spans:
                    errors.append(f"positive item {item_id} requires supporting_spans")
                    continue
                text = _candidate_text(candidate)
                for span in spans:
                    normalized = str(span).casefold().strip()
                    if len(normalized) < 3 or normalized not in text:
                        errors.append(f"positive evidence span is absent from item {item_id}: {span}")
    for value in negatives if isinstance(negatives, list) else []:
        if isinstance(value, dict) and value.get("item_id") in positive_ids:
            errors.append(f"hard negative duplicates a positive candidate: {value.get('item_id')}")
    object_aliases = _required_object_aliases(case_input)
    if object_aliases:
        for candidate in positive_candidates:
            if not _candidate_matches_object(candidate, object_aliases):
                errors.append(
                    f"positive candidate {candidate['item_id']} does not match the requested object aliases: {object_aliases}"
                )
    gender = _gender_requirement(case_input)
    if gender and positive_candidates:
        for candidate in positive_candidates:
            if not _candidate_matches_gender(candidate, gender):
                errors.append(f"positive candidate {candidate['item_id']} conflicts with requested gender: {gender}")
    family = case_input["family"]
    transition = trajectory.get("transition", {})
    action = str(transition.get("agent_action_required", "")).strip()
    reply = str(transition.get("user_reply", "")).strip()
    state_update = str(transition.get("state_update", "")).strip()
    max_system_turns = contract.get("max_system_turns")
    latent = trajectory.get("private_state", {}).get("latent_stable_preference", {})
    if family == "direct":
        if max_system_turns != 0:
            errors.append("direct trajectory must set max_system_turns to 0")
        if reply:
            errors.append("direct trajectory must not fabricate a user reply")
    else:
        if max_system_turns != 1:
            errors.append("non-direct trajectory must set max_system_turns to 1")
        if not action or action.casefold().startswith("none") or not reply or not state_update:
            errors.append("non-direct trajectory requires an actual agent action, user reply, and state update")
        if not isinstance(latent, dict) or not str(latent.get("name", "")).strip():
            errors.append("non-direct trajectory requires a latent stable preference name")
    if family in {"progressive", "ambiguous", "revision", "conflict"}:
        terms = _priority_terms(transition)
        if not terms:
            errors.append("non-direct trajectory requires resolved_priority.required_catalog_terms")
        else:
            for candidate in positive_candidates:
                if not any(_has_term(_candidate_text(candidate), term) for term in terms):
                    errors.append(f"positive candidate {candidate['item_id']} does not support the selected priority terms: {terms}")
    return errors


def _json_from_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response did not contain a JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def openai_json_caller(
    base_url: str,
    api_key: str,
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1400,
) -> Callable[[list[dict[str, str]]], dict[str, Any]]:
    """Build a caller without persisting credentials in code or artifacts.

    A transient malformed response must not turn into manual benchmark work.
    The same fixed model is retried before an item can be rejected.
    """

    def call(messages: list[dict[str, str]]) -> dict[str, Any]:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)
        last_error: Exception | None = None
        for _ in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=120,
                )
                return _json_from_text(response.choices[0].message.content or "")
            except Exception as error:
                last_error = error
        assert last_error is not None
        raise last_error

    return call


def _trajectory_output_schema(family: str) -> dict[str, Any]:
    """Return the full contract on both author and repair calls.

    A repair is a replacement trajectory, not a partial diff.  Keeping the
    schema in one helper prevents the repair prompt from silently omitting
    evaluation evidence or multi-turn state fields.
    """

    return {
        "public_initial_query": "copy the legacy source query exactly",
        "private_state": {
            "visible_facts": ["at most 4 short query-grounded phrases"],
            "latent_stable_preference": {"name": "one short missing decision axis, or empty for direct"},
        },
        "transition": {
            "family": family,
            "agent_action_required": "one short question / repair / negotiation; empty for direct",
            "user_reply": "one short deterministic reply; empty for direct",
            "state_update": "one short state label",
            "resolved_priority": {"label": "short selected side", "required_catalog_terms": ["one or two exact terms in the positive item"]},
        },
        "evaluation_contract": {
            "positive_evidence": [{"item_id": "from catalog_evidence_menu", "supporting_spans": ["exact title/category/description spans"]}],
            "hard_negative_evidence": [{"item_id": "from catalog_evidence_menu", "missing_or_conflicting_requirement": ""}],
            "max_system_turns": "0 for direct; exactly 1 for progressive, ambiguous, revision, or conflict",
        },
        "thought_pattern": {
            "task_description": "one concise retrieval key",
            "solution_description": "one concise transferable insight",
            "thought_template": ["exactly 3 short SOP steps"],
        },
    }


def author_messages(case_input: dict[str, Any]) -> list[dict[str, str]]:
    contract = _trajectory_output_schema(case_input["family"])
    return [
        {
            "role": "system",
            "content": (
                "You are authoring a publication-quality benchmark trajectory for a multi-turn complex-intent recommender. "
                "Produce a complete trajectory for every supplied blueprint; do not return a refusal, a diagnostic, or empty evidence lists. "
                "Never invent a catalog item or claim a legacy query said something it did not say. "
                "You may cite only item ids from catalog_evidence_menu and every positive supporting span must appear verbatim in that item's title, category, or description. "
                "Choose at least one positive and one hard-negative item from that menu. "
                "Use short phrases only: no self-check, no explanation paragraph, and no field longer than one sentence. "
                "The Thought Pattern must retain Task Description, Solution Description, and a three-step Thought Template. Reply with exactly one JSON object."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "case_input": case_input,
                    "required_output_schema": contract,
                    "family_rules": {
                        "direct": "Use max_system_turns=0. Do not add a hidden decision-critical constraint. Explain why asking is redundant.",
                        "progressive": "Use max_system_turns=1. Reveal exactly one initially absent stable preference only after a targeted question. Populate resolved_priority with catalog terms that every final positive actually contains.",
                        "ambiguous": "Use max_system_turns=1. Name two plausible interpretations and ask one discriminating question. Populate resolved_priority with catalog terms that every final positive actually contains.",
                        "revision": "Use max_system_turns=1. The revision must be a plausible response to seeing a first otherwise-valid option, and must alter ranking. Populate resolved_priority with catalog terms that every repaired positive actually contains.",
                        "conflict": "Use max_system_turns=1. Name two competing priorities and ask the user to select which one dominates. Populate resolved_priority with catalog terms that every final positive actually contains.",
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]


def _compact_review_case(case_input: dict[str, Any]) -> dict[str, Any]:
    """Keep strong-model review short; evidence spans are checked locally."""

    return {
        key: value
        for key, value in case_input.items()
        if key not in {"catalog_evidence_menu", "anchor_item"}
    } | {
        "catalog_evidence_ids": [
            {"item_id": candidate["item_id"], "title": candidate["title"], "category": candidate["category"]}
            for candidate in case_input.get("catalog_evidence_menu", [])
        ]
    }


def _canonicalize_trajectory(case_input: dict[str, Any], trajectory: dict[str, Any]) -> dict[str, Any]:
    """The public initial utterance is immutable source data, never model prose."""

    if not isinstance(trajectory, dict):
        return trajectory
    canonical = dict(trajectory)
    canonical["public_initial_query"] = case_input["legacy_source"]["query"]
    return canonical


def judge_messages(case_input: dict[str, Any], trajectory: dict[str, Any]) -> list[dict[str, str]]:
    schema = {
        "verdict": "admit | repair | reject",
        "scores": {dimension: "integer 1-5" for dimension in QUALITY_DIMENSIONS},
        "failure_reasons": ["specific issue"],
        "required_repairs": ["actionable repair"],
        "showcase_sentence": "why this is an intelligible paper case",
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an independent, adversarial reviewer for a TOIS benchmark. Judge whether the proposed trajectory is a logically necessary, "
                "reproducible multi-turn extension of the given legacy query. Do not reward polished prose. A score below 4 on any dimension requires repair or rejection. Reply with exactly one JSON object."
            ),
        },
        {"role": "user", "content": json.dumps({"case_input": _compact_review_case(case_input), "trajectory": trajectory, "required_output_schema": schema}, ensure_ascii=False)},
    ]


def repair_messages(case_input: dict[str, Any], trajectory: dict[str, Any], judgement: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are repairing a benchmark trajectory after independent review. Preserve source-query fidelity and the family mechanism. "
                "Apply every required repair using only the supplied catalogue evidence. Always return a complete replacement trajectory JSON using the original author schema; "
                "never return a refusal, an unrepairable marker, or an incomplete schema."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "case_input": _compact_review_case(case_input),
                    "trajectory": trajectory,
                    "judgement": judgement,
                    "required_output_schema": _trajectory_output_schema(case_input["family"]),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _admitted(judgement: dict[str, Any]) -> bool:
    scores = judgement.get("scores", {})
    return judgement.get("verdict") == "admit" and all(int(scores.get(dimension, 0)) >= 4 for dimension in QUALITY_DIMENSIONS)


def build_development_cases(
    data_root: Path,
    author_caller: Callable[[list[dict[str, str]]], dict[str, Any]],
    *,
    judge_caller: Callable[[list[dict[str, str]]], dict[str, Any]] | None = None,
    max_repairs: int = 1,
    case_limit: int = 0,
    case_ids: set[str] | None = None,
    checkpoint: Callable[[list[dict[str, Any]]], None] | None = None,
    generation_only: bool = False,
) -> list[dict[str, Any]]:
    """Build records, optionally skipping per-case review for a runnable set."""

    judge_caller = judge_caller or author_caller
    case_inputs = make_case_inputs(data_root)
    if case_ids:
        case_inputs = [case_input for case_input in case_inputs if case_input["case_id"] in case_ids]
        missing = case_ids.difference(case_input["case_id"] for case_input in case_inputs)
        if missing:
            raise ValueError(f"Unknown development case ids: {sorted(missing)}")
    if case_limit:
        case_inputs = case_inputs[:case_limit]
    case_inputs = attach_catalog_evidence(data_root, case_inputs)
    records: list[dict[str, Any]] = []
    for position, case_input in enumerate(case_inputs, start=1):
        print(f"阶段进度：正在构造并审校开发轨迹 {position}/{len(case_inputs)}：{case_input['case_id']}。", flush=True)
        try:
            draft = _canonicalize_trajectory(case_input, author_caller(author_messages(case_input)))
            if generation_only:
                records.append(
                    {
                        "version": DEVELOPMENT_VERSION,
                        "case_input": case_input,
                        "trajectory": draft,
                        "initial_draft": draft,
                        "final_judgement": {"verdict": "not_run", "scores": {}},
                        "repair_history": [],
                        "local_contract_errors": local_contract_errors(case_input, draft),
                        "admitted": True,
                        "generation_status": "generated_without_per_case_gate",
                    }
                )
                if checkpoint:
                    checkpoint(records)
                continue
            draft_local_errors = local_contract_errors(case_input, draft)
            judgement = (
                {"verdict": "repair", "scores": {}, "failure_reasons": draft_local_errors, "required_repairs": draft_local_errors}
                if draft_local_errors
                else judge_caller(judge_messages(case_input, draft))
            )
        except Exception as exc:
            print(f"阶段告警：{case_input['case_id']} 的作者或首轮审校调用失败：{type(exc).__name__}。", flush=True)
            records.append(
                {
                    "version": DEVELOPMENT_VERSION,
                    "case_input": case_input,
                    "trajectory": {},
                    "initial_draft": {},
                    "final_judgement": {},
                    "repair_history": [],
                    "build_error": f"{type(exc).__name__}: {exc}",
                    "admitted": False,
                }
            )
            if checkpoint:
                checkpoint(records)
            continue
        repairs: list[dict[str, Any]] = []
        current = draft
        local_errors = local_contract_errors(case_input, current)
        try:
            while not _admitted(judgement) and len(repairs) < max_repairs:
                repaired = _canonicalize_trajectory(case_input, author_caller(repair_messages(case_input, current, judgement)))
                repairs.append({"input_judgement": judgement, "repaired_trajectory": repaired})
                if repaired.get("unrepairable"):
                    current = repaired
                    break
                current = repaired
                local_errors = local_contract_errors(case_input, current)
                judgement = (
                    {"verdict": "repair", "scores": {}, "failure_reasons": local_errors, "required_repairs": local_errors}
                    if local_errors
                    else judge_caller(judge_messages(case_input, current))
                )
        except Exception as exc:
            print(f"阶段告警：{case_input['case_id']} 的修订或复审调用失败：{type(exc).__name__}。", flush=True)
            judgement = {"verdict": "reject", "scores": {}, "failure_reasons": [f"review_workflow_error: {type(exc).__name__}: {exc}"]}
        records.append(
            {
                "version": DEVELOPMENT_VERSION,
                "case_input": case_input,
                "trajectory": current,
                "initial_draft": draft,
                "final_judgement": judgement,
                "repair_history": repairs,
                "local_contract_errors": local_contract_errors(case_input, current),
                "admitted": _admitted(judgement) and not local_contract_errors(case_input, current),
            }
        )
        if checkpoint:
            checkpoint(records)
    return records


def write_development_cases(records: Iterable[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    values = list(records)
    (output_dir / "development_trajectories.jsonl").write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values), encoding="utf-8"
    )
    admitted = [value for value in values if value["admitted"]]
    generation_only = [value for value in values if value.get("generation_status") == "generated_without_per_case_gate"]
    summary = {
        "version": DEVELOPMENT_VERSION,
        "case_count": len(values),
        "admitted_count": len(admitted),
        "rejected_or_repair_failed_count": len(values) - len(admitted),
        "generation_only_count": len(generation_only),
        "per_case_quality_gate": "disabled" if generation_only else "enabled",
        "families": {family: sum(value["case_input"]["family"] == family for value in values) for family in sorted({value["case_input"]["family"] for value in values})},
        "domains": {domain: sum(value["case_input"]["domain"] == domain for value in values) for domain in sorted({value["case_input"]["domain"] for value in values})},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def render_development_audit(records: Iterable[dict[str, Any]]) -> str:
    """Render every trajectory for a human paper-case audit, without hiding failures."""

    values = list(records)
    lines = [
        "# Multi-turn development-set quality audit",
        "",
        "This file is an inspection surface, not a result table. A case is final only after its evidence, transition, and Thought Pattern are individually readable here.",
        "",
        f"- Records currently persisted: {len(values)}",
        f"- Admitted: {sum(value.get('admitted', False) for value in values)}",
        "",
    ]
    for value in values:
        case = value["case_input"]
        trajectory = value.get("trajectory", {})
        contract = trajectory.get("evaluation_contract", {}) if isinstance(trajectory, dict) else {}
        judgement = value.get("final_judgement", {})
        status = "GENERATED" if value.get("generation_status") == "generated_without_per_case_gate" else ("ADMIT" if value.get("admitted") else "NOT ADMITTED")
        lines.extend(
            [
                f"## {case['case_id']} — {case['domain']} / {case['family']} / {status}",
                "",
                f"**Initial query:** {case['legacy_source']['query']}",
                f"**Inherited intent topology:** {case['intent_topology']}",
                f"**Decision axis:** {case['required_decision_axis']}",
                f"**Transition rationale:** {case['transition_rationale']}",
                f"**Expected state path:** {' → '.join(case['expected_state_path'])}",
                f"**Local evidence errors:** {value.get('local_contract_errors', value.get('build_error', []))}",
                f"**Judge verdict/scores:** {judgement.get('verdict')} / {judgement.get('scores')}",
                "",
                "### Evidence cited by the trajectory",
                "",
            ]
        )
        for evidence_type in ("positive_evidence", "hard_negative_evidence"):
            lines.append(f"- {evidence_type}: {contract.get(evidence_type, [])}")
        lines.extend(["", "### State and Pattern", ""])
        lines.append(f"- private_state: {trajectory.get('private_state')}")
        lines.append(f"- transition: {trajectory.get('transition')}")
        lines.append(f"- thought_pattern: {trajectory.get('thought_pattern')}")
        if judgement.get("failure_reasons"):
            lines.append(f"- judge failure reasons: {judgement.get('failure_reasons')}")
        lines.append("")
    return "\n".join(lines)
