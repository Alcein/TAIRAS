"""Frozen large-scale test benchmark built with TAIRA-style global allocation.

The 30 manually curated development trajectories are deliberately not used as
the test population.  This module follows the old TAIRA construction pattern:

1. a model *scores* which multi-turn mechanism naturally fits every legacy
   query;
2. a deterministic allocator enforces balanced global quotas; and only then
3. a model writes the compact, frozen user-state trajectory for the assigned
   mechanism.

The allocator, rather than an individual prompt, controls test-set balance.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from taira_mt.multiturn_benchmark.development import (
    DEVELOPMENT_BLUEPRINTS,
    _canonicalize_trajectory,
    attach_catalog_evidence,
    author_messages,
    local_contract_errors,
)


FROZEN_TEST_VERSION = "taira_mt_frozen_test_v1"
MECHANISMS = ("direct", "progressive", "ambiguous", "revision", "conflict")
DOMAINS = ("amazon_clothing", "amazon_beauty", "amazon_music")

STATE_PATHS: dict[str, tuple[str, ...]] = {
    "direct": ("InitialDemand", "RecommendationReady", "Success"),
    "progressive": ("InitialDemand", "ConstraintMissing", "PreferenceClarification", "RecommendationReady", "Success"),
    "ambiguous": ("InitialDemand", "AmbiguousIntent", "PreferenceClarification", "RecommendationReady", "Success"),
    "revision": ("InitialDemand", "RecommendationReady", "NegativeFeedback", "RepairReplan", "Success"),
    "conflict": ("InitialDemand", "PreferenceConflict", "PreferenceClarification", "RecommendationReady", "Success"),
}

DEFAULT_HINTS = {
    "direct": "the initial query already identifies a sufficiently specific recommendation direction",
    "progressive": "one unstated but decision-critical stable preference",
    "ambiguous": "two plausible interpretations of the initial request",
    "revision": "a new decisive requirement after an initially plausible recommendation",
    "conflict": "two explicit priorities that cannot both dominate ranking",
}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _stable_hash(seed: int, *parts: str) -> str:
    return hashlib.sha256("|".join((str(seed), *parts)).encode("utf-8")).hexdigest()


def development_source_ids() -> set[tuple[str, str]]:
    """Keep paper showcase/development queries disjoint from the frozen test set."""

    return {(blueprint.domain, blueprint.source_item_id) for blueprint in DEVELOPMENT_BLUEPRINTS}


def select_source_pool(data_root: Path, *, seed: int, pool_per_domain: int) -> list[dict[str, Any]]:
    """Deterministically sample legacy TAIRA queries before LLM mechanism scoring."""

    excluded = development_source_ids()
    selected: list[dict[str, Any]] = []
    for domain in DOMAINS:
        rows = _read_csv(data_root / domain / "query_data1.csv")
        seen: set[str] = set()
        candidates: list[dict[str, Any]] = []
        for row in rows:
            item_id = _clean(row.get("id"))
            query = _clean(row.get("query"))
            if not item_id or not query or (domain, item_id) in excluded:
                continue
            candidate_key = f"{domain}:{item_id}"
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            candidates.append(
                {
                    "candidate_key": candidate_key,
                    "domain": domain,
                    "source_item_id": item_id,
                    "query": query,
                    "targets": _clean(row.get("targets")),
                    "title": _clean(row.get("title")),
                    "category": _clean(row.get("category")),
                    "description": _clean(row.get("description"))[:500],
                    "user_id": _clean(row.get("user_id")),
                    "legacy_classification": _clean(row.get("classification")),
                }
            )
        candidates.sort(key=lambda value: _stable_hash(seed, value["candidate_key"], value["query"]))
        if len(candidates) < pool_per_domain:
            raise ValueError(f"{domain} has only {len(candidates)} eligible sources; need {pool_per_domain}")
        selected.extend(candidates[:pool_per_domain])
    return selected


def planning_messages(batch: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Ask the LLM for suitability scores, not a final mechanism assignment."""

    compact = [
        {
            "candidate_key": row["candidate_key"],
            "domain": row["domain"],
            "legacy_query": row["query"],
            "target_hint": row["targets"],
            "item_title": row["title"],
            "item_category": row["category"],
            "item_description": row["description"],
            "legacy_query_classification": row["legacy_classification"],
        }
        for row in batch
    ]
    schema = {
        "plans": [
            {
                "candidate_key": "copy exactly",
                "mechanism_scores": {mechanism: "integer 0-100" for mechanism in MECHANISMS},
                "mechanism_hints": {mechanism: "at most 12 words" for mechanism in MECHANISMS},
            }
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "You are planning a frozen multi-turn recommendation benchmark. Do not generate a dialogue and do not choose a final class. "
                "For every supplied legacy TAIRA query, score how naturally it can support each state mechanism. "
                "Scores reflect semantic naturalness, not desired dataset balance. Keep each hint short and grounded in the query/item metadata. "
                "Never invent product properties. Reply with exactly one JSON object."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "mechanism_definitions": {
                        "direct": "The query already has enough decision-critical information; asking would be redundant.",
                        "progressive": "One stable preference is absent and a targeted question would change ranking.",
                        "ambiguous": "The query has two plausible semantic interpretations requiring disambiguation.",
                        "revision": "An initially plausible recommendation can trigger a natural new decisive requirement.",
                        "conflict": "The query contains competing priorities that need explicit prioritization.",
                    },
                    "candidates": compact,
                    "required_output_schema": schema,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _score(value: Any) -> int:
    try:
        return max(0, min(100, int(float(value))))
    except (TypeError, ValueError):
        return 50


def _short_hint(value: Any, fallback: str) -> str:
    text = _clean(value)
    return text[:180] if text else fallback


def normalize_plan_response(response: dict[str, Any], batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        _clean(value.get("candidate_key")): value
        for value in response.get("plans", [])
        if isinstance(value, dict) and _clean(value.get("candidate_key"))
    }
    normalized: list[dict[str, Any]] = []
    for candidate in batch:
        raw = by_key.get(candidate["candidate_key"], {})
        raw_scores = raw.get("mechanism_scores", {}) if isinstance(raw.get("mechanism_scores"), dict) else {}
        raw_hints = raw.get("mechanism_hints", {}) if isinstance(raw.get("mechanism_hints"), dict) else {}
        normalized.append(
            {
                **candidate,
                "mechanism_scores": {mechanism: _score(raw_scores.get(mechanism)) for mechanism in MECHANISMS},
                "mechanism_hints": {
                    mechanism: _short_hint(raw_hints.get(mechanism), DEFAULT_HINTS[mechanism]) for mechanism in MECHANISMS
                },
                "planner_fallback": not bool(raw),
            }
        )
    return normalized


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def plan_source_pool(
    sources: list[dict[str, Any]],
    planner: Callable[[list[dict[str, str]]], dict[str, Any]],
    *,
    cache_path: Path,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Score every candidate once; cache makes a large planning pass resumable."""

    cached = {value["candidate_key"]: value for value in _load_jsonl(cache_path) if "candidate_key" in value}
    planned: list[dict[str, Any]] = []
    pending = [source for source in sources if source["candidate_key"] not in cached]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        try:
            response = planner(planning_messages(batch))
            values = normalize_plan_response(response, batch)
        except Exception as error:
            print(f"Warning: planning batch {start // batch_size + 1} failed with {type(error).__name__}; continuing with neutral scores.", flush=True)
            values = normalize_plan_response({}, batch)
        for value in values:
            _append_jsonl(cache_path, value)
            cached[value["candidate_key"]] = value
        print(f"Progress: scored multi-turn mechanism fit for {min(start + len(batch), len(pending))}/{len(pending)} cases.", flush=True)
    for source in sources:
        planned.append(cached[source["candidate_key"]])
    return planned


def allocate_balanced_mechanisms(planned: list[dict[str, Any]], *, cases_per_mechanism: int) -> list[dict[str, Any]]:
    """Apply deterministic global quotas after the LLM scores all candidates."""

    allocations: list[dict[str, Any]] = []
    for domain in DOMAINS:
        rows = [row for row in planned if row["domain"] == domain]
        remaining = {mechanism: cases_per_mechanism for mechanism in MECHANISMS}
        rows.sort(
            key=lambda row: (
                max(row["mechanism_scores"].values()) - min(row["mechanism_scores"].values()),
                max(row["mechanism_scores"].values()),
                row["candidate_key"],
            ),
            reverse=True,
        )
        for row in rows:
            if not any(remaining.values()):
                break
            ranked = sorted(MECHANISMS, key=lambda mechanism: row["mechanism_scores"][mechanism], reverse=True)
            mechanism = next((label for label in ranked if remaining[label] > 0), None)
            if mechanism is None:
                continue
            remaining[mechanism] -= 1
            best_score = max(row["mechanism_scores"].values())
            allocations.append(
                {
                    **row,
                    "assigned_mechanism": mechanism,
                    "assigned_score": row["mechanism_scores"][mechanism],
                    "best_score": best_score,
                    "assignment_regret": best_score - row["mechanism_scores"][mechanism],
                    "quota_per_mechanism": cases_per_mechanism,
                }
            )
        if any(remaining.values()):
            raise ValueError(f"Unable to meet balanced quota for {domain}: {remaining}")
    return allocations


def make_test_case_inputs(allocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn globally assigned legacy queries into compact trajectory inputs."""

    values: list[dict[str, Any]] = []
    for row in allocations:
        mechanism = row["assigned_mechanism"]
        values.append(
            {
                "case_id": f"test-{row['domain'].removeprefix('amazon_')}-{mechanism}-{row['source_item_id']}",
                "domain": row["domain"],
                "family": mechanism,
                "intent_topology": f"llm_scored_{mechanism}",
                "expected_state_path": list(STATE_PATHS[mechanism]),
                "required_decision_axis": row["mechanism_hints"][mechanism],
                "transition_rationale": (
                    f"LLM suitability score={row['assigned_score']} for {mechanism}; "
                    f"global allocator selected this mechanism under a fixed balanced quota."
                ),
                "legacy_source": {
                    "item_id": row["source_item_id"],
                    "query": row["query"],
                    "targets": row["targets"],
                    "user_id": row["user_id"],
                },
                "anchor_item": {
                    "title": row["title"],
                    "category": row["category"],
                    "description": row["description"],
                    "price": "",
                },
                "allocation": {
                    "mechanism_scores": row["mechanism_scores"],
                    "assigned_mechanism": mechanism,
                    "assigned_score": row["assigned_score"],
                    "best_score": row["best_score"],
                    "assignment_regret": row["assignment_regret"],
                    "planner_fallback": row["planner_fallback"],
                },
            }
        )
    return values


def _fallback_trajectory(case_input: dict[str, Any]) -> dict[str, Any]:
    """Keep a test epoch runnable if a transient author call exhausts retries."""

    mechanism = case_input["family"]
    evidence = case_input.get("catalog_evidence_menu", [])
    positive = evidence[0] if evidence else {"item_id": case_input["legacy_source"]["item_id"], "title": "", "category": ""}
    negative = evidence[1] if len(evidence) > 1 else positive
    direct = mechanism == "direct"
    return {
        "public_initial_query": case_input["legacy_source"]["query"],
        "private_state": {
            "visible_facts": [case_input["legacy_source"]["targets"] or "legacy complex query"],
            "latent_stable_preference": {"name": "" if direct else case_input["required_decision_axis"]},
        },
        "transition": {
            "family": mechanism,
            "agent_action_required": "" if direct else f"Clarify {case_input['required_decision_axis']}.",
            "user_reply": "" if direct else "I prioritize the clarified preference.",
            "state_update": "RecommendationReady" if direct else "PreferenceClarification",
            "resolved_priority": {"label": case_input["required_decision_axis"], "required_catalog_terms": []},
        },
        "evaluation_contract": {
            "positive_evidence": [{"item_id": positive["item_id"], "supporting_spans": [positive.get("title", "")] if positive.get("title") else []}],
            "hard_negative_evidence": [{"item_id": negative["item_id"], "missing_or_conflicting_requirement": "contrast candidate"}],
            "max_system_turns": 0 if direct else 1,
        },
        "thought_pattern": {
            "task_description": f"Handle {mechanism} recommendation state.",
            "solution_description": case_input["required_decision_axis"],
            "thought_template": ["Read state", "Choose action", "Ground recommendation"],
        },
    }


def generate_test_trajectories(
    case_inputs: list[dict[str, Any]],
    author: Callable[[list[dict[str, str]]], dict[str, Any]],
    *,
    cache_path: Path,
) -> list[dict[str, Any]]:
    """Generate compact trajectories after assignments are frozen; never gate them away."""

    cached = {value["case_input"]["case_id"]: value for value in _load_jsonl(cache_path) if value.get("case_input")}
    records: list[dict[str, Any]] = []
    for position, case_input in enumerate(case_inputs, start=1):
        case_id = case_input["case_id"]
        if case_id in cached:
            records.append(cached[case_id])
            continue
        try:
            trajectory = _canonicalize_trajectory(case_input, author(author_messages(case_input)))
            generation_source = "llm"
            generation_error = ""
        except Exception as error:
            trajectory = _fallback_trajectory(case_input)
            generation_source = "fallback"
            generation_error = f"{type(error).__name__}: {error}"
        record = {
            "version": FROZEN_TEST_VERSION,
            "case_input": case_input,
            "trajectory": trajectory,
            "generation_source": generation_source,
            "generation_error": generation_error,
            "contract_warnings": local_contract_errors(case_input, trajectory),
            "frozen": True,
        }
        _append_jsonl(cache_path, record)
        cached[case_id] = record
        records.append(record)
        print(f"Progress: generated frozen test trajectories {position}/{len(case_inputs)}.", flush=True)
    return records


def write_frozen_test_artifacts(
    output_dir: Path,
    *,
    source_pool: list[dict[str, Any]],
    allocations: list[dict[str, Any]],
    records: list[dict[str, Any]],
    seed: int,
    pool_per_domain: int,
    cases_per_mechanism: int,
) -> None:
    """Write the immutable source, allocation, and Agent-facing episode surfaces."""

    output_dir.mkdir(parents=True, exist_ok=True)
    def write_jsonl(name: str, values: Iterable[dict[str, Any]]) -> None:
        (output_dir / name).write_text(
            "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values), encoding="utf-8"
        )

    write_jsonl("source_pool.jsonl", source_pool)
    write_jsonl("allocation_manifest.jsonl", allocations)
    write_jsonl("episodes.jsonl", records)
    agent_episodes = [
        {
            "episode_id": record["case_input"]["case_id"],
            "domain": record["case_input"]["domain"],
            "mechanism": record["case_input"]["family"],
            "initial_query": record["trajectory"].get("public_initial_query", ""),
            "private_state": record["trajectory"].get("private_state", {}),
            "transition": record["trajectory"].get("transition", {}),
            "catalog_evidence_menu": record["case_input"].get("catalog_evidence_menu", []),
            "evaluation_contract": record["trajectory"].get("evaluation_contract", {}),
            "thought_pattern": record["trajectory"].get("thought_pattern", {}),
        }
        for record in records
    ]
    write_jsonl("agent_test_episodes.jsonl", agent_episodes)
    mechanism_counts = {
        domain: {mechanism: sum(record["case_input"]["domain"] == domain and record["case_input"]["family"] == mechanism for record in records) for mechanism in MECHANISMS}
        for domain in DOMAINS
    }
    summary = {
        "version": FROZEN_TEST_VERSION,
        "seed": seed,
        "source_pool_per_domain": pool_per_domain,
        "cases_per_domain_mechanism": cases_per_mechanism,
        "case_count": len(records),
        "generation_sources": {kind: sum(record["generation_source"] == kind for record in records) for kind in ("llm", "fallback")},
        "contract_warning_count": sum(bool(record["contract_warnings"]) for record in records),
        "mechanism_counts": mechanism_counts,
        "development_source_disjoint": True,
        "runtime_generation_required": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Frozen TAIRAS Agent Test Benchmark\n\n"
        "This artifact is frozen for Agent evaluation. `agent_test_episodes.jsonl` is consumed by "
        "`taira_mt.multiturn_benchmark.runtime`: the Agent receives only the public query and a full-domain catalog tool, "
        "while the simulator/evaluator owns the stored private state, transition, evidence, and Thought Pattern. "
        "Run `taira-mt-materialize-frozen-runtime` to write the deterministic evaluator-side dialogue protocol. "
        "No generation LLM is called during a benchmark rollout.\n",
        encoding="utf-8",
    )
