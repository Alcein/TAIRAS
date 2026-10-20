"""Construct DSUS supervision from the frozen evolving-user environment."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from taira_mt.artifacts import read_jsonl_unique, verify_artifact_bundle
from taira_mt.benchmark import FrozenEvolvingUserBenchmark
from taira_mt.benchmark.diagnostics import (
    DiagnosticPolicyName,
    PublicDialogueContext,
    diagnostic_public_action,
    private_oracle_diagnostic_action,
)
from taira_mt.benchmark.schemas import (
    BenchmarkActionType,
    BenchmarkEpisodeTrace,
    BenchmarkSystemAction,
    CandidateFeasibility,
    IntentCompleteness,
    RuntimeUserState,
    TransitionTrigger,
)
from taira_mt.constraints.predicates import predicate_evidence_status
from taira_mt.constraints.rendering import render_constraint
from taira_mt.dsus.schemas import (
    DecisionObligation,
    DSUSArtifactFile,
    DSUSBehaviorPolicy,
    DSUSDatasetConfig,
    DSUSDevelopmentManifest,
    DSUSDevelopmentSplit,
    DSUSDevelopmentSummary,
    DSUSSplitRecord,
    DSUSStateFeature,
    DSUSStateLabel,
    PatienceRisk,
    PublicConstraintObservation,
    SatisfactionBand,
)
from taira_mt.models import ConstraintEvidenceStatus, ConstraintImportance
from taira_mt.provenance import canonical_json, content_sha256, source_tree_sha256


@dataclass(frozen=True)
class _CaseSplitDescriptor:
    case_id: str
    split_group_id: str
    domain: str
    scenario_family: str

    @property
    def stratum(self) -> tuple[str, str]:
        return self.domain, self.scenario_family


@dataclass(frozen=True)
class DSUSDevelopmentBuildResult:
    build_id: str
    config: DSUSDatasetConfig
    freeze_id: str
    parent_benchmark_build_id: str
    parent_benchmark_manifest_sha256: str
    parent_language_bank_id: str
    parent_language_bank_manifest_sha256: str
    language_renderer_id: str
    features: tuple[DSUSStateFeature, ...]
    labels: tuple[DSUSStateLabel, ...]
    episodes: tuple[BenchmarkEpisodeTrace, ...]
    split_registry: tuple[DSUSSplitRecord, ...]
    summary: DSUSDevelopmentSummary


@dataclass(frozen=True)
class LoadedDSUSDevelopmentData:
    manifest_sha256: str
    manifest: DSUSDevelopmentManifest
    features: tuple[DSUSStateFeature, ...]
    labels: tuple[DSUSStateLabel, ...]
    episodes: tuple[BenchmarkEpisodeTrace, ...]
    split_registry: tuple[DSUSSplitRecord, ...]
    summary: DSUSDevelopmentSummary


def load_dsus_development_artifact(
    run_dir: Path,
) -> LoadedDSUSDevelopmentData:
    """Load DSUS data only after manifest, file, and join verification."""

    bundle = verify_artifact_bundle(run_dir)
    manifest = DSUSDevelopmentManifest.model_validate(bundle.manifest)
    features = tuple(
        read_jsonl_unique(
            bundle.require_output("features.public.jsonl"),
            DSUSStateFeature.model_validate,
            lambda item: item.state_id,
            record_name="DSUS public feature",
        )
    )
    labels = tuple(
        read_jsonl_unique(
            bundle.require_output("labels.private.jsonl"),
            DSUSStateLabel.model_validate,
            lambda item: item.state_id,
            record_name="DSUS private label",
        )
    )
    episodes = tuple(
        read_jsonl_unique(
            bundle.require_output("episodes.private.jsonl"),
            BenchmarkEpisodeTrace.model_validate,
            lambda item: item.episode_id,
            record_name="DSUS private episode",
        )
    )
    split_registry = tuple(
        read_jsonl_unique(
            bundle.require_output("split_registry.public.jsonl"),
            DSUSSplitRecord.model_validate,
            lambda item: item.split_group_id,
            record_name="DSUS split group",
        )
    )
    summary = DSUSDevelopmentSummary.model_validate(
        json.loads(bundle.require_output("summary.json").read_text(encoding="utf-8"))
    )
    feature_payload = [item.model_dump(mode="json") for item in features]
    label_payload = [item.model_dump(mode="json") for item in labels]
    episode_payload = [item.model_dump(mode="json") for item in episodes]
    split_payload = [item.model_dump(mode="json") for item in split_registry]
    checks = {
        "features": (content_sha256(feature_payload), manifest.features_sha256),
        "labels": (content_sha256(label_payload), manifest.labels_sha256),
        "episodes": (content_sha256(episode_payload), manifest.episodes_sha256),
        "split registry": (
            content_sha256(split_payload),
            manifest.split_registry_sha256,
        ),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"DSUS {name} content hash differs from manifest")
    if summary.build_id != manifest.build_id:
        raise ValueError("DSUS summary and manifest build IDs differ")
    if {item.state_id for item in features} != {item.state_id for item in labels}:
        raise ValueError("DSUS loaded public/private state IDs differ")
    if len(features) != summary.state_count or len(episodes) != summary.episode_count:
        raise ValueError("DSUS loaded record counts differ from summary")
    return LoadedDSUSDevelopmentData(
        manifest_sha256=bundle.manifest_sha256,
        manifest=manifest,
        features=features,
        labels=labels,
        episodes=episodes,
        split_registry=split_registry,
        summary=summary,
    )


def _load_freeze_metadata(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("user-simulator freeze config must contain a mapping")
    for key in ("freeze_id", "benchmark", "language_bank", "runtime"):
        if key not in payload:
            raise ValueError(f"user-simulator freeze config is missing {key}")
    if payload["runtime"].get("external_api_calls") is not False:
        raise ValueError("DSUS data construction requires a zero-API simulator runtime")
    return payload


def _assign_splits(
    descriptors: tuple[_CaseSplitDescriptor, ...],
    *,
    dev_cases_per_stratum: int,
    split_seed: str,
) -> tuple[dict[str, DSUSDevelopmentSplit], tuple[DSUSSplitRecord, ...]]:
    grouped: dict[str, list[_CaseSplitDescriptor]] = defaultdict(list)
    for item in descriptors:
        grouped[item.split_group_id].append(item)
    strata = sorted({item.stratum for item in descriptors})
    stratum_totals = Counter(item.stratum for item in descriptors)
    if any(stratum_totals[stratum] < 2 * dev_cases_per_stratum for stratum in strata):
        raise ValueError("DSUS split target leaves too few training cases in a stratum")
    vectors = {
        group_id: Counter(item.stratum for item in values)
        for group_id, values in grouped.items()
    }
    target = {stratum: dev_cases_per_stratum for stratum in strata}

    def objective(counts: Counter[tuple[str, str]]) -> int:
        return sum((counts[stratum] - target[stratum]) ** 2 for stratum in strata)

    selected: set[str] = set()
    realized: Counter[tuple[str, str]] = Counter()
    while objective(realized) > 0:
        candidates = []
        for group_id, vector in vectors.items():
            if group_id in selected:
                continue
            updated = realized + vector
            delta = objective(updated) - objective(realized)
            tie_break = hashlib.sha256(
                f"{split_seed}|{group_id}".encode()
            ).hexdigest()
            candidates.append((delta, tie_break, group_id, updated))
        if not candidates:
            raise ValueError("DSUS split selection exhausted all relation groups")
        delta, _, group_id, updated = min(candidates)
        if delta >= 0:
            raise ValueError("cannot satisfy exact group-disjoint DSUS stratum quotas")
        selected.add(group_id)
        realized = updated
    if any(realized[stratum] != target[stratum] for stratum in strata):
        raise AssertionError("DSUS split does not satisfy its exact stratum target")

    assignment = {
        group_id: (
            DSUSDevelopmentSplit.DEV
            if group_id in selected
            else DSUSDevelopmentSplit.TRAIN
        )
        for group_id in grouped
    }
    records = tuple(
        DSUSSplitRecord(
            split_group_id=group_id,
            split=assignment[group_id],
            case_ids=sorted(item.case_id for item in values),
            domain_scenario_counts=dict(
                sorted(
                    Counter(
                        f"{item.domain}:{item.scenario_family}" for item in values
                    ).items()
                )
            ),
        )
        for group_id, values in sorted(grouped.items())
    )
    return assignment, records


def _public_constraints(environment) -> list[PublicConstraintObservation]:
    active_goal = environment.case.private.initial_state.latent_goal.version(
        environment.runtime.goal_version
    )
    disclosed = set(environment.runtime.disclosed_constraint_ids)
    return [
        PublicConstraintObservation(
            slot=constraint.slot,
            predicate=constraint.predicate,
            surface_form=constraint.surface_form,
        )
        for constraint in active_goal.constraints
        if constraint.constraint_id in disclosed
    ]


def _public_catalog_counts(environment) -> tuple[int, int]:
    observations = _public_constraints(environment)
    hard = [
        item for item in observations if item.predicate.importance == ConstraintImportance.HARD
    ]
    verified = 0
    unknown = 0
    rejected = set(environment.runtime.rejected_item_ids)
    for item in environment.candidate_pool.remaining_items(rejected):
        statuses = [
            predicate_evidence_status(item.attributes.get(value.slot), value.predicate)
            for value in hard
        ]
        if ConstraintEvidenceStatus.VIOLATED in statuses:
            continue
        if ConstraintEvidenceStatus.UNKNOWN in statuses:
            unknown += 1
        else:
            verified += 1
    return verified, unknown


def _field_surface(field: str, value: object) -> str:
    label = field.replace("_", " ")
    if field == "price" and isinstance(value, (int, float)):
        return f"price ${float(value):.2f}"
    if field == "rating" and isinstance(value, (int, float)):
        return f"rating {float(value):.1f} stars"
    if field == "popularity_percentile" and isinstance(value, (int, float)):
        return f"popularity score {float(value):.2f}"
    return f"{label} {value}"


def _system_action_text(environment, action: BenchmarkSystemAction) -> str:
    if action.action_type == BenchmarkActionType.ASK:
        assert action.question_target is not None
        return f"System asks about the user's {action.question_target.replace('_', ' ')}."
    if action.action_type == BenchmarkActionType.RECOMMEND:
        titles = [environment.candidate_pool.index.get(item_id).title for item_id in action.item_ids]
        return "System recommends: " + "; ".join(titles) + "."
    if action.action_type == BenchmarkActionType.EXPLAIN:
        item_ids = action.item_ids or sorted({item.item_id for item in action.evidence_refs})
        parts = []
        for item_id in item_ids:
            item = environment.candidate_pool.index.get(item_id)
            fields = [
                _field_surface(ref.catalog_field, item.attributes.get(ref.catalog_field))
                for ref in action.evidence_refs
                if ref.item_id == item_id
            ]
            parts.append(f"{item.title}: " + ", ".join(fields))
        return "System explains the recommendation using catalog evidence: " + "; ".join(parts) + "."
    if action.action_type == BenchmarkActionType.PROPOSE_RELAX:
        assert action.relaxation_proposal is not None
        proposal = action.relaxation_proposal
        matches = [
            constraint
            for goal in environment.case.private.initial_state.latent_goal.versions
            for constraint in goal.constraints
            if constraint.constraint_id == proposal.constraint_id
            and constraint.predicate == proposal.proposed_predicate
        ]
        if len(matches) != 1:
            raise ValueError("cannot uniquely render the public relaxation proposal")
        surface = render_constraint(matches[0].slot, proposal.proposed_predicate)
        return f"System proposes this trade-off: {surface}"
    return "System ends the recommendation attempt."


def _patience_risk(state: RuntimeUserState, budget: int) -> tuple[float, PatienceRisk]:
    ratio = state.patience_remaining / budget
    if ratio <= 1 / 3:
        return ratio, PatienceRisk.HIGH
    if ratio <= 2 / 3:
        return ratio, PatienceRisk.MEDIUM
    return ratio, PatienceRisk.LOW


def _satisfaction_band(value: float) -> SatisfactionBand:
    if value < 0.4:
        return SatisfactionBand.LOW
    if value < 0.7:
        return SatisfactionBand.MEDIUM
    return SatisfactionBand.HIGH


def _decision_obligation(
    state: RuntimeUserState,
    optimal_action: BenchmarkSystemAction,
) -> DecisionObligation:
    if optimal_action.action_type == BenchmarkActionType.ASK:
        if state.feasibility == CandidateFeasibility.EMPTY:
            return DecisionObligation.NEGOTIATE_TRADEOFF
        return (
            DecisionObligation.RESOLVE_AMBIGUITY
            if state.completeness == IntentCompleteness.AMBIGUOUS
            else DecisionObligation.ELICIT_PREFERENCE
        )
    return {
        BenchmarkActionType.RECOMMEND: DecisionObligation.RECOMMEND,
        BenchmarkActionType.EXPLAIN: DecisionObligation.GROUND_EXPLANATION,
        BenchmarkActionType.PROPOSE_RELAX: DecisionObligation.NEGOTIATE_TRADEOFF,
        BenchmarkActionType.STOP: DecisionObligation.TERMINATE,
    }[optimal_action.action_type]


def _public_information_corrective_action(environment) -> BenchmarkSystemAction:
    """Return an expert action whose payload is derivable from public history.

    The evaluator-private oracle may directly submit the registered relaxation
    predicate.  A real Agent has not observed that boundary yet, so an empty-set
    state is corrected by asking which disclosed numeric condition can move.
    The simulator then reveals the precommitted boundary in public language.
    """

    state = environment.runtime
    if state.feasibility != CandidateFeasibility.EMPTY:
        return private_oracle_diagnostic_action(environment)
    private = environment.case.private.initial_state
    events = [
        event
        for event in private.evolution_plan.events
        if event.trigger == TransitionTrigger.ASK_CONSTRAINT
        and event.from_goal_version == state.goal_version
        and event.event_id not in state.fired_event_ids
        and event.trigger_ref is not None
    ]
    if len(events) != 1:
        return BenchmarkSystemAction(
            action_type=BenchmarkActionType.STOP,
            utterance="No public, authorized trade-off question remains.",
        )
    event = events[0]
    goal = private.latent_goal.version(state.goal_version)
    constraint = next(
        item for item in goal.constraints if item.constraint_id == event.trigger_ref
    )
    if constraint.constraint_id not in state.disclosed_constraint_ids:
        raise ValueError("trade-off expert cannot ask about an undisclosed constraint")
    return BenchmarkSystemAction(
        action_type=BenchmarkActionType.ASK,
        question_target=constraint.slot,
        utterance=(
            f"Would you be flexible about your {constraint.slot.replace('_', ' ')} "
            "requirement?"
        ),
    )


def _state_archetype_id(
    state: RuntimeUserState,
    *,
    patience_risk: PatienceRisk,
    satisfaction_band: SatisfactionBand,
) -> str:
    return content_sha256(
        {
            "phase": state.phase.value,
            "completeness": state.completeness.value,
            "stability": state.stability.value,
            "feasibility": state.feasibility.value,
            "last_user_act": state.last_user_act.value if state.last_user_act else None,
            "explanation_pending": state.pending_acceptable_item_id is not None,
            "patience_risk": patience_risk.value,
            "satisfaction_band": satisfaction_band.value,
        }
    )


def build_dsus_development_dataset(
    freeze_config_path: Path,
    *,
    config: DSUSDatasetConfig | None = None,
) -> DSUSDevelopmentBuildResult:
    config = config or DSUSDatasetConfig()
    freeze_config_path = freeze_config_path.resolve()
    freeze = _load_freeze_metadata(freeze_config_path)
    suite = FrozenEvolvingUserBenchmark.from_config(freeze_config_path)

    descriptors = []
    for case_id in suite.case_ids:
        public = suite.public_case(case_id)
        if public.split_group_id is None:
            raise ValueError("DSUS development splitting requires split_group_id")
        environment = suite.open_session(case_id)
        descriptors.append(
            _CaseSplitDescriptor(
                case_id=case_id,
                split_group_id=public.split_group_id,
                domain=public.domain,
                scenario_family=(
                    environment.case.private.initial_state.scenario_family.value
                ),
            )
        )
    split_by_group, split_registry = _assign_splits(
        tuple(descriptors),
        dev_cases_per_stratum=config.dev_cases_per_domain_scenario,
        split_seed=config.split_seed,
    )

    features: list[DSUSStateFeature] = []
    labels: list[DSUSStateLabel] = []
    episodes: list[BenchmarkEpisodeTrace] = []
    for case_id in suite.case_ids:
        for behavior_policy in config.behavior_policies:
            environment = suite.open_session(case_id)
            initial_query = environment.reset()
            public = environment.case.public
            assert public.split_group_id is not None
            split = split_by_group[public.split_group_id]
            episode_id = f"dsus-{behavior_policy.value}-{case_id}"
            dialogue_parts = [f"User: {initial_query}"]
            semantic_history: list[dict[str, Any]] = []
            last_action_type = None
            context = PublicDialogueContext(case=public)
            while (
                not environment.terminal
                and len(context.actions) < config.maximum_behavior_turns
            ):
                state = environment.runtime.model_copy(deep=True)
                optimal_action = _public_information_corrective_action(environment)
                semantic_state_id = content_sha256(
                    {
                        "case_id": case_id,
                        "runtime": state.model_dump(mode="json"),
                        "semantic_history": semantic_history,
                    }
                )
                state_id = content_sha256(
                    {
                        "semantic_state_id": semantic_state_id,
                        "language_renderer_id": suite.renderer_id,
                        "episode_id": episode_id,
                    }
                )
                verified_count, unknown_count = _public_catalog_counts(environment)
                patience_ratio, patience_risk = _patience_risk(
                    state,
                    environment.case.private.initial_state.behavior.patience_budget,
                )
                satisfaction_band = _satisfaction_band(state.satisfaction)
                features.append(
                    DSUSStateFeature(
                        state_id=state_id,
                        semantic_state_id=semantic_state_id,
                        episode_id=episode_id,
                        behavior_policy=behavior_policy,
                        case_id=case_id,
                        split_group_id=public.split_group_id,
                        split=split,
                        domain=public.domain,
                        turn_index=state.turn_index,
                        language_renderer_id=suite.renderer_id,
                        serialized_dialogue="\n".join(dialogue_parts),
                        public_constraints=_public_constraints(environment),
                        public_verified_candidate_count=verified_count,
                        public_unknown_candidate_count=unknown_count,
                        asked_question_targets=list(state.asked_question_targets),
                        rejected_item_count=len(state.rejected_item_ids),
                        last_system_action_type=last_action_type,
                    )
                )
                labels.append(
                    DSUSStateLabel(
                        state_id=state_id,
                        semantic_state_id=semantic_state_id,
                        case_id=case_id,
                        behavior_policy=behavior_policy,
                        split=split,
                        scenario_family=(
                            environment.case.private.initial_state.scenario_family
                        ),
                        state_archetype_id=_state_archetype_id(
                            state,
                            patience_risk=patience_risk,
                            satisfaction_band=satisfaction_band,
                        ),
                        intent_completeness=state.completeness,
                        intent_stability=state.stability,
                        candidate_feasibility=state.feasibility,
                        dialogue_phase=state.phase,
                        last_user_act=state.last_user_act,
                        explanation_pending=(
                            state.pending_acceptable_item_id is not None
                        ),
                        patience_remaining_ratio=patience_ratio,
                        patience_risk=patience_risk,
                        satisfaction=state.satisfaction,
                        satisfaction_band=satisfaction_band,
                        decision_obligation=_decision_obligation(
                            state, optimal_action
                        ),
                        optimal_action_type=optimal_action.action_type,
                        optimal_question_target=optimal_action.question_target,
                    )
                )
                if behavior_policy == DSUSBehaviorPolicy.CORRECTIVE_EXPERT:
                    behavior_action = optimal_action
                else:
                    behavior_action = diagnostic_public_action(
                        DiagnosticPolicyName(behavior_policy.value),
                        context,
                        environment.candidate_pool,
                        root_seed=config.policy_root_seed,
                    )
                system_text = _system_action_text(environment, behavior_action)
                response = environment.step(behavior_action)
                trace = environment.private_traces[-1]
                dialogue_parts.extend(
                    [f"System: {system_text}", f"User: {response.utterance}"]
                )
                semantic_history.append(
                    {
                        "system_action": behavior_action.model_dump(mode="json"),
                        "response_frame": trace.response_frame.model_dump(
                            mode="json", exclude={"surface_text"}
                        ),
                    }
                )
                context.actions.append(behavior_action)
                context.user_utterances.append(response.utterance)
                last_action_type = behavior_action.action_type
            if not environment.terminal:
                environment.step(
                    BenchmarkSystemAction(
                        action_type=BenchmarkActionType.STOP,
                        utterance="The fixed behavior-policy horizon has been reached.",
                    )
                )
            episodes.append(environment.episode_trace(episode_id))

    if len({item.state_id for item in features}) != len(features):
        raise ValueError("DSUS public features contain duplicate state IDs")
    if {item.state_id for item in features} != {item.state_id for item in labels}:
        raise ValueError("DSUS public/private state IDs differ")
    policy_by_episode = {
        f"dsus-{policy.value}-{case_id}": policy
        for case_id in suite.case_ids
        for policy in config.behavior_policies
    }
    expert_episodes = [
        item
        for item in episodes
        if policy_by_episode[item.episode_id] == DSUSBehaviorPolicy.CORRECTIVE_EXPERT
    ]
    if not all(item.success for item in expert_episodes):
        raise ValueError("DSUS supervision contains an unsuccessful corrective-expert episode")
    split_groups = defaultdict(set)
    for item in features:
        split_groups[item.split].add(item.split_group_id)
    group_overlap = len(
        split_groups[DSUSDevelopmentSplit.TRAIN]
        & split_groups[DSUSDevelopmentSplit.DEV]
    )
    if group_overlap:
        raise ValueError("DSUS train/dev relation groups overlap")

    feature_payload = [item.model_dump(mode="json") for item in features]
    label_payload = [item.model_dump(mode="json") for item in labels]
    episode_payload = [item.model_dump(mode="json") for item in episodes]
    split_payload = [item.model_dump(mode="json") for item in split_registry]
    identity = {
        "schema_version": config.builder_version,
        "freeze_id": freeze["freeze_id"],
        "parent_benchmark_build_id": suite.benchmark_build_id,
        "parent_language_bank_id": suite.language_bank_id,
        "language_renderer_id": suite.renderer_id,
        "config": config.model_dump(mode="json"),
        "features_sha256": content_sha256(feature_payload),
        "labels_sha256": content_sha256(label_payload),
        "episodes_sha256": content_sha256(episode_payload),
        "split_registry_sha256": content_sha256(split_payload),
    }
    build_id = f"dsus-development-{content_sha256(identity)[:16]}"
    split_by_case = {
        item.case_id: split_by_group[item.split_group_id] for item in descriptors
    }
    summary = DSUSDevelopmentSummary(
        build_id=build_id,
        parent_benchmark_build_id=suite.benchmark_build_id,
        parent_language_bank_id=suite.language_bank_id,
        case_count=len(suite.case_ids),
        episode_count=len(episodes),
        state_count=len(features),
        unique_semantic_state_count=len(
            {item.semantic_state_id for item in features}
        ),
        semantic_duplicate_count=(
            len(features) - len({item.semantic_state_id for item in features})
        ),
        state_archetype_count=len({item.state_archetype_id for item in labels}),
        split_case_counts=dict(
            sorted(Counter(value.value for value in split_by_case.values()).items())
        ),
        split_state_counts=dict(
            sorted(Counter(item.split.value for item in features).items())
        ),
        split_group_counts=dict(
            sorted(Counter(item.split.value for item in split_registry).items())
        ),
        state_domain_scenario_split_counts=dict(
            sorted(
                Counter(
                    f"{item.split.value}:{item.domain}:{label.scenario_family.value}"
                    for item, label in zip(features, labels, strict=True)
                ).items()
            )
        ),
        behavior_policy_episode_counts=dict(
            sorted(Counter(value.value for value in policy_by_episode.values()).items())
        ),
        behavior_policy_state_counts=dict(
            sorted(Counter(item.behavior_policy.value for item in features).items())
        ),
        behavior_policy_success_rates={
            policy.value: sum(
                item.success
                for item in episodes
                if policy_by_episode[item.episode_id] == policy
            )
            / sum(
                policy_by_episode[item.episode_id] == policy for item in episodes
            )
            for policy in config.behavior_policies
        },
        optimal_action_counts=dict(
            sorted(Counter(item.optimal_action_type.value for item in labels).items())
        ),
        decision_obligation_counts=dict(
            sorted(Counter(item.decision_obligation.value for item in labels).items())
        ),
        dialogue_phase_counts=dict(
            sorted(Counter(item.dialogue_phase.value for item in labels).items())
        ),
        intent_stability_counts=dict(
            sorted(Counter(item.intent_stability.value for item in labels).items())
        ),
        candidate_feasibility_counts=dict(
            sorted(Counter(item.candidate_feasibility.value for item in labels).items())
        ),
        group_overlap_count=group_overlap,
        all_corrective_expert_episodes_successful=True,
        feature_label_join_exact=True,
    )
    return DSUSDevelopmentBuildResult(
        build_id=build_id,
        config=config,
        freeze_id=freeze["freeze_id"],
        parent_benchmark_build_id=suite.benchmark_build_id,
        parent_benchmark_manifest_sha256=freeze["benchmark"]["manifest_sha256"],
        parent_language_bank_id=suite.language_bank_id,
        parent_language_bank_manifest_sha256=freeze["language_bank"][
            "manifest_sha256"
        ],
        language_renderer_id=suite.renderer_id,
        features=tuple(features),
        labels=tuple(labels),
        episodes=tuple(episodes),
        split_registry=split_registry,
        summary=summary,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_idempotently(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different DSUS file: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(canonical_json(item) + "\n" for item in records)


def write_dsus_development_artifact(
    result: DSUSDevelopmentBuildResult,
    output_root: Path,
    *,
    project_root: Path,
) -> Path:
    run_dir = output_root / result.build_id
    run_dir.mkdir(parents=True, exist_ok=True)
    features = [item.model_dump(mode="json") for item in result.features]
    labels = [item.model_dump(mode="json") for item in result.labels]
    episodes = [item.model_dump(mode="json") for item in result.episodes]
    split_registry = [item.model_dump(mode="json") for item in result.split_registry]
    outputs = {
        "features.public.jsonl": _jsonl(features),
        "labels.private.jsonl": _jsonl(labels),
        "episodes.private.jsonl": _jsonl(episodes),
        "split_registry.public.jsonl": _jsonl(split_registry),
        "summary.json": json.dumps(
            result.summary.model_dump(mode="json"), ensure_ascii=False, indent=2
        )
        + "\n",
    }
    for filename, text in outputs.items():
        _write_idempotently(run_dir / filename, text)
    files = [
        DSUSArtifactFile(
            relative_path=filename,
            sha256=_file_sha256(run_dir / filename),
            byte_size=(run_dir / filename).stat().st_size,
            line_count=text.count("\n") if filename.endswith(".jsonl") else None,
        )
        for filename, text in sorted(outputs.items())
    ]
    manifest = DSUSDevelopmentManifest(
        build_id=result.build_id,
        config=result.config,
        freeze_id=result.freeze_id,
        parent_benchmark_build_id=result.parent_benchmark_build_id,
        parent_benchmark_manifest_sha256=result.parent_benchmark_manifest_sha256,
        parent_language_bank_id=result.parent_language_bank_id,
        parent_language_bank_manifest_sha256=(
            result.parent_language_bank_manifest_sha256
        ),
        language_renderer_id=result.language_renderer_id,
        features_sha256=content_sha256(features),
        labels_sha256=content_sha256(labels),
        episodes_sha256=content_sha256(episodes),
        split_registry_sha256=content_sha256(split_registry),
        source_tree_sha256=source_tree_sha256(project_root),
        outputs=files,
    )
    _write_idempotently(
        run_dir / "manifest.json",
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
    )
    return run_dir
