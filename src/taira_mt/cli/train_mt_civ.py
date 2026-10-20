"""Train the paper-canonical public pairwise CIV ranker on development forks."""

from __future__ import annotations

import argparse
import gc
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from taira_mt.benchmark.schemas import BenchmarkActionType
from taira_mt.mt_agent.civ import (
    CIV_PUBLIC_FEATURE_NAMES,
    CIVPairwiseLinearRanker,
    civ_public_feature_vector,
)
from taira_mt.mt_agent.memory import PublicDialogueMemory
from taira_mt.mt_agent.retrieval import LexicalCatalogRetriever
from taira_mt.mt_agent.sctp import SCTPActionFrontier, SCTPSkillCompiler, SkillActionProposal
from taira_mt.mt_agent.skills import ThoughtPatternLibrary
from taira_mt.mt_agent.state import (
    DialoguePhase,
    DisclosureStatus,
    DSUSDecisionPosterior,
    RecommendationReadiness,
    RuleDSUSEstimator,
)
from taira_mt.provenance import canonical_json

_ASK_FAMILIES = {"progressive", "ambiguous", "conflict"}
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train source-disjoint pairwise CIV on development forks.")
    parser.add_argument(
        "--development-trajectories",
        type=Path,
        default=Path("data/benchmark/development_trajectories.jsonl"),
    )
    parser.add_argument(
        "--test-allocation-manifest",
        type=Path,
        default=Path("data/benchmark/allocation_manifest.jsonl"),
    )
    parser.add_argument("--retriever-root", type=Path, default=Path("artifacts/mt_catalog_indexes_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/mt_civ_models_v1"))
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _split(records: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    by_family: dict[str, list[str]] = defaultdict(list)
    for record in records:
        case = record["case_input"]
        by_family[str(case["family"])].append(str(case["case_id"]))
    train: set[str] = set()
    dev: set[str] = set()
    for family, case_ids in sorted(by_family.items()):
        ordered = sorted(case_ids)
        if len(ordered) < 3:
            raise ValueError(f"CIV needs at least three development cases for {family}")
        boundary = max(1, len(ordered) - 2)
        train.update(ordered[:boundary])
        dev.update(ordered[boundary:])
    return train, dev


def _posterior(family: str) -> DSUSDecisionPosterior:
    if family in _ASK_FAMILIES:
        return DSUSDecisionPosterior(
            recommend_now=0.12,
            ask_decisive=0.82,
            probe_recommend=0.04,
            repair_recommend=0.01,
            safe_stop=0.01,
        )
    if family == "revision":
        return DSUSDecisionPosterior(
            recommend_now=0.04,
            ask_decisive=0.12,
            probe_recommend=0.82,
            repair_recommend=0.01,
            safe_stop=0.01,
        )
    return DSUSDecisionPosterior(
        recommend_now=0.82,
        ask_decisive=0.12,
        probe_recommend=0.04,
        repair_recommend=0.01,
        safe_stop=0.01,
    )


def _gold_state(
    *,
    record: dict[str, Any],
    memory: PublicDialogueMemory,
    retriever: LexicalCatalogRetriever,
):
    case = record["case_input"]
    family = str(case["family"])
    axis = str(case.get("required_decision_axis") or "the deciding preference")
    base = RuleDSUSEstimator().infer(memory=memory, retriever=retriever).state
    if family in _ASK_FAMILIES:
        disclosure = {
            "progressive": DisclosureStatus.MISSING_DECISIVE,
            "ambiguous": DisclosureStatus.AMBIGUOUS,
            "conflict": DisclosureStatus.CONFLICT,
        }[family]
        return base.model_copy(
            update={
                "dialogue_phase": DialoguePhase.ELICITATION,
                "disclosure_status": disclosure,
                "unresolved_axes": [axis],
                "recommendation_readiness": RecommendationReadiness.ASK_DECISIVE,
                "confidence": 0.82,
                "public_evidence_summary": "Development-only state label for counterfactual CIV training.",
            },
            deep=True,
        )
    readiness = (
        RecommendationReadiness.PROBE_RECOMMEND if family == "revision" else RecommendationReadiness.RECOMMEND_NOW
    )
    return base.model_copy(
        update={
            "dialogue_phase": DialoguePhase.RECOMMENDATION,
            "disclosure_status": DisclosureStatus.SUFFICIENT,
            "unresolved_axes": [],
            "recommendation_readiness": readiness,
            "confidence": 0.82,
            "public_evidence_summary": "Development-only state label for counterfactual CIV training.",
        },
        deep=True,
    )


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.casefold()))


def _proposal_reward(record: dict[str, Any], proposal: SkillActionProposal) -> float:
    case = record["case_input"]
    family = str(case["family"])
    expected_action = BenchmarkActionType.ASK if family in _ASK_FAMILIES else BenchmarkActionType.RECOMMEND
    action_type = proposal.action.action_type
    reward = 1.5 if action_type == expected_action else -1.0
    trajectory = record["trajectory"]
    contract = trajectory.get("evaluation_contract", {})
    positives = {
        str(item.get("item_id"))
        for item in contract.get("positive_evidence", [])
        if isinstance(item, dict) and item.get("item_id")
    }
    negatives = {
        str(item.get("item_id"))
        for item in contract.get("hard_negative_evidence", [])
        if isinstance(item, dict) and item.get("item_id")
    }
    if action_type == BenchmarkActionType.RECOMMEND:
        recommended = set(proposal.action.item_ids)
        reward += 2.0 if recommended.intersection(positives) else -0.5
        reward -= 0.5 * float(bool(recommended.intersection(negatives)))
    elif action_type == BenchmarkActionType.ASK:
        axis_tokens = _tokens(str(case.get("required_decision_axis", "")))
        question_tokens = _tokens(str(proposal.action.question_target or ""))
        overlap = len(axis_tokens.intersection(question_tokens)) / max(1, len(axis_tokens))
        reward += 0.6 * overlap if family in _ASK_FAMILIES else -0.4
    elif action_type == BenchmarkActionType.STOP:
        reward -= 2.0
    return reward


def _decision_row(
    *,
    record: dict[str, Any],
    frontier: SCTPActionFrontier,
) -> dict[str, Any]:
    rewards = {proposal.proposal_id: _proposal_reward(record, proposal) for proposal in frontier.proposals}
    best_reward = max(rewards.values())
    best_ids = sorted(proposal_id for proposal_id, reward in rewards.items() if reward == best_reward)
    case = record["case_input"]
    return {
        "case_id": str(case["case_id"]),
        "family": str(case["family"]),
        "domain": str(case["domain"]),
        "frontier": frontier,
        "rewards": rewards,
        "best_reward": best_reward,
        "best_proposal_ids": best_ids,
    }


def _pairwise_rows(decisions: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    differences: list[np.ndarray] = []
    labels: list[int] = []
    for decision in decisions:
        frontier: SCTPActionFrontier = decision["frontier"]
        by_id = {proposal.proposal_id: proposal for proposal in frontier.proposals}
        winner_id = decision["best_proposal_ids"][0]
        winner = civ_public_feature_vector(by_id[winner_id])
        for proposal_id, reward in decision["rewards"].items():
            if reward >= decision["best_reward"] or proposal_id == winner_id:
                continue
            loser = civ_public_feature_vector(by_id[proposal_id])
            difference = winner - loser
            differences.extend((difference, -difference))
            labels.extend((1, 0))
    if not differences:
        raise ValueError("CIV development decisions produced no strict pairwise preferences")
    return np.vstack(differences), np.asarray(labels, dtype=np.int64)


def _evaluate(model: CIVPairwiseLinearRanker, decisions: list[dict[str, Any]]) -> dict[str, float]:
    optimal: list[float] = []
    action_correct: list[float] = []
    for decision in decisions:
        frontier: SCTPActionFrontier = decision["frontier"]
        selection = model.select(frontier)
        selected = model.selected_proposal(frontier, selection)
        optimal.append(float(selection.selected_proposal_id in decision["best_proposal_ids"]))
        expected = BenchmarkActionType.ASK if decision["family"] in _ASK_FAMILIES else BenchmarkActionType.RECOMMEND
        action_correct.append(float(selected.action.action_type == expected))
    return {
        "optimal_proposal_rate": mean(optimal),
        "initial_action_accuracy": mean(action_correct),
    }


def main() -> None:
    args = parse_args()
    print("步骤 1/5：加载 30 条开发轨迹并核验与冻结测试 source-disjoint。", flush=True)
    records = _read_jsonl(args.development_trajectories)
    test_keys = {str(row["candidate_key"]) for row in _read_jsonl(args.test_allocation_manifest)}
    development_keys = {
        f"{record['case_input']['domain']}:{record['case_input']['legacy_source']['item_id']}" for record in records
    }
    overlap = development_keys.intersection(test_keys)
    if overlap:
        raise ValueError(f"CIV development sources overlap frozen test: {sorted(overlap)[:3]}")
    train_ids, dev_ids = _split(records)
    print(
        f"步骤 1/5 完成：train cases={len(train_ids)}，dev cases={len(dev_ids)}，source overlap=0。",
        flush=True,
    )

    print("步骤 2/5：用开发期 gold state 枚举 SCTP 合法反事实行动。", flush=True)
    library = ThoughtPatternLibrary.default()
    compiler = SCTPSkillCompiler(
        library=library,
        include_initial_counterfactuals=True,
    )
    decisions: list[dict[str, Any]] = []
    completed = 0
    for domain in sorted({str(record["case_input"]["domain"]) for record in records}):
        retriever = LexicalCatalogRetriever.load(args.retriever_root / domain / "lexical_index.joblib")
        domain_records = [record for record in records if str(record["case_input"]["domain"]) == domain]
        for record in domain_records:
            case = record["case_input"]
            memory = PublicDialogueMemory.begin(
                domain=domain,
                initial_user_utterance=str(case["legacy_source"]["query"]),
            )
            state = _gold_state(record=record, memory=memory, retriever=retriever)
            frontier = compiler.compile(
                memory=memory,
                state=state,
                retriever=retriever,
                decision_posterior=_posterior(str(case["family"])),
            )
            decisions.append(_decision_row(record=record, frontier=frontier))
            completed += 1
            if completed == len(records) or completed % 5 == 0:
                print(f"阶段进度：已构建 {completed}/{len(records)} 个 CIV 决策分叉。", flush=True)
        del retriever
        gc.collect()
    print("步骤 2/5 完成：所有训练标签均来自离线开发分叉，在线模型只保留公开特征。", flush=True)

    print("步骤 3/5：把最优分叉与其余合法分叉转换为成对偏好。", flush=True)
    train_decisions = [row for row in decisions if row["case_id"] in train_ids]
    dev_decisions = [row for row in decisions if row["case_id"] in dev_ids]
    differences, labels = _pairwise_rows(train_decisions)
    print(
        f"步骤 3/5 完成：pairwise rows={len(labels)}，feature_dim={differences.shape[1]}。",
        flush=True,
    )

    print("步骤 4/5：训练透明的线性 pairwise CIV，并在留出的 10 条开发轨迹上评估。", flush=True)
    model = CIVPairwiseLinearRanker.fit(differences, labels)
    train_metrics = _evaluate(model, train_decisions)
    dev_metrics = _evaluate(model, dev_decisions)
    print(
        "步骤 4/5 完成："
        f"dev optimal={dev_metrics['optimal_proposal_rate']:.3f}，"
        f"dev action={dev_metrics['initial_action_accuracy']:.3f}。",
        flush=True,
    )

    print("步骤 5/5：保存模型、公开特征契约和 evaluator-only 开发诊断。", flush=True)
    output_dir = args.output_root / model.model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.joblib")
    public_manifest = {
        "version": "mt_civ_pairwise_training_v1",
        "model_id": model.model_id,
        "feature_names": list(CIV_PUBLIC_FEATURE_NAMES),
        "coefficients": model.coefficients.tolist(),
        "train_case_count": len(train_decisions),
        "dev_case_count": len(dev_decisions),
        "pairwise_row_count": len(labels),
        "source_disjoint_from_frozen_test": True,
        "thought_pattern_library_id": library.library_id,
        "compiler_id": compiler.compiler_id,
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
    }
    (output_dir / "manifest.json").write_text(
        canonical_json(public_manifest) + "\n",
        encoding="utf-8",
    )
    diagnostic_rows: list[dict[str, Any]] = []
    for decision in decisions:
        frontier: SCTPActionFrontier = decision["frontier"]
        selection = model.select(frontier)
        diagnostic_rows.append(
            {
                "case_id": decision["case_id"],
                "split": "train" if decision["case_id"] in train_ids else "dev",
                "family": decision["family"],
                "best_proposal_ids": decision["best_proposal_ids"],
                "selected_proposal_id": selection.selected_proposal_id,
                "selected_is_optimal": selection.selected_proposal_id in decision["best_proposal_ids"],
                "rewards": decision["rewards"],
            }
        )
    (output_dir / "development_diagnostics.private.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in diagnostic_rows),
        encoding="utf-8",
    )
    print(f"全部完成：CIV pairwise 模型已保存到 {output_dir.resolve()}。", flush=True)


if __name__ == "__main__":
    main()
