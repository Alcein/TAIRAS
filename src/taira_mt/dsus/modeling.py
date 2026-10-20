"""Public-history DSUS baselines and posterior serialization.

This module is intentionally small: it validates that the frozen evolving-user
benchmark can support a learned state estimate before a heavier encoder is
introduced.  It never consumes private labels at prediction time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from taira_mt.dsus.schemas import (
    DSUSPosterior,
    DSUSStateFeature,
    DSUSStateLabel,
    PublicDSUSFeature,
)

DSUS_CATEGORICAL_FACTORS = (
    "intent_completeness",
    "intent_stability",
    "candidate_feasibility",
    "dialogue_phase",
    "last_user_act",
    "explanation_pending",
    "patience_risk",
    "satisfaction_band",
)
DSUS_REGRESSION_FACTORS = ("patience_remaining_ratio", "satisfaction")


@dataclass
class FactorizedDSUSTfidfModel:
    """A shared text representation with independently auditable factor heads."""

    vectorizer: TfidfVectorizer
    classifiers: dict[str, Any]
    regressors: dict[str, Ridge]


DSUSPredictionFeature = DSUSStateFeature | PublicDSUSFeature


def _online_feature(feature: DSUSPredictionFeature) -> PublicDSUSFeature:
    return feature.online_feature() if isinstance(feature, DSUSStateFeature) else feature


def _feature_text(feature: DSUSPredictionFeature) -> str:
    feature = _online_feature(feature)
    constraints = " ".join(
        (
            f"constraint_slot={item.slot} "
            f"constraint={json.dumps(item.predicate.model_dump(mode='json'), sort_keys=True)}"
        )
        for item in feature.public_constraints
    )
    asked = " ".join(sorted(feature.asked_question_targets)) or "none"
    last_action = feature.last_system_action_type.value if feature.last_system_action_type else "none"
    return "\n".join(
        (
            f"Domain: {feature.domain}.",
            f"Turn: {feature.turn_index}.",
            f"Dialogue:\n{feature.serialized_dialogue}",
            f"Public constraints: {constraints or 'none'}.",
            f"Verified candidates: {feature.public_verified_candidate_count}.",
            f"Unknown candidates: {feature.public_unknown_candidate_count}.",
            f"Asked targets: {asked}.",
            f"Rejected item count: {feature.rejected_item_count}.",
            f"Last system action: {last_action}.",
        )
    )


def _label_targets(label: DSUSStateLabel) -> dict[str, str | float]:
    return {
        "intent_completeness": label.intent_completeness.value,
        "intent_stability": label.intent_stability.value,
        "candidate_feasibility": label.candidate_feasibility.value,
        "dialogue_phase": label.dialogue_phase.value,
        "last_user_act": label.last_user_act.value if label.last_user_act else "none",
        "explanation_pending": str(label.explanation_pending).lower(),
        "patience_risk": label.patience_risk.value,
        "satisfaction_band": label.satisfaction_band.value,
        "patience_remaining_ratio": label.patience_remaining_ratio,
        "satisfaction": label.satisfaction,
    }


def deduplicate_semantic_states(
    features: list[DSUSStateFeature], labels: list[DSUSStateLabel]
) -> tuple[list[DSUSStateFeature], list[DSUSStateLabel]]:
    """Keep one deterministic rendering of every semantic state.

    Five behavior policies can visit the identical semantic state.  Retaining
    all copies would silently let frequent diagnostic policies reweight the
    DSUS training objective.  Public language still varies across trajectories,
    but the canonical first state ID is deterministic and split-local.
    """

    label_by_id = {item.state_id: item for item in labels}
    if set(label_by_id) != {item.state_id for item in features}:
        raise ValueError("DSUS features and labels must have an exact state join")
    selected: dict[str, DSUSStateFeature] = {}
    for feature in sorted(features, key=lambda item: item.state_id):
        existing = selected.get(feature.semantic_state_id)
        if existing is None:
            selected[feature.semantic_state_id] = feature
            continue
        old_label = label_by_id[existing.state_id]
        new_label = label_by_id[feature.state_id]
        if _label_targets(old_label) != _label_targets(new_label):
            raise ValueError("duplicate DSUS semantic state has inconsistent private labels")
        if existing.split != feature.split:
            raise ValueError("duplicate DSUS semantic state crosses a frozen split")
    selected_features = list(selected.values())
    return selected_features, [label_by_id[item.state_id] for item in selected_features]


def train_factorized_dsus_tfidf(
    features: list[DSUSStateFeature],
    labels: list[DSUSStateLabel],
    *,
    seed: int = 20260722,
) -> FactorizedDSUSTfidfModel:
    """Fit public-feature-only multi-head TF-IDF state estimation."""

    if not features:
        raise ValueError("DSUS training requires at least one state")
    label_by_id = {item.state_id: item for item in labels}
    if set(label_by_id) != {item.state_id for item in features}:
        raise ValueError("DSUS training features and labels differ")
    return _train_factorized_from_public_pairs(
        list(features),
        [label_by_id[item.state_id] for item in features],
        seed=seed,
    )


def _train_factorized_from_public_pairs(
    features: list[DSUSPredictionFeature],
    labels: list[DSUSStateLabel],
    *,
    seed: int,
) -> FactorizedDSUSTfidfModel:
    if len(features) != len(labels):
        raise ValueError("DSUS public features and private labels must have equal length")
    if not features:
        raise ValueError("DSUS training requires at least one state")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=40_000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([_feature_text(item) for item in features])
    targets = [_label_targets(label) for label in labels]
    classifiers: dict[str, Any] = {}
    for factor in DSUS_CATEGORICAL_FACTORS:
        target = np.asarray([str(row[factor]) for row in targets])
        if len(set(target)) == 1:
            classifier: Any = DummyClassifier(strategy="most_frequent")
        else:
            classifier = LogisticRegression(
                max_iter=2_000,
                class_weight="balanced",
                random_state=seed,
            )
        classifier.fit(matrix, target)
        classifiers[factor] = classifier
    regressors: dict[str, Ridge] = {}
    for factor in DSUS_REGRESSION_FACTORS:
        target = np.asarray([float(row[factor]) for row in targets])
        regressor = Ridge(alpha=1.0, solver="lsqr")
        regressor.fit(matrix, target)
        regressors[factor] = regressor
    return FactorizedDSUSTfidfModel(
        vectorizer=vectorizer,
        classifiers=classifiers,
        regressors=regressors,
    )


def train_factorized_dsus_tfidf_online_features(
    features: list[PublicDSUSFeature],
    labels: list[DSUSStateLabel],
    *,
    seed: int = 20260722,
) -> FactorizedDSUSTfidfModel:
    """Fit DSUS from parser-aligned online features and aligned private labels.

    The pairing is positional because parser-aligned artifacts retain state IDs
    outside ``PublicDSUSFeature`` precisely to keep those lineage fields out of
    model input.
    """

    return _train_factorized_from_public_pairs(list(features), list(labels), seed=seed)


def _prediction_distribution(classifier: Any, probabilities: np.ndarray) -> dict[str, float]:
    return {
        str(label): float(probability)
        for label, probability in zip(classifier.classes_, probabilities, strict=True)
    }


def predict_factorized_dsus_tfidf(
    model: FactorizedDSUSTfidfModel,
    features: list[DSUSPredictionFeature],
) -> list[DSUSPosterior]:
    """Return factor probability distributions for public state snapshots."""

    if not features:
        return []
    matrix = model.vectorizer.transform([_feature_text(item) for item in features])
    distributions: list[dict[str, dict[str, float]]] = [dict() for _ in features]
    for factor, classifier in model.classifiers.items():
        probabilities = classifier.predict_proba(matrix)
        for output, row in zip(distributions, probabilities, strict=True):
            output[factor] = _prediction_distribution(classifier, row)
    regression_values = {
        factor: np.clip(regressor.predict(matrix), 0.0, 1.0)
        for factor, regressor in model.regressors.items()
    }
    return [
        DSUSPosterior(
            **distribution,
            patience_remaining_ratio=float(regression_values["patience_remaining_ratio"][index]),
            satisfaction=float(regression_values["satisfaction"][index]),
        )
        for index, distribution in enumerate(distributions)
    ]


def posterior_from_private_label(label: DSUSStateLabel) -> DSUSPosterior:
    """Build an evaluator-only one-hot posterior for upper-bound diagnostics."""

    targets = _label_targets(label)
    return DSUSPosterior(
        **{
            factor: {str(targets[factor]): 1.0}
            for factor in DSUS_CATEGORICAL_FACTORS
        },
        patience_remaining_ratio=float(targets["patience_remaining_ratio"]),
        satisfaction=float(targets["satisfaction"]),
    )


def _multiclass_brier(distributions: list[dict[str, float]], truths: list[str]) -> float:
    values = []
    for distribution, truth in zip(distributions, truths, strict=True):
        values.append(
            sum(
                (probability - float(label == truth)) ** 2
                for label, probability in distribution.items()
            )
        )
    return float(np.mean(values))


def _expected_calibration_error(
    distributions: list[dict[str, float]], truths: list[str], *, bins: int = 10
) -> float:
    confidences = []
    correctness = []
    for distribution, truth in zip(distributions, truths, strict=True):
        prediction, confidence = DSUSPosterior.maximum_probability(distribution)
        confidences.append(confidence)
        correctness.append(float(prediction == truth))
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            position
            for position, value in enumerate(confidences)
            if lower <= value < upper or (index == bins - 1 and value == upper)
        ]
        if not members:
            continue
        mean_confidence = float(np.mean([confidences[item] for item in members]))
        mean_accuracy = float(np.mean([correctness[item] for item in members]))
        ece += len(members) / len(confidences) * abs(mean_accuracy - mean_confidence)
    return ece


def evaluate_dsus_posteriors(
    labels: list[DSUSStateLabel],
    predictions: list[DSUSPosterior],
) -> dict[str, dict[str, float]]:
    """Report factor recovery and calibration without policy-action targets."""

    if len(labels) != len(predictions):
        raise ValueError("DSUS prediction and label counts differ")
    targets = [_label_targets(item) for item in labels]
    metrics: dict[str, dict[str, float]] = {}
    for factor in DSUS_CATEGORICAL_FACTORS:
        truths = [str(row[factor]) for row in targets]
        distributions = [getattr(item, factor) for item in predictions]
        predicted = [DSUSPosterior.maximum_probability(item)[0] for item in distributions]
        metrics[factor] = {
            "accuracy": float(accuracy_score(truths, predicted)),
            "macro_f1": float(f1_score(truths, predicted, average="macro", zero_division=0)),
            "brier": _multiclass_brier(distributions, truths),
            "ece_10": _expected_calibration_error(distributions, truths),
        }
    for factor in DSUS_REGRESSION_FACTORS:
        truths = [float(row[factor]) for row in targets]
        values = [float(getattr(item, factor)) for item in predictions]
        metrics[factor] = {"mae": float(mean_absolute_error(truths, values))}
    return metrics
