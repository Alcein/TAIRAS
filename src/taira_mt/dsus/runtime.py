"""Verified online DSUS predictor loading for the TAIRAS Agent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from taira_mt.artifacts import verify_artifact_bundle
from taira_mt.dsus.modeling import (
    FactorizedDSUSTfidfModel,
    predict_factorized_dsus_tfidf,
)
from taira_mt.dsus.schemas import DSUSPosterior, PublicDSUSFeature

DSUS_TFIDF_RUNTIME_VERSION = "dsus_public_history_tfidf_v1"
_SUPPORTED_MODEL_SCHEMA_VERSIONS = {
    DSUS_TFIDF_RUNTIME_VERSION,
    # The OOF bundle carries additional offline-only files, but its declared
    # ``model.joblib`` still obeys the same PublicDSUSFeature runtime contract.
    "parser_aligned_dsus_oof_posteriors_v1",
}
_SUPPORTED_FACTOR_MODEL_ROLES = {
    "public_history_factorized_dsus_development_baseline",
    "parser_aligned_public_history_factorized_dsus_development_baseline",
    "parser_aligned_public_history_factorized_dsus_oof_full_train_predictor",
}


class PublicDSUSPredictor(Protocol):
    """A state estimator that consumes only the online public feature contract."""

    predictor_id: str

    def predict(self, feature: PublicDSUSFeature) -> DSUSPosterior: ...


@dataclass(frozen=True)
class LoadedFactorizedDSUSPredictor:
    """A content-verified TF-IDF DSUS model ready for online inference."""

    predictor_id: str
    manifest_sha256: str
    model: FactorizedDSUSTfidfModel
    # Fold models inside legacy OOF artifacts predate an explicit parser field.
    # They are offline-only and may therefore retain ``None``; online local
    # controls require a non-null exact match before running.
    source_interpreter_id: str | None = None

    def predict(self, feature: PublicDSUSFeature) -> DSUSPosterior:
        predictions = predict_factorized_dsus_tfidf(self.model, [feature])
        if len(predictions) != 1:  # pragma: no cover - defensive model contract
            raise RuntimeError("DSUS runtime predictor did not return exactly one posterior")
        return predictions[0]


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"DSUS model manifest is missing non-empty {key}")
    return value


def load_factorized_dsus_predictor(run_dir: Path) -> LoadedFactorizedDSUSPredictor:
    """Load the registered baseline only after every declared file verifies."""

    bundle = verify_artifact_bundle(run_dir)
    manifest = bundle.manifest
    if manifest.get("schema_version") not in _SUPPORTED_MODEL_SCHEMA_VERSIONS:
        raise ValueError("unsupported DSUS runtime model schema")
    if manifest.get("artifact_role") not in _SUPPORTED_FACTOR_MODEL_ROLES:
        raise ValueError("artifact is not a supported public-history DSUS baseline")
    config = manifest.get("config")
    if not isinstance(config, Mapping) or config.get("version") != DSUS_TFIDF_RUNTIME_VERSION:
        raise ValueError("DSUS runtime model manifest has an incompatible config")
    if config.get("external_calls") is not False:
        raise ValueError("DSUS runtime model must be a zero-external-call artifact")
    source_interpreter_id = config.get("interpreter_id")
    if source_interpreter_id is not None and (
        not isinstance(source_interpreter_id, str) or not source_interpreter_id
    ):
        raise ValueError("DSUS runtime model has an invalid source interpreter ID")
    model_path = bundle.require_output("model.joblib")
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - exercised only without ML extras
        raise RuntimeError("load the DSUS runtime with the project's ML dependencies installed") from exc
    model = joblib.load(model_path)
    if not isinstance(model, FactorizedDSUSTfidfModel):
        raise ValueError("DSUS model.joblib does not contain a factorized TF-IDF model")
    return LoadedFactorizedDSUSPredictor(
        predictor_id=_required_string(manifest, "model_id"),
        manifest_sha256=bundle.manifest_sha256,
        source_interpreter_id=source_interpreter_id,
        model=model,
    )
