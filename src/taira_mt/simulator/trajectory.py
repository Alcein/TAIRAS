"""Versioned public/private trajectory plans for offline simulator audits.

The plans in this module are an engineering protocol only.  They define a
bounded, deterministic crossing of a frozen semantic case with the frozen
persona schedule and a trajectory index.  They intentionally do *not* run a
dialogue, inspect a rollout outcome, or modify the existing episode and
counterfactual schemas.

The public view is deliberately small: it contains opaque identifiers, the
base case lineage, split group, and a commitment to private seeds.  Persona
identity, seeds, and the hidden-constraint reveal order remain exclusively in
the private plan.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator, model_validator

from taira_mt.models import StrictModel, UserPersona, normalize_slot_name
from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.simulator.personas import PERSONA_TEMPLATES, PersonaProfile

if TYPE_CHECKING:
    from taira_mt.data.scalable_goals import AutomaticPrivateGoal, AutomaticPublicGoal


TRAJECTORY_PROTOCOL_VERSION = "trajectory_protocol_v1"
TRAJECTORY_SEED_DERIVATION_VERSION = "trajectory_seed_derivation_v1"
TRAJECTORY_ID_PREFIX = "trajectory-"
EPISODE_ID_PREFIX = "episode-"
_OPAQUE_ID_HEX_LENGTH = 24
_SHA256_HEX_LENGTH = 64
_PUBLIC_PLAN_FIELDS = {
    "trajectory_id",
    "base_case_id",
    "episode_id",
    "trajectory_index",
    "protocol_version",
    "split_group_id",
    "seed_commitment",
}


def _require_opaque_id(value: str, *, prefix: str, field_name: str) -> str:
    pattern = rf"{re.escape(prefix)}[0-9a-f]{{{_OPAQUE_ID_HEX_LENGTH}}}"
    if re.fullmatch(pattern, value) is None:
        raise ValueError(f"{field_name} must be an opaque {prefix}<sha256-prefix> identifier")
    return value


def _require_sha256(value: str, *, field_name: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


class PublicTrajectoryPlan(StrictModel):
    """The shareable trajectory schedule record.

    This schema deliberately has no persona, raw seed, or reveal-order field.
    ``seed_commitment`` can later be verified against the corresponding private
    plan without exposing either execution seed in a public artifact.
    """

    trajectory_id: str = Field(min_length=1)
    base_case_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    trajectory_index: int = Field(ge=0)
    protocol_version: str = TRAJECTORY_PROTOCOL_VERSION
    split_group_id: str | None = None
    seed_commitment: str = Field(min_length=_SHA256_HEX_LENGTH, max_length=_SHA256_HEX_LENGTH)

    @field_validator("trajectory_id")
    @classmethod
    def validate_trajectory_id(cls, value: str) -> str:
        return _require_opaque_id(value, prefix=TRAJECTORY_ID_PREFIX, field_name="trajectory_id")

    @field_validator("episode_id")
    @classmethod
    def validate_episode_id(cls, value: str) -> str:
        return _require_opaque_id(value, prefix=EPISODE_ID_PREFIX, field_name="episode_id")

    @field_validator("protocol_version")
    @classmethod
    def validate_protocol_version(cls, value: str) -> str:
        if value != TRAJECTORY_PROTOCOL_VERSION:
            raise ValueError(f"unsupported trajectory protocol version: {value}")
        return value

    @field_validator("seed_commitment")
    @classmethod
    def validate_seed_commitment(cls, value: str) -> str:
        return _require_sha256(value, field_name="seed_commitment")


class PrivateTrajectoryPlan(PublicTrajectoryPlan):
    """The confidential execution details corresponding to one public plan."""

    behavior_seed: int = Field(ge=0)
    counterfactual_seed: int = Field(ge=0)
    persona: UserPersona
    persona_profile: PersonaProfile
    reveal_order: list[str] = Field(default_factory=list)
    plan_hash: str = Field(min_length=_SHA256_HEX_LENGTH, max_length=_SHA256_HEX_LENGTH)

    @field_validator("behavior_seed", "counterfactual_seed", mode="before")
    @classmethod
    def validate_seed_type(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("trajectory seeds must be integer values")
        return value

    @field_validator("reveal_order")
    @classmethod
    def validate_reveal_order(cls, values: list[str]) -> list[str]:
        normalized = [normalize_slot_name(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("reveal_order slots must be unique")
        return normalized

    @field_validator("plan_hash")
    @classmethod
    def validate_plan_hash_format(cls, value: str) -> str:
        return _require_sha256(value, field_name="plan_hash")

    @model_validator(mode="after")
    def validate_private_plan(self) -> PrivateTrajectoryPlan:
        if self.behavior_seed == self.counterfactual_seed:
            raise ValueError("behavior_seed and counterfactual_seed must be distinct namespaces")
        expected_persona_prefix = f"{self.persona_profile.value}-"
        if not self.persona.persona_id.startswith(expected_persona_prefix):
            raise ValueError("persona_id does not match persona_profile")
        expected_commitment = _seed_commitment(
            trajectory_id=self.trajectory_id,
            behavior_seed=self.behavior_seed,
            counterfactual_seed=self.counterfactual_seed,
        )
        if self.seed_commitment != expected_commitment:
            raise ValueError("seed_commitment does not match the private execution seeds")
        expected_hash = content_sha256(self.model_dump(mode="json", exclude={"plan_hash"}))
        if self.plan_hash != expected_hash:
            raise ValueError("plan_hash does not match the private trajectory plan")
        return self


@dataclass(frozen=True)
class TrajectoryPlanPair:
    """A validated public/private pair for a single scheduled trajectory."""

    public_plan: PublicTrajectoryPlan
    private_plan: PrivateTrajectoryPlan

    def __post_init__(self) -> None:
        public_payload = self.public_plan.model_dump(mode="json")
        private_public_payload = self.private_plan.model_dump(
            mode="json", include=_PUBLIC_PLAN_FIELDS
        )
        if public_payload != private_public_payload:
            raise ValueError("public and private trajectory plan fields do not correspond")


def _hash_payload(payload: dict[str, Any]) -> bytes:
    """Return a stable standard-library SHA-256 digest for a schedule payload."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()


def _derive_namespaced_seed(
    *,
    schedule_seed: int,
    base_case_id: str,
    trajectory_id: str,
    namespace: str,
) -> int:
    """Derive one non-negative seed without consuming process-global randomness."""

    payload = {
        "derivation_version": TRAJECTORY_SEED_DERIVATION_VERSION,
        "namespace": namespace,
        "schedule_seed": schedule_seed,
        "base_case_id": base_case_id,
        "trajectory_id": trajectory_id,
    }
    return int.from_bytes(_hash_payload(payload)[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def _trajectory_id(
    *,
    base_case_id: str,
    persona_profile: PersonaProfile,
    trajectory_index: int,
    schedule_seed: int,
) -> str:
    """Create an opaque ID whose preimage is intentionally private."""

    identity = {
        "protocol_version": TRAJECTORY_PROTOCOL_VERSION,
        "base_case_id": base_case_id,
        "persona_profile": persona_profile.value,
        "trajectory_index": trajectory_index,
        "schedule_seed": schedule_seed,
    }
    return f"{TRAJECTORY_ID_PREFIX}{content_sha256(identity)[:_OPAQUE_ID_HEX_LENGTH]}"


def _episode_id(trajectory_id: str) -> str:
    payload = {
        "protocol_version": TRAJECTORY_PROTOCOL_VERSION,
        "trajectory_id": trajectory_id,
    }
    return f"{EPISODE_ID_PREFIX}{content_sha256(payload)[:_OPAQUE_ID_HEX_LENGTH]}"


def _seed_commitment(
    *, trajectory_id: str, behavior_seed: int, counterfactual_seed: int
) -> str:
    """Commit to private seeds while exposing neither seed in the public plan."""

    return content_sha256(
        {
            "protocol_version": TRAJECTORY_PROTOCOL_VERSION,
            "commitment_version": TRAJECTORY_SEED_DERIVATION_VERSION,
            "trajectory_id": trajectory_id,
            "behavior_seed": behavior_seed,
            "counterfactual_seed": counterfactual_seed,
        }
    )


def _reveal_order(
    private_goal: AutomaticPrivateGoal,
    *,
    schedule_seed: int,
    trajectory_id: str,
) -> list[str]:
    """Order unrevealed hidden slots deterministically before any interaction.

    Initial constraints are already public and are not candidates for a future
    reveal.  The ordering key uses only frozen goal slots and schedule inputs;
    it does not inspect an action, user response, acceptance label, or rollout
    result.
    """

    slots = [
        constraint.slot
        for constraint in private_goal.hidden_goal.constraints
        if not constraint.initially_revealed
    ]
    if len(slots) != len(set(slots)):
        raise ValueError("cannot create a slot-level reveal order with duplicate hidden slots")

    def order_key(slot: str) -> tuple[bytes, str]:
        return (
            _hash_payload(
                {
                    "protocol_version": TRAJECTORY_PROTOCOL_VERSION,
                    "namespace": "reveal_order",
                    "schedule_seed": schedule_seed,
                    "trajectory_id": trajectory_id,
                    "slot": slot,
                }
            ),
            slot,
        )

    return sorted(slots, key=order_key)


def _validate_goal_correspondence(
    public_goal: AutomaticPublicGoal, private_goal: AutomaticPrivateGoal
) -> None:
    if public_goal.case_id != private_goal.case_id:
        raise ValueError("public and private goals must share the same base case ID")
    if public_goal.domain != private_goal.hidden_goal.domain:
        raise ValueError("public and private goals must share the same domain")
    if public_goal.initial_request != private_goal.hidden_goal.initial_request:
        raise ValueError("public and private goals must share the same initial request")


def make_trajectory_plans(
    public_goal: AutomaticPublicGoal,
    private_goal: AutomaticPrivateGoal,
    trajectories_per_persona: int,
    schedule_seed: int,
) -> tuple[TrajectoryPlanPair, ...]:
    """Create a fixed persona-by-trajectory schedule for one semantic case.

    The function crosses every entry in :data:`PERSONA_TEMPLATES` with each
    ``trajectory_index``.  It has no selection/retry path and returns opaque,
    deterministic IDs.  Changing a private acceptance outcome cannot change
    this schedule because no outcome field is read.
    """

    if isinstance(trajectories_per_persona, bool) or not isinstance(trajectories_per_persona, int):
        raise ValueError("trajectories_per_persona must be an integer")
    if trajectories_per_persona < 1:
        raise ValueError("trajectories_per_persona must be at least one")
    if isinstance(schedule_seed, bool) or not isinstance(schedule_seed, int):
        raise ValueError("schedule_seed must be an integer")
    _validate_goal_correspondence(public_goal, private_goal)

    pairs: list[TrajectoryPlanPair] = []
    for template in PERSONA_TEMPLATES:
        for trajectory_index in range(trajectories_per_persona):
            trajectory_id = _trajectory_id(
                base_case_id=public_goal.case_id,
                persona_profile=template.profile,
                trajectory_index=trajectory_index,
                schedule_seed=schedule_seed,
            )
            behavior_seed = _derive_namespaced_seed(
                schedule_seed=schedule_seed,
                base_case_id=public_goal.case_id,
                trajectory_id=trajectory_id,
                namespace="behavior",
            )
            counterfactual_seed = _derive_namespaced_seed(
                schedule_seed=schedule_seed,
                base_case_id=public_goal.case_id,
                trajectory_id=trajectory_id,
                namespace="counterfactual",
            )
            if behavior_seed == counterfactual_seed:
                # SHA-256 namespace collisions are extraordinarily unlikely,
                # but this deterministic fallback preserves the schema's hard
                # separation invariant without using a random retry.
                counterfactual_seed = _derive_namespaced_seed(
                    schedule_seed=schedule_seed,
                    base_case_id=public_goal.case_id,
                    trajectory_id=trajectory_id,
                    namespace="counterfactual_retry_1",
                )
            if behavior_seed == counterfactual_seed:
                raise AssertionError("namespaced trajectory seed derivation collided twice")

            public_plan = PublicTrajectoryPlan(
                trajectory_id=trajectory_id,
                base_case_id=public_goal.case_id,
                episode_id=_episode_id(trajectory_id),
                trajectory_index=trajectory_index,
                protocol_version=TRAJECTORY_PROTOCOL_VERSION,
                split_group_id=public_goal.split_group_id,
                seed_commitment=_seed_commitment(
                    trajectory_id=trajectory_id,
                    behavior_seed=behavior_seed,
                    counterfactual_seed=counterfactual_seed,
                ),
            )
            persona = template.instantiate(trajectory_id)
            private_payload = {
                **public_plan.model_dump(mode="json"),
                "behavior_seed": behavior_seed,
                "counterfactual_seed": counterfactual_seed,
                "persona": persona.model_dump(mode="json"),
                "persona_profile": template.profile.value,
                "reveal_order": _reveal_order(
                    private_goal,
                    schedule_seed=schedule_seed,
                    trajectory_id=trajectory_id,
                ),
            }
            private_plan = PrivateTrajectoryPlan.model_validate(
                {
                    **private_payload,
                    "plan_hash": content_sha256(private_payload),
                }
            )
            pairs.append(TrajectoryPlanPair(public_plan=public_plan, private_plan=private_plan))

    ordered_pairs = tuple(sorted(pairs, key=lambda pair: pair.public_plan.trajectory_id))
    trajectory_ids = [pair.public_plan.trajectory_id for pair in ordered_pairs]
    episode_ids = [pair.public_plan.episode_id for pair in ordered_pairs]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise AssertionError("trajectory schedule generated duplicate opaque trajectory IDs")
    if len(episode_ids) != len(set(episode_ids)):
        raise AssertionError("trajectory schedule generated duplicate opaque episode IDs")
    expected_count = len(PERSONA_TEMPLATES) * trajectories_per_persona
    if len(ordered_pairs) != expected_count:
        raise AssertionError("trajectory schedule did not cover every frozen persona/index cell")
    return ordered_pairs


__all__ = [
    "EPISODE_ID_PREFIX",
    "PrivateTrajectoryPlan",
    "PublicTrajectoryPlan",
    "TRAJECTORY_ID_PREFIX",
    "TRAJECTORY_PROTOCOL_VERSION",
    "TRAJECTORY_SEED_DERIVATION_VERSION",
    "TrajectoryPlanPair",
    "make_trajectory_plans",
]
