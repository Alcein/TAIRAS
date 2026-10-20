"""Deterministic, zero-API pilot over the three audited legacy domains."""

from taira_mt.pilot.batch import PilotRunResult, run_offline_pilot
from taira_mt.pilot.construction import DEFAULT_ANCHOR_SPECS, PilotScenario

__all__ = ["DEFAULT_ANCHOR_SPECS", "PilotRunResult", "PilotScenario", "run_offline_pilot"]
