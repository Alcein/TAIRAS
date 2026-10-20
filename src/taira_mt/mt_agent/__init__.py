"""Paper-canonical multi-turn recommendation Agent implementation.

Legacy Agent modules remain available as audited references.  New code for the
frozen full-catalog benchmark belongs here so the paper-facing execution path
does not silently depend on the retired 70-candidate environment.
"""

from .baselines import LegacyTAIRALLMPolicy
from .civ import CIVRanker
from .direct_llm import DirectLLMToolPolicy
from .literature_baselines import (
    ChatCRSGoalGuidedPolicy,
    ItemCenteredIncrementalPolicy,
    LLMAskThenRecommendPolicy,
    RecMindSelfInspiringPolicy,
)
from .proposed import TAIRAMTPolicy, TAIRASPolicy
from .retrieval import (
    CatalogRetrievalItem,
    CatalogRetrievalResult,
    LexicalCatalogRetriever,
    SearchSpec,
)
from .runner import MTAgentRunner, MultiDomainRetrieverRegistry
from .sctp import SCTPSkillCompiler
from .skills import ThoughtPatternLibrary
from .state import LLMDSUSEstimator, RuleDSUSEstimator

__all__ = [
    "CatalogRetrievalItem",
    "CatalogRetrievalResult",
    "ChatCRSGoalGuidedPolicy",
    "CIVRanker",
    "DirectLLMToolPolicy",
    "ItemCenteredIncrementalPolicy",
    "LexicalCatalogRetriever",
    "LegacyTAIRALLMPolicy",
    "LLMAskThenRecommendPolicy",
    "LLMDSUSEstimator",
    "MTAgentRunner",
    "MultiDomainRetrieverRegistry",
    "RecMindSelfInspiringPolicy",
    "RuleDSUSEstimator",
    "SearchSpec",
    "SCTPSkillCompiler",
    "TAIRAMTPolicy",
    "TAIRASPolicy",
    "ThoughtPatternLibrary",
]
