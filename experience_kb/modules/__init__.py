"""
KBQA Experience Knowledge Base - Modules Package.

Contains:
- trajectory_collector: Parses step-level KBQA/SPARQL trajectories into structured episodes
- experience_extractor: Uses LLM to extract structured error-correction rules from episodes
- knowledge_base: Core FAISS-indexed KB manager for error-correction rules
- rule_retriever: Runtime retrieval interface for LLM prompt injection
- pipeline_integration: Adapters to integrate experience KB into KBQA pipelines
"""

from .trajectory_collector import TrajectoryCollector, EPISODE_TYPES
from .experience_extractor import ExperienceExtractor, LLMClient, RULE_TYPES
from .knowledge_base import ExperienceKB
from .rule_retriever import RuleRetriever
from .pipeline_integration import (
    ExperienceGuidedPipeline,
    KBQAPipelineAdapter,
    SPARQLExecutor,
)

__all__ = [
    "TrajectoryCollector",
    "ExperienceExtractor",
    "LLMClient",
    "ExperienceKB",
    "RuleRetriever",
    "ExperienceGuidedPipeline",
    "KBQAPipelineAdapter",
    "SPARQLExecutor",
    "EPISODE_TYPES",
    "RULE_TYPES",
]
