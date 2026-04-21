"""
KBQA Experience Knowledge Base - Configuration.

Centralized configuration for the experience KB system.
Adapted from vkbqa/config.py pattern.
"""

import os


class Config:
    # ---- Paths ----
    # Base directory for the experience KB
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # KB storage directory
    KB_DIR = os.getenv("EXP_KB_DIR", os.path.join(BASE_DIR, "data", "experience_kb"))
    
    # Trajectory storage
    TRAJECTORY_DIR = os.getenv("EXP_TRAJECTORY_DIR", os.path.join(BASE_DIR, "data", "trajectories"))
    
    # Raw trajectory input (for collector)
    RAW_TRAJECTORY_DIR = os.getenv("EXP_RAW_TRAJECTORY_DIR", "")

    # ---- Embedding Model ----
    # Sentence-transformers model name (text-only, no CLIP needed)
    EMBEDDING_MODEL = os.getenv("EXP_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    # Alternative models: "BAAI/bge-base-en-v1.5", "all-mpnet-base-v2"
    
    # ---- LLM Configuration (for rule extraction and consolidation) ----
    # Same pattern as vkbqa config
    LLM_BASE_URL = os.getenv("EXP_LLM_BASE_URL", os.getenv("LOCAL_API_BASE_URL", "http://localhost:8000/v1"))
    LLM_API_KEY = os.getenv("EXP_LLM_API_KEY", os.getenv("LOCAL_API_KEY", "EMPTY"))
    LLM_MODEL_NAME = os.getenv("EXP_LLM_MODEL_NAME", "qwen3.5-9b")
    
    # Alternative: use SiliconFlow cloud API
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
    SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

    # ---- Retrieval Parameters ----
    SEARCH_TOP_K = int(os.getenv("EXP_SEARCH_TOP_K", "5"))
    SEARCH_THRESHOLD = float(os.getenv("EXP_SEARCH_THRESHOLD", "0.5"))
    COMPACT_GUIDANCE = os.getenv("EXP_COMPACT_GUIDANCE", "false").lower() == "true"

    # ---- Pipeline Parameters ----
    MAX_CORRECTION_STEPS = int(os.getenv("EXP_MAX_STEPS", "5"))
    EXPERIENCE_TOP_K = int(os.getenv("EXP_TOP_K", "3"))

    # ---- Consolidation Parameters ----
    CONSOLIDATION_SIM_THRESHOLD = float(os.getenv("EXP_CONSOLIDATE_SIM", "0.75"))
    CONSOLIDATION_MIN_RULES = int(os.getenv("EXP_CONSOLIDATE_MIN", "3"))
    CONSOLIDATION_MAX_PER_BATCH = int(os.getenv("EXP_CONSOLIDATE_BATCH", "10"))

    # ---- API Stability Controls (from vkbqa) ----
    API_CONNECT_TIMEOUT = int(os.getenv("API_CONNECT_TIMEOUT", "20"))
    API_READ_TIMEOUT = int(os.getenv("API_READ_TIMEOUT", "300"))
    API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))
    API_RETRY_BASE_DELAY = float(os.getenv("API_RETRY_BASE_DELAY", "2.0"))
    API_RETRY_JITTER_MAX = float(os.getenv("API_RETRY_JITTER_MAX", "0.5"))
    API_MAX_TOKENS = int(os.getenv("API_MAX_TOKENS", "4096"))

    @classmethod
    def get_llm_config(cls):
        """Return LLM client configuration dict."""
        return {
            "base_url": cls.LLM_BASE_URL,
            "api_key": cls.LLM_API_KEY,
            "model_name": cls.LLM_MODEL_NAME,
            "connect_timeout": cls.API_CONNECT_TIMEOUT,
            "read_timeout": cls.API_READ_TIMEOUT,
            "max_retries": cls.API_MAX_RETRIES,
            "retry_base_delay": cls.API_RETRY_BASE_DELAY,
            "retry_jitter_max": cls.API_RETRY_JITTER_MAX,
            "max_tokens": cls.API_MAX_TOKENS,
        }

    @classmethod
    def get_kb_config(cls):
        """Return KB configuration dict."""
        return {
            "kb_dir": cls.KB_DIR,
            "embedding_model_name": cls.EMBEDDING_MODEL,
        }

    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("=== Experience KB Configuration ===")
        print(f"  KB_DIR: {cls.KB_DIR}")
        print(f"  TRAJECTORY_DIR: {cls.TRAJECTORY_DIR}")
        print(f"  EMBEDDING_MODEL: {cls.EMBEDDING_MODEL}")
        print(f"  LLM_BASE_URL: {cls.LLM_BASE_URL}")
        print(f"  LLM_MODEL_NAME: {cls.LLM_MODEL_NAME}")
        print(f"  SEARCH_TOP_K: {cls.SEARCH_TOP_K}")
        print(f"  SEARCH_THRESHOLD: {cls.SEARCH_THRESHOLD}")
        print(f"  MAX_CORRECTION_STEPS: {cls.MAX_CORRECTION_STEPS}")
        print(f"  CONSOLIDATION_SIM_THRESHOLD: {cls.CONSOLIDATION_SIM_THRESHOLD}")
        print("==================================")
