"""
Configuration Package
"""

try:
    from .settings import (
        BASE_DIR,
        DATA_DIR,
        MOVIES_FILE,
        RATINGS_FILE,
        TAGS_FILE,
        LINKS_FILE,
        README_FILE,
        CHECKSUMS_FILE,
        ARTIFACTS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        DTYPE_MAPPINGS,
        RELEVANCE_THRESHOLD,
        DEFAULT_TOP_K_LIST,
        RANDOM_SEED,
    )
except ImportError:
    from config.settings import (
        BASE_DIR,
        DATA_DIR,
        MOVIES_FILE,
        RATINGS_FILE,
        TAGS_FILE,
        LINKS_FILE,
        README_FILE,
        CHECKSUMS_FILE,
        ARTIFACTS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        DTYPE_MAPPINGS,
        RELEVANCE_THRESHOLD,
        DEFAULT_TOP_K_LIST,
        RANDOM_SEED,
    )

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "MOVIES_FILE",
    "RATINGS_FILE",
    "TAGS_FILE",
    "LINKS_FILE",
    "README_FILE",
    "CHECKSUMS_FILE",
    "ARTIFACTS_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "DTYPE_MAPPINGS",
    "RELEVANCE_THRESHOLD",
    "DEFAULT_TOP_K_LIST",
    "RANDOM_SEED",
]
