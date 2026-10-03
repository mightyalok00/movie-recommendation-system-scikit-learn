"""
MovieLens 32M Recommendation System - Central Settings & Configuration
======================================================================
Provides dynamic path resolution, memory optimization data types,
and model hyperparameter defaults.
"""

import os
from pathlib import Path

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset Search Paths (Prioritizes E:\ml-32m, falls back to environment or local data/)
DEFAULT_DATA_DIR = Path(os.environ.get("MOVIELENS_DATA_DIR", r"E:\ml-32m"))
FALLBACK_DATA_DIR = BASE_DIR / "data"

if DEFAULT_DATA_DIR.exists():
    DATA_DIR = DEFAULT_DATA_DIR
elif FALLBACK_DATA_DIR.exists():
    DATA_DIR = FALLBACK_DATA_DIR
else:
    DATA_DIR = DEFAULT_DATA_DIR

# Dataset file paths
MOVIES_FILE = DATA_DIR / "movies.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"
TAGS_FILE = DATA_DIR / "tags.csv"
LINKS_FILE = DATA_DIR / "links.csv"
README_FILE = DATA_DIR / "README.txt"
CHECKSUMS_FILE = DATA_DIR / "checksums.txt"

# Output and Artifact Directories
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR  # Backward-compatibility alias
REPORTS_DIR = BASE_DIR / "reports"

# Ensure output directories exist
for folder in [ARTIFACTS_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Processing & Memory Optimization Defaults
DTYPE_MAPPINGS = {
    "ratings": {
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32",
        "timestamp": "int64",
    },
    "movies": {
        "movieId": "int32",
        "title": "string",
        "genres": "string",
    },
    "tags": {
        "userId": "int32",
        "movieId": "int32",
        "tag": "string",
        "timestamp": "int64",
    },
    "links": {
        "movieId": "int32",
        "imdbId": "int32",
        "tmdbId": "float64",
    }
}

# Evaluation Configuration
RELEVANCE_THRESHOLD = 4.0   # Ratings >= 4.0 are considered relevant items
DEFAULT_TOP_K_LIST = [5, 10, 20]
RANDOM_SEED = 42
