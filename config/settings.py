"""
MovieLens 32M Recommendation System - Central Settings & Configuration
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

configured_data_dir = os.environ.get("MOVIELENS_DATA_DIR", "").strip()
DATA_DIR = Path(configured_data_dir) if configured_data_dir else BASE_DIR / "data"

MOVIES_FILE = DATA_DIR / "movies.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"
TAGS_FILE = DATA_DIR / "tags.csv"
LINKS_FILE = DATA_DIR / "links.csv"
README_FILE = DATA_DIR / "README.txt"
CHECKSUMS_FILE = DATA_DIR / "checksums.txt"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR
REPORTS_DIR = BASE_DIR / "reports"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DTYPE_MAPPINGS = {
    "ratings": {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    "movies": {"movieId": "int32", "title": "string", "genres": "string"},
    "tags": {"userId": "int32", "movieId": "int32", "tag": "string", "timestamp": "int64"},
    "links": {"movieId": "int32", "imdbId": "int32", "tmdbId": "float64"},
}

RELEVANCE_THRESHOLD = 4.0
DEFAULT_TOP_K_LIST = [5, 10, 20]
RANDOM_SEED = 42
