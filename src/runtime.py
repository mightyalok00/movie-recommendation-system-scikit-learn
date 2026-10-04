"""
Reusable recommendation runtime.

The runtime is artifact-first so web applications can load pre-trained models
without reading or fitting the full MovieLens dataset. Training belongs in
scripts/build_artifacts.py and should never happen during application startup.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from config.settings import ARTIFACTS_DIR
from data.loader import MovieLensDataLoader
from src.cold_start import GenrePriorRecommender
from src.pipeline import MovieLensRecommendationPipeline


RUNTIME_ARTIFACT_VERSION = 1
DEFAULT_ARTIFACT_PATH = ARTIFACTS_DIR / "movielens_runtime.joblib"


def get_artifact_path() -> Path:
    """Return the configured runtime artifact path."""
    configured = os.environ.get("MOVIELENS_ARTIFACT_PATH", "").strip()
    return Path(configured) if configured else DEFAULT_ARTIFACT_PATH


def load_runtime_artifact(path: str | Path | None = None) -> Dict[str, Any]:
    """Load a pre-built runtime bundle."""
    artifact_path = Path(path) if path else get_artifact_path()
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Runtime artifact not found: {artifact_path}. "
            "Run scripts/build_artifacts.py first."
        )

    bundle = joblib.load(artifact_path)
    if isinstance(bundle, MovieLensRecommendationPipeline):
        return {"version": 0, "pipeline": bundle, "movies_df": pd.DataFrame()}

    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(
            f"Invalid runtime artifact format: {artifact_path}. "
            "Expected a dictionary containing a fitted pipeline."
        )

    return bundle

def build_cloud_demo_runtime() -> Dict[str, Any]:
    """Build a tiny deterministic fallback runtime when no artifact is bundled.

    This path exists for hosted demos such as Streamlit Community Cloud. It
    never attempts to train the full MovieLens 32M dataset. Production/API
    deployments remain artifact-first.
    """
    loader = MovieLensDataLoader()
    movies = loader.load_movies()
    ratings = loader.load_ratings(max_rows=5_000)
    tags = loader.load_tags()
    links = loader.load_links()

    pipeline = MovieLensRecommendationPipeline(
        n_svd_components=8,
        min_popularity_m=3,
        random_state=42,
    )
    pipeline.fit(ratings_df=ratings, movies_df=movies, tags_df=tags)

    return prepare_runtime(
        {
            "version": RUNTIME_ARTIFACT_VERSION,
            "pipeline": pipeline,
            "movies_df": movies,
            "links_df": links,
            "baseline_ratings": ratings,
            "cloud_demo": True,
        }
    )


def _hydrate_popularity_stats(
    pipeline: MovieLensRecommendationPipeline,
    ratings_df: pd.DataFrame | None,
) -> None:
    """Backfill popularity statistics for artifacts built before v2.

    Older serialized artifacts only contain Bayesian scores. The UI can still
    use those artifacts safely by deriving rating counts and means from the
    bundled baseline ratings when available.
    """
    pop_model = pipeline.pop_model
    if hasattr(pop_model, "movie_counts") and hasattr(pop_model, "movie_means"):
        return

    counts: Dict[int, int] = {}
    means: Dict[int, float] = {}
    if (
        ratings_df is not None
        and not ratings_df.empty
        and {"movieId", "rating"}.issubset(ratings_df.columns)
    ):
        grouped = ratings_df.groupby("movieId")["rating"].agg(["count", "mean"])
        counts = {int(mid): int(row["count"]) for mid, row in grouped.iterrows()}
        means = {int(mid): float(row["mean"]) for mid, row in grouped.iterrows()}

    pop_model.movie_counts = counts
    pop_model.movie_means = means


def prepare_runtime(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Build lightweight lookup objects from an already-loaded bundle."""
    pipeline = bundle["pipeline"]
    movies_df = bundle.get("movies_df", pd.DataFrame()).copy()
    links_df = bundle.get("links_df", pd.DataFrame()).copy()
    baseline_ratings = bundle.get("baseline_ratings")
    _hydrate_popularity_stats(pipeline, baseline_ratings)

    if not isinstance(pipeline, MovieLensRecommendationPipeline):
        raise TypeError("Runtime artifact contains an unexpected pipeline type.")

    if movies_df.empty:
        movie_title_map: Dict[int, str] = {}
        movie_genre_map: Dict[int, str] = {}
        movie_lookup: Dict[int, Dict[str, Any]] = {}
        genre_prior = GenrePriorRecommender(pipeline.pop_model)
    else:
        movies_df["movieId"] = movies_df["movieId"].astype(int)
        if "year" not in movies_df.columns:
            movies_df["year"] = (
                movies_df["title"].astype(str).str.extract(r"\((\d{4})\)\s*$", expand=False)
            )
            movies_df["year"] = pd.to_numeric(movies_df["year"], errors="coerce").astype("Int64")
        movie_title_map = dict(zip(movies_df["movieId"].astype(int), movies_df["title"]))
        movie_genre_map = dict(zip(movies_df["movieId"].astype(int), movies_df["genres"]))
        movie_lookup = movies_df.set_index("movieId").to_dict(orient="index")
        genre_prior = GenrePriorRecommender(pipeline.pop_model).fit(movies_df)

    return {
        "movies_df": movies_df,
        "links_df": links_df,
        "movie_title_map": movie_title_map,
        "movie_genre_map": movie_genre_map,
        "movie_lookup": movie_lookup,
        "content_model": pipeline.content_model,
        "pop_model": pipeline.pop_model,
        "genre_prior": genre_prior,
        "svd_model": pipeline.svd_model,
        "hybrid_model": pipeline.hybrid_model,
        "pipeline": pipeline,
        "baseline_ratings": baseline_ratings,
    }
