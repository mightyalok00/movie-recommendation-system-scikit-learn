"""
Build the production recommendation runtime artifact.

This is an OFFLINE training/build command. It must not be called by a web
application during startup.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np

from config.settings import ARTIFACTS_DIR
from data.loader import MovieLensDataLoader
from src.pipeline import MovieLensRecommendationPipeline


def build_artifact(
    output: Path,
    max_ratings: int | None = None,
    n_svd_components: int = 16,
    baseline_sample_size: int = 100_000,
) -> Path:
    """Train the recommender offline and write an atomic runtime bundle."""
    started = time.perf_counter()
    loader = MovieLensDataLoader()

    movies_df = loader.load_movies()
    tags_df = loader.load_tags(min_tag_freq=2)
    ratings_df = loader.load_ratings(max_rows=max_ratings, min_user_ratings=1)
    links_df = loader.load_links()

    pipeline = MovieLensRecommendationPipeline(
        n_svd_components=n_svd_components,
        random_state=42,
    )
    pipeline.fit(ratings_df=ratings_df, movies_df=movies_df, tags_df=tags_df)

    ratings = ratings_df["rating"].to_numpy(dtype=np.float32, copy=True)
    if len(ratings) > baseline_sample_size:
        rng = np.random.default_rng(42)
        ratings = rng.choice(ratings, size=baseline_sample_size, replace=False)

    bundle = {
        "version": 1,
        "pipeline": pipeline,
        "movies_df": movies_df[["movieId", "title", "genres"]].copy(),
        "links_df": links_df.copy(),
        "baseline_ratings": ratings,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_suffix(output.suffix + ".tmp")
    joblib.dump(bundle, temp_path, compress=3)
    temp_path.replace(output)

    elapsed = time.perf_counter() - started
    print(f"Runtime artifact written to: {output}")
    print(f"Movies: {len(movies_df):,}")
    print(f"Ratings used for training: {len(ratings_df):,}")
    print(f"Build time: {elapsed:.2f}s")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the offline MovieLens runtime artifact.")
    parser.add_argument("--output", type=Path, default=ARTIFACTS_DIR / "movielens_runtime.joblib")
    parser.add_argument("--max-ratings", type=int, default=None)
    parser.add_argument("--svd-components", type=int, default=16)
    args = parser.parse_args()

    build_artifact(
        output=args.output,
        max_ratings=args.max_ratings,
        n_svd_components=args.svd_components,
    )


if __name__ == "__main__":
    main()
