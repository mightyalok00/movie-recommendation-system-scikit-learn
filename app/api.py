"""
FastAPI production recommendation service.

Startup is intentionally lightweight. Models are loaded lazily on the first
model request and should normally come from a pre-built runtime artifact.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.monitoring import ModelDriftMonitor
from data.loader import MovieLensDataLoader
from src.cold_start import ColdStartPopularityRecommender, GenrePriorRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative import MatrixFactorizationSVD
from src.hybrid import HybridRecommender
from src.runtime import get_artifact_path, load_runtime_artifact, prepare_runtime

app = FastAPI(
    title="MovieLens 32M Recommender API",
    description="Production recommendation service powered by Scikit-learn, SVD, and multi-modal content filtering.",
    version="1.1.0",
)

state = {}


def _train_runtime_for_local_development() -> dict:
    """Explicit local fallback when no artifact exists."""
    loader = MovieLensDataLoader()
    movies_df = loader.load_movies()
    content_meta = loader.load_full_content_metadata()
    ratings_df = loader.load_ratings(max_rows=500_000, min_user_ratings=10)

    content_model = ContentBasedRecommender().fit(content_meta)
    pop_model = ColdStartPopularityRecommender().fit(ratings_df)
    genre_prior = GenrePriorRecommender(pop_model).fit(movies_df)

    matrix, u2i, _, m2i, _ = loader.get_user_movie_sparse_matrix(ratings_df)
    n_comps = min(16, max(1, len(u2i) - 1), max(1, len(m2i) - 1))
    svd_model = MatrixFactorizationSVD(n_components=n_comps).fit(
        matrix, u2i, m2i, ratings_df=ratings_df
    )
    hybrid_model = HybridRecommender(
        collaborative_model=svd_model,
        content_model=content_model,
        popularity_model=pop_model,
    ).fit(movies_df=movies_df, ratings_df=ratings_df)

    return {
        "movies_df": movies_df,
        "movie_title_map": dict(zip(movies_df["movieId"], movies_df["title"])),
        "movie_genre_map": dict(zip(movies_df["movieId"], movies_df["genres"])),
        "content_model": content_model,
        "pop_model": pop_model,
        "genre_prior": genre_prior,
        "svd_model": svd_model,
        "hybrid_model": hybrid_model,
        "baseline_ratings": ratings_df["rating"].to_numpy(dtype=np.float32),
    }


def ensure_models_initialized():
    """Load a pre-built artifact or explicitly train for local development."""
    if state:
        return

    artifact_path = get_artifact_path()
    if artifact_path.exists():
        state.update(prepare_runtime(load_runtime_artifact(artifact_path)))
    elif os.environ.get("MOVIELENS_RUNTIME_MODE", "artifact").lower() == "train":
        state.update(_train_runtime_for_local_development())
    else:
        raise RuntimeError(
            f"Runtime artifact not found at {artifact_path}. "
            "Build it with scripts/build_artifacts.py, or set "
            "MOVIELENS_RUNTIME_MODE=train for local development."
        )

    baseline = state.get("baseline_ratings")
    if baseline is None or len(baseline) == 0:
        baseline = np.array([3.0, 3.5, 4.0, 4.5, 5.0], dtype=np.float32)
    state["monitor"] = ModelDriftMonitor(baseline_ratings=baseline)


@app.get("/health")
def health_check():
    """Fast liveness response without loading the ML stack."""
    artifact_exists = get_artifact_path().exists()
    return {
        "status": "healthy",
        "models_loaded": bool(state),
        "artifact_available": artifact_exists,
        "total_movies": len(state.get("movies_df", [])),
    }


class RecommendationItem(BaseModel):
    movieId: int
    title: str
    genres: str
    score: float


class RecommendationResponse(BaseModel):
    status: str
    user_id: Optional[int] = None
    userId: Optional[int] = None
    movie_id: Optional[int] = None
    movieId: Optional[int] = None
    recommendations: List[RecommendationItem]
    latency_ms: float


class UserRecommendationRequest(BaseModel):
    user_id: Optional[int] = Field(None)
    userId: Optional[int] = Field(None)
    top_k: int = Field(10, ge=1, le=50)
    use_hybrid: bool = Field(True)
    diversity_penalty: float = Field(0.15, ge=0.0, le=1.0)


class ItemRecommendationRequest(BaseModel):
    movie_id: Optional[int] = Field(None)
    movieId: Optional[int] = Field(None)
    top_k: int = Field(10, ge=1, le=50)
    content_engine: str = Field("unified")


class ColdStartRequest(BaseModel):
    preferred_genres: List[str] = Field(default_factory=list)
    top_k: int = Field(10, ge=1, le=50)


class DriftMonitoringRequest(BaseModel):
    recent_ratings: List[float] = Field(...)


@app.get("/ready")
def readiness_check():
    """Report artifact readiness without loading it."""
    artifact_path = get_artifact_path()
    if state:
        return {"status": "ready", "models_loaded": True}
    if artifact_path.exists():
        return {"status": "ready", "models_loaded": False, "artifact": str(artifact_path)}
    return {"status": "not_ready", "models_loaded": False, "artifact": str(artifact_path)}


@app.get("/movies/search")
def search_movies(q: str = Query(..., min_length=1)):
    ensure_models_initialized()
    movies_df = state["movies_df"]
    matches = movies_df[movies_df["title"].str.contains(q, case=False, na=False)].head(10)
    return matches[["movieId", "title", "genres"]].to_dict(orient="records")


def _initialize_or_503():
    try:
        ensure_models_initialized()
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/recommend/item/{movie_id}", response_model=RecommendationResponse)
def get_recommend_item(movie_id: int, top_k: int = Query(10, ge=1, le=50)):
    return process_item_recommendation(movie_id, top_k)


@app.post("/recommend/item", response_model=RecommendationResponse)
def post_recommend_item(req: ItemRecommendationRequest):
    mid = req.movie_id or req.movieId or 1
    return process_item_recommendation(mid, req.top_k)


def process_item_recommendation(movie_id: int, top_k: int) -> RecommendationResponse:
    _initialize_or_503()
    t0 = time.perf_counter()
    recs = state["content_model"].recommend(item_id=movie_id, top_k=top_k)
    latency = (time.perf_counter() - t0) * 1000.0
    state["monitor"].record_latency(latency)
    items = [
        RecommendationItem(
            movieId=mid,
            title=state["movie_title_map"].get(mid, "Unknown"),
            genres=state["movie_genre_map"].get(mid, ""),
            score=score,
        )
        for mid, score in recs
    ]
    return RecommendationResponse(
        status="SUCCESS",
        movie_id=movie_id,
        movieId=movie_id,
        recommendations=items,
        latency_ms=latency,
    )


@app.post("/recommend/user", response_model=RecommendationResponse)
def recommend_user(req: UserRecommendationRequest):
    _initialize_or_503()
    t0 = time.perf_counter()
    uid = req.user_id if req.user_id is not None else (req.userId if req.userId is not None else 1)
    model = state["hybrid_model"] if req.use_hybrid else state["svd_model"]
    recs = model.recommend(user_id=uid, top_k=req.top_k, exclude_seen=True)
    latency = (time.perf_counter() - t0) * 1000.0
    state["monitor"].record_latency(latency)
    items = [
        RecommendationItem(
            movieId=mid,
            title=state["movie_title_map"].get(mid, "Unknown"),
            genres=state["movie_genre_map"].get(mid, ""),
            score=score,
        )
        for mid, score in recs
    ]
    return RecommendationResponse(
        status="SUCCESS",
        user_id=uid,
        userId=uid,
        recommendations=items,
        latency_ms=latency,
    )


@app.post("/recommend/cold-start", response_model=RecommendationResponse)
def recommend_cold_start(req: ColdStartRequest):
    _initialize_or_503()
    t0 = time.perf_counter()
    recs = state["genre_prior"].recommend(preferred_genres=req.preferred_genres, top_k=req.top_k)
    latency = (time.perf_counter() - t0) * 1000.0
    state["monitor"].record_latency(latency)
    items = [
        RecommendationItem(
            movieId=mid,
            title=state["movie_title_map"].get(mid, "Unknown"),
            genres=state["movie_genre_map"].get(mid, ""),
            score=score,
        )
        for mid, score in recs
    ]
    return RecommendationResponse(status="SUCCESS", recommendations=items, latency_ms=latency)


@app.post("/monitoring/drift")
def check_drift(req: DriftMonitoringRequest):
    _initialize_or_503()
    result = state["monitor"].check_drift(new_ratings=req.recent_ratings)
    return {
        "status": "SUCCESS",
        "drift_detected": result["drift_detected"],
        "ks_statistic": result["ks_statistic"],
        "p_value": result["p_value"],
        "interpretation": result["interpretation"],
    }
