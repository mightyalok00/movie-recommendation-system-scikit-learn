"""
FastAPI Production Recommendation REST Service
=============================================
Provides asynchronous endpoints for item-to-item similarity, personalized user ranking,
onboarding cold-start recommendations, and drift telemetry.
"""

import time
import sys
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import MovieLensDataLoader
from src.content_based import ContentBasedRecommender
from src.cold_start import ColdStartPopularityRecommender, GenrePriorRecommender
from src.collaborative import MatrixFactorizationSVD
from src.hybrid import HybridRecommender
from app.monitoring import ModelDriftMonitor

# Initialize FastAPI App
app = FastAPI(
    title="MovieLens 32M Recommender API",
    description="High-throughput production recommendation service powered by Scikit-learn, SVD, and Content TF-IDF",
    version="1.0.0"
)

# Global model state
state = {}


class RecommendationItem(BaseModel):
    movieId: int
    title: str
    genres: str
    score: float


class RecommendationResponse(BaseModel):
    status: str
    recommendations: List[RecommendationItem]
    latency_ms: float


class UserRecommendationRequest(BaseModel):
    userId: int = Field(..., description="Target user ID")
    top_k: int = Field(10, ge=1, le=50)
    use_hybrid: bool = Field(True, description="Whether to use Hybrid model or pure SVD")


class ColdStartRequest(BaseModel):
    preferred_genres: List[str] = Field(default_factory=list, description="List of preferred genres (e.g. ['Sci-Fi', 'Action'])")
    top_k: int = Field(10, ge=1, le=50)


@app.on_event("startup")
def startup_event():
    """Load metadata and fit models on startup."""
    loader = MovieLensDataLoader()
    movies_df = loader.load_movies()
    content_meta = loader.load_full_content_metadata()
    ratings_df = loader.load_ratings(max_rows=500_000, min_user_ratings=10)

    content_model = ContentBasedRecommender().fit(content_meta)
    pop_model = ColdStartPopularityRecommender().fit(ratings_df)
    genre_prior = GenrePriorRecommender(pop_model).fit(movies_df)

    matrix, u2i, _, m2i, _ = loader.get_user_movie_sparse_matrix(ratings_df)
    svd_model = MatrixFactorizationSVD(n_components=32).fit(matrix, u2i, m2i, ratings_df=ratings_df)

    hybrid_model = HybridRecommender(
        collaborative_model=svd_model,
        content_model=content_model,
        popularity_model=pop_model
    ).fit(movies_df=movies_df, ratings_df=ratings_df)

    movie_title_map = dict(zip(movies_df["movieId"], movies_df["title"]))
    movie_genre_map = dict(zip(movies_df["movieId"], movies_df["genres"]))

    monitor = ModelDriftMonitor(baseline_ratings=ratings_df["rating"].to_numpy())

    state["movies_df"] = movies_df
    state["movie_title_map"] = movie_title_map
    state["movie_genre_map"] = movie_genre_map
    state["content_model"] = content_model
    state["pop_model"] = pop_model
    state["genre_prior"] = genre_prior
    state["svd_model"] = svd_model
    state["hybrid_model"] = hybrid_model
    state["monitor"] = monitor


@app.get("/health")
def health_check():
    """Service health and uptime endpoint."""
    return {"status": "HEALTHY", "models_loaded": "hybrid_model" in state}


@app.get("/movies/search")
def search_movies(q: str = Query(..., min_length=1, description="Movie search query")):
    """Search movies catalog by title substring."""
    movies_df = state["movies_df"]
    matches = movies_df[movies_df["title"].str.contains(q, case=False, na=False)].head(10)
    return matches[["movieId", "title", "genres"]].to_dict(orient="records")


@app.get("/recommend/item/{movie_id}", response_model=RecommendationResponse)
def recommend_item(movie_id: int, top_k: int = Query(10, ge=1, le=50)):
    """Item-to-item Content recommendation."""
    t0 = time.perf_counter()
    content_model = state["content_model"]
    recs = content_model.recommend(item_id=movie_id, top_k=top_k)
    
    latency = (time.perf_counter() - t0) * 1000.0
    state["monitor"].record_latency(latency)

    items = [
        RecommendationItem(
            movieId=mid,
            title=state["movie_title_map"].get(mid, "Unknown"),
            genres=state["movie_genre_map"].get(mid, ""),
            score=score
        )
        for mid, score in recs
    ]
    return RecommendationResponse(status="SUCCESS", recommendations=items, latency_ms=latency)


@app.post("/recommend/user", response_model=RecommendationResponse)
def recommend_user(req: UserRecommendationRequest):
    """Personalized user recommendation via Hybrid or SVD."""
    t0 = time.perf_counter()
    model = state["hybrid_model"] if req.use_hybrid else state["svd_model"]
    recs = model.recommend(user_id=req.userId, top_k=req.top_k, exclude_seen=True)
    
    latency = (time.perf_counter() - t0) * 1000.0
    state["monitor"].record_latency(latency)

    items = [
        RecommendationItem(
            movieId=mid,
            title=state["movie_title_map"].get(mid, "Unknown"),
            genres=state["movie_genre_map"].get(mid, ""),
            score=score
        )
        for mid, score in recs
    ]
    return RecommendationResponse(status="SUCCESS", recommendations=items, latency_ms=latency)


@app.post("/recommend/cold-start", response_model=RecommendationResponse)
def recommend_cold_start(req: ColdStartRequest):
    """Onboarding cold-start recommendation for new users."""
    t0 = time.perf_counter()
    genre_prior = state["genre_prior"]
    recs = genre_prior.recommend(preferred_genres=req.preferred_genres, top_k=req.top_k)
    
    latency = (time.perf_counter() - t0) * 1000.0
    state["monitor"].record_latency(latency)

    items = [
        RecommendationItem(
            movieId=mid,
            title=state["movie_title_map"].get(mid, "Unknown"),
            genres=state["movie_genre_map"].get(mid, ""),
            score=score
        )
        for mid, score in recs
    ]
    return RecommendationResponse(status="SUCCESS", recommendations=items, latency_ms=latency)
