"""
MovieLens 32M Benchmark Reproduction Script
===========================================
Executes full temporal train/test split offline evaluation comparing:
1. Global Popularity Baseline
2. TruncatedSVD Matrix Factorization
3. Multi-Modal Content Recommender
4. Weighted Hybrid Recommender with MMR Diversity

Outputs metrics to reports/benchmark_results.csv.
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import MovieLensDataLoader
from src.cold_start import ColdStartPopularityRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative import MatrixFactorizationSVD
from src.hybrid import HybridRecommender
from src.evaluation import RecommendationEvaluator, temporal_train_test_split
from config.settings import REPORTS_DIR


def run_benchmark_reproduction(max_ratings: int = 100_000):
    print("=" * 70)
    print(">>> MovieLens Offline Benchmark Reproduction Suite")
    print("=" * 70)
    
    t0 = time.time()
    loader = MovieLensDataLoader()
    print(">>> [1/5] Loading MovieLens Metadata & Interactions...")
    movies_df = loader.load_movies()
    ratings_df = loader.load_ratings(max_rows=max_ratings)
    tags_df = loader.load_tags()
    
    print(f"    Loaded {len(movies_df):,} movies, {len(ratings_df):,} ratings.")
    
    print(">>> [2/5] Partitioning Temporal Train/Test Sets (Hold-Out Last 20% Interactions Per User)...")
    train_df, test_df = temporal_train_test_split(ratings_df, test_ratio=0.20)
    print(f"    Train size: {len(train_df):,} | Test size: {len(test_df):,}")
    
    print(">>> [3/5] Fitting Models...")
    # 1. Popularity Baseline
    pop_model = ColdStartPopularityRecommender().fit(train_df)
    
    # 2. Content Model
    content_meta = movies_df.copy()
    content_meta["combined_tags"] = content_meta["genres"].fillna("") + " " + content_meta["clean_title"].fillna("")
    content_model = ContentBasedRecommender().fit(content_meta)
    
    # 3. SVD Collaborative Filtering
    matrix, u2i, _, m2i, _ = loader.get_user_movie_sparse_matrix(train_df)
    n_comps = min(32, len(u2i) - 1, len(m2i) - 1)
    svd_model = MatrixFactorizationSVD(n_components=n_comps, random_state=42).fit(matrix, u2i, m2i, ratings_df=train_df)
    
    # 4. Weighted Hybrid Model
    hybrid_model = HybridRecommender(
        collab_model=svd_model,
        content_model=content_model,
        popularity_model=pop_model,
        collab_weight=0.60,
        content_weight=0.30,
        popularity_weight=0.10,
        diversity_penalty=0.15
    ).fit(movies_df, train_df)
    
    print(">>> [4/5] Evaluating Offline Recommendation Metrics...")
    pop_dict = train_df["movieId"].value_counts().to_dict()
    evaluator = RecommendationEvaluator(test_df=test_df, relevance_threshold=3.5, popularity_dict=pop_dict)
    
    models = {
        "Popularity Baseline": pop_model,
        "SVD Matrix Factorization": svd_model,
        "Weighted Hybrid (Proposed)": hybrid_model
    }
    
    eval_results = []
    
    for name, model in models.items():
        print(f"    Evaluating {name}...")
        metrics = evaluator.evaluate_model(model=model, sample_users=100, top_k_list=[5, 10, 20])
        metrics["Model Architecture"] = name
        eval_results.append(metrics)
        
    df_results = pd.DataFrame(eval_results)
    cols = ["Model Architecture"] + [c for c in df_results.columns if c != "Model Architecture"]
    df_results = df_results[cols]
    
    output_path = REPORTS_DIR / "benchmark_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n>>> [5/5] Benchmark results saved to: {output_path}")
    print("\n" + df_results.to_string(index=False))
    print(f"\nTotal elapsed time: {time.time()-t0:.2f} seconds.")


if __name__ == "__main__":
    run_benchmark_reproduction()
