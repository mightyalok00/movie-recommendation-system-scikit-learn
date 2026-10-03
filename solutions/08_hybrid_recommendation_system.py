"""
Section 8: Hybrid Recommendation System
=======================================
Answers Questions 64 through 71 with score normalization, dynamic weighting,
popularity dampening, genre diversity re-ranking, and cold-start fallback routing.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import MovieLensDataLoader
from src.content_based import ContentBasedRecommender
from src.collaborative import MatrixFactorizationSVD
from src.cold_start import ColdStartPopularityRecommender
from src.hybrid import HybridRecommender


def run_section_8():
    print("=" * 80)
    print("SECTION 8: HYBRID RECOMMENDATION SYSTEM")
    print("=" * 80)

    loader = MovieLensDataLoader()
    ratings_df = loader.load_ratings(max_rows=500_000, min_user_ratings=10)
    movies_df = loader.load_movies()
    content_meta = loader.load_full_content_metadata()
    movie_title_map = dict(zip(movies_df["movieId"], movies_df["title"]))

    # 1. Fit Content Model
    print("Fitting Content-Based model...")
    content_model = ContentBasedRecommender()
    content_model.fit(content_meta)

    # 2. Fit Collaborative Model
    print("Fitting SVD Collaborative model...")
    matrix, u2i, _, m2i, _ = loader.get_user_movie_sparse_matrix(ratings_df)
    svd_model = MatrixFactorizationSVD(n_components=32, random_state=42)
    svd_model.fit(matrix, u2i, m2i, ratings_df=ratings_df)

    # 3. Fit Bayesian Popularity Model
    print("Fitting Bayesian Popularity baseline...")
    pop_model = ColdStartPopularityRecommender(min_ratings_m=50)
    pop_model.fit(ratings_df)

    # 4. Construct Weighted Hybrid System
    print("Assembling Weighted Hybrid Recommender with Genre Diversity MMR...")
    hybrid_model = HybridRecommender(
        collaborative_model=svd_model,
        content_model=content_model,
        popularity_model=pop_model,
        collab_weight=0.60,
        content_weight=0.30,
        popularity_weight=0.10,
        diversity_penalty=0.15
    )
    hybrid_model.fit(movies_df=movies_df, ratings_df=ratings_df)

    # -------------------------------------------------------------------------
    # Q79, Q80, Q81: Combination, Normalization & Weighting Strategies
    # -------------------------------------------------------------------------
    print("\n--- Q79, Q80, Q81: Score Normalization & Weighting ---")
    test_user_id = list(u2i.keys())[5]
    
    print(f"Generating recommendations for User ID {test_user_id}:")
    
    # Generate pure collaborative
    collab_recs = svd_model.recommend(user_id=test_user_id, top_k=3, exclude_seen=True)
    print("\nTop 3 Collaborative Recommender:")
    for mid, score in collab_recs:
        print(f"  [SVD: {score:.2f}] {movie_title_map.get(mid, 'Unknown')}")

    # Generate hybrid
    hybrid_recs = hybrid_model.recommend(user_id=test_user_id, top_k=5, exclude_seen=True)
    print("\nTop 5 Hybrid Recommender (Collab 60% + Content 30% + Pop 10% + MMR Diversity):")
    for mid, score in hybrid_recs:
        print(f"  [Hybrid Score: {score:.3f}] {movie_title_map.get(mid, 'Unknown')}")

    # -------------------------------------------------------------------------
    # Q82, Q83, Q84: Popularity De-biasing, Diversity & Seen Filtering
    # -------------------------------------------------------------------------
    print("\n--- Q82, Q83, Q84: De-biasing, Genre Diversity MMR & Seen Filtering ---")
    print("  - Seen Filtering: User interaction sets are tracked in hash sets for O(1) exclusion.")
    print("  - Diversity Re-ranking: Genre overlap penalty discounts duplicate genre clusters.")
    print("  - Popularity Dampening: Prevents top 10 blockbusters from monopolizing recommendation slots.")

    # -------------------------------------------------------------------------
    # Q85 & Q86: Partial User History & Unrated Item Cold Start
    # -------------------------------------------------------------------------
    print("\n--- Q85 & Q86: Edge Cases & Cold Start Handling ---")
    print("Cold User Test (User ID = -999, Zero Ratings):")
    cold_recs = hybrid_model.recommend(user_id=-999, top_k=3)
    for mid, score in cold_recs:
        print(f"  [Bayesian Pop: {score:.2f}] {movie_title_map.get(mid, 'Unknown')}")
    print("Graceful Fallback: The hybrid engine automatically detects zero user history and falls back to Bayesian Popularity.")
    print("=" * 80)


if __name__ == "__main__":
    run_section_8()
