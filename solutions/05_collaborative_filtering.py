"""
Section 5: Collaborative Filtering
==================================
Answers Questions 40 through 50 with CSR sparse matrix construction, sparsity calculations,
TruncatedSVD latent factor decomposition, Item-Item KNN, and unrated item filtering.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import MovieLensDataLoader
from src.collaborative import MatrixFactorizationSVD, ItemItemCollaborativeFiltering


def run_section_5():
    print("=" * 80)
    print("SECTION 5: COLLABORATIVE FILTERING")
    print("=" * 80)

    loader = MovieLensDataLoader()
    # Load representative active subset for quick demonstration
    print("Loading ratings data for Collaborative Filtering...")
    ratings_sample = loader.load_ratings(max_rows=1_000_000, min_user_ratings=10, min_movie_ratings=10)
    movies_df = loader.load_movies()
    movie_title_map = dict(zip(movies_df["movieId"], movies_df["title"]))

    # -------------------------------------------------------------------------
    # Q40 & Q41: Sparse Matrix Construction & Sparsity Calculation
    # -------------------------------------------------------------------------
    print("\n--- Q40 & Q41: Matrix Construction & Matrix Sparsity ---")
    matrix, u2i, i2u, m2i, i2m = loader.get_user_movie_sparse_matrix(ratings_sample)
    n_users, n_items = matrix.shape
    nnz = matrix.nnz
    total_possible_entries = n_users * n_items
    sparsity = (1.0 - (nnz / total_possible_entries)) * 100.0
    
    print(f"Rating Matrix Dimensions: {n_users:,} users x {n_items:,} movies")
    print(f"Non-Zero Entries (Ratings): {nnz:,}")
    print(f"Matrix Density : {nnz / total_possible_entries * 100:.4f}%")
    print(f"Matrix Sparsity: {sparsity:.4f}% (Extremely sparse!)")

    # -------------------------------------------------------------------------
    # Q42 & Q43: KNN User and Movie Similarity
    # -------------------------------------------------------------------------
    print("\n--- Q42 & Q43: Item-Item KNN Collaborative Filtering ---")
    item_knn = ItemItemCollaborativeFiltering(n_neighbors=10)
    item_knn.fit(matrix, m2i)
    
    query_mid = 1  # Toy Story
    if query_mid in m2i:
        item_recs = item_knn.recommend(item_id=query_mid, top_k=5)
        print(f"Item-Item Collaborative Neighbors for '{movie_title_map.get(query_mid, 'Toy Story')}':")
        for mid, sim in item_recs:
            print(f"  [{sim:.3f}] {movie_title_map.get(mid, 'Unknown')} (ID: {mid})")

    # -------------------------------------------------------------------------
    # Q44: Missing vs Explicit Low Ratings
    # -------------------------------------------------------------------------
    print("\n--- Q44: Missing vs Explicit Low Ratings (Implicit vs Explicit) ---")
    print("Crucial Distinction:")
    print("  - Explicit 0.5 rating: User actively disliked the movie.")
    print("  - Missing entry (0 in CSR): User has NOT yet discovered or interacted with the item.")
    print("  - Treatment: SVD on sparse matrix treats unrated as unobserved, predicting positive latent alignment.")

    # -------------------------------------------------------------------------
    # Q45, Q46, Q47: TruncatedSVD & Latent Component Selection
    # -------------------------------------------------------------------------
    print("\n--- Q45, Q46, Q47: TruncatedSVD Latent Factor Decomposition ---")
    for k in [16, 32, 64]:
        svd_model = MatrixFactorizationSVD(n_components=k, random_state=42)
        svd_model.fit(matrix, u2i, m2i, ratings_df=ratings_sample)
        print(f"Components k={k:2d} -> Cumulative Explained Variance: {svd_model.total_explained_variance_ * 100:.2f}%")

    # -------------------------------------------------------------------------
    # Q48 & Q49: User Rating Prediction & Unseen Recommendation
    # -------------------------------------------------------------------------
    print("\n--- Q48 & Q49: User Recommendations Excluding Already Seen Items ---")
    fitted_svd = MatrixFactorizationSVD(n_components=32, random_state=42)
    fitted_svd.fit(matrix, u2i, m2i, ratings_df=ratings_sample)
    
    target_user = list(u2i.keys())[0]
    user_recs = fitted_svd.recommend(user_id=target_user, top_k=5, exclude_seen=True)
    print(f"Personalized Latent-Factor Recommendations for User {target_user}:")
    for mid, score in user_recs:
        print(f"  [Predicted Score: {score:.2f}] {movie_title_map.get(mid, 'Unknown')} (ID: {mid})")
    print("=" * 80)


if __name__ == "__main__":
    run_section_5()
