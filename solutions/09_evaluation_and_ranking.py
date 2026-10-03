"""
Section 9: Evaluation & Ranking
===============================
Answers Questions 72 through 83 with temporal train/test partitioning,
Precision@K, Recall@K, MAP@K, NDCG@K, Catalog Coverage, Novelty, Diversity, and multi-model benchmarking.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import MovieLensDataLoader
from src.cold_start import ColdStartPopularityRecommender
from src.collaborative import MatrixFactorizationSVD
from src.content_based import ContentBasedRecommender
from src.hybrid import HybridRecommender
from src.evaluation import RecommendationEvaluator, temporal_train_test_split


def run_section_9():
    print("=" * 80)
    print("SECTION 9: EVALUATION & RANKING METRICS")
    print("=" * 80)

    loader = MovieLensDataLoader()
    ratings_df = loader.load_ratings(max_rows=500_000, min_user_ratings=15)
    movies_df = loader.load_movies()
    content_meta = loader.load_full_content_metadata()

    # -------------------------------------------------------------------------
    # Q88 & Q89: Realistic Temporal Train/Test Split vs Random Row Splitting
    # -------------------------------------------------------------------------
    print("\n--- Q88 & Q89: Temporal Split vs Random Splitting ---")
    print("Why Random Split Fails:")
    print("  - Random splitting introduces future-to-past data leakage (a user's later ratings leak tastes into training).")
    print("  - Real-world production requires predicting future interactions from past actions.")
    
    train_df, test_df = temporal_train_test_split(ratings_df, test_ratio=0.2, by_user=True)
    print(f"Train Set: {len(train_df):,d} interactions | Test Set: {len(test_df):,d} interactions")

    # Build popularity dictionary and genre vectors for diversity evaluation
    pop_dict = train_df["movieId"].value_counts().to_dict()
    
    cv = CountVectorizer(tokenizer=lambda x: x.split("|"), binary=True)
    g_mat = cv.fit_transform(movies_df["genres"].fillna("")).toarray()
    movie_genre_vecs = {mid: g_mat[idx] for idx, mid in enumerate(movies_df["movieId"])}

    # Initialize Evaluator
    evaluator = RecommendationEvaluator(
        test_df=test_df,
        relevance_threshold=4.0,
        total_catalog_size=len(movies_df),
        popularity_dict=pop_dict
    )

    # -------------------------------------------------------------------------
    # Fit Models on Train Partition
    # -------------------------------------------------------------------------
    print("\nTraining models on Train partition...")
    # 1. Popularity Baseline
    pop_model = ColdStartPopularityRecommender(min_ratings_m=50).fit(train_df)
    
    # 2. Content Model
    content_model = ContentBasedRecommender().fit(content_meta)
    
    # 3. SVD Collaborative Model
    matrix, u2i, _, m2i, _ = loader.get_user_movie_sparse_matrix(train_df)
    svd_model = MatrixFactorizationSVD(n_components=32, random_state=42).fit(matrix, u2i, m2i, ratings_df=train_df)
    
    # 4. Hybrid Model
    hybrid_model = HybridRecommender(
        collaborative_model=svd_model,
        content_model=content_model,
        popularity_model=pop_model,
        collab_weight=0.60,
        content_weight=0.30,
        popularity_weight=0.10,
        diversity_penalty=0.15
    ).fit(movies_df=movies_df, ratings_df=train_df)

    # -------------------------------------------------------------------------
    # Q90 - Q99: Multi-Model Offline Benchmark Table
    # -------------------------------------------------------------------------
    print("\n--- Q90 - Q99: Multi-Model Offline Benchmark Results ---")
    models_to_test = {
        "Popularity Baseline": pop_model,
        "MatrixFactorization (SVD)": svd_model,
        "Weighted Hybrid (Proposed)": hybrid_model,
    }

    benchmark_rows = []
    for m_name, model in models_to_test.items():
        print(f"Evaluating {m_name}...")
        res = evaluator.evaluate_model(
            model=model,
            sample_users=200,
            top_k_list=[5, 10, 20],
            genre_vectors=movie_genre_vecs
        )
        res["Model"] = m_name
        benchmark_rows.append(res)

    benchmark_df = pd.DataFrame(benchmark_rows).set_index("Model")
    
    # Re-order columns nicely
    cols_order = [
        "Precision@5", "Precision@10", "Precision@20",
        "Recall@5", "Recall@10", "Recall@20",
        "NDCG@10", "MAP@10", "Coverage@10", "Novelty@10", "IntraListDiversity@10"
    ]
    benchmark_df = benchmark_df[[c for c in cols_order if c in benchmark_df.columns]]
    print("\n" + benchmark_df.to_string(float_format=lambda x: f"{x:.4f}"))

    # Save benchmark table to reports directory
    reports_path = Path(__file__).resolve().parent.parent / "reports" / "benchmark_results.csv"
    benchmark_df.to_csv(reports_path)
    print(f"\nSaved benchmark table to {reports_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_section_9()
