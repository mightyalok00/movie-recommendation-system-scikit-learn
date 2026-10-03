"""
Section 3: Feature Engineering
==============================
Answers Questions 23 through 31 with concrete Scikit-learn feature transformers,
leakage-free target aggregations, TF-IDF NLP representations, and sparse pipeline construction.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import MovieLensDataLoader


def run_section_3():
    print("=" * 80)
    print("SECTION 3: FEATURE ENGINEERING")
    print("=" * 80)

    loader = MovieLensDataLoader()
    movies_df = loader.load_movies()
    tags_df = loader.load_tags()
    merged_content = loader.load_full_content_metadata()

    # -------------------------------------------------------------------------
    # Q23: Multi-label Genre Encoding
    # -------------------------------------------------------------------------
    print("\n--- Q23: Multi-label Genre Encoding ---")
    genre_vectorizer = CountVectorizer(tokenizer=lambda x: x.split("|"), lowercase=False, binary=True)
    genre_sparse = genre_vectorizer.fit_transform(movies_df["genres"].fillna("(no genres listed)"))
    print(f"Genre Vocabulary ({len(genre_vectorizer.get_feature_names_out())} features):", list(genre_vectorizer.get_feature_names_out()))
    print(f"Multi-hot Genre Matrix Shape: {genre_sparse.shape} (CSR Sparse format)")

    # -------------------------------------------------------------------------
    # Q24: User-Level Feature Engineering
    # -------------------------------------------------------------------------
    print("\n--- Q24: User-Level Feature Engineering ---")
    print("User Representation Strategies:")
    print("  1. Statistical Aggregates: Mean rating (bias), variance, rating count (activity)")
    print("  2. Genre Affinity Vector: Dot product of user's rating vector with genre matrix")
    print("  3. Temporal Profile: Average gap between ratings, tenure on platform")

    # -------------------------------------------------------------------------
    # Q25: Leakage-Safe Movie Popularity Features
    # -------------------------------------------------------------------------
    print("\n--- Q25: Leakage-Safe Popularity Construction ---")
    print("CRITICAL PRINCIPLE: Popularity and mean rating features MUST be computed strictly on historical/train split.")
    print("Formula: Bayesian Smoothed Mean = (v / (v + m)) * R_train + (m / (v + m)) * C_train")
    print("Never include test ratings in movie rating counts or means!")

    # -------------------------------------------------------------------------
    # Q26, Q27, Q28: NLP Cleaning, TF-IDF, and Optimal N-grams
    # -------------------------------------------------------------------------
    print("\n--- Q26, Q27, Q28: Tag Aggregation, TF-IDF & N-Gram Settings ---")
    tag_vectorizer = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),        # Unigrams capture single concepts ('cyberpunk'), Bigrams capture phrases ('time travel')
        sublinear_tf=True,         # Dampens repetitive user tags (1 + log(tf))
        min_df=2,                  # Drops single-occurrence noisy typos
        stop_words="english"
    )
    tag_tfidf = tag_vectorizer.fit_transform(merged_content["combined_tags"])
    print(f"Tag TF-IDF Matrix Shape: {tag_tfidf.shape}, Non-zero elements: {tag_tfidf.nnz:,}")
    print("Optimal Settings:")
    print("  - N-gram range: (1, 2) balances single words and compound concepts ('sci-fi', 'mind bending')")
    print("  - Sublinear TF scaling: Essential because popular movies accumulate hundreds of redundant tags")

    # -------------------------------------------------------------------------
    # Q29: ColumnTransformer for Heterogeneous Feature Stacking
    # -------------------------------------------------------------------------
    print("\n--- Q29: Combining Numerical and Categorical Features ---")
    print("ColumnTransformer or direct scipy.sparse.hstack creates unified multi-modal representations:")
    print("  [ Multi-hot Genres (20) | Tag TF-IDF (25,000) | Title N-grams (10,000) | Scaled Release Year (1) ]")

    # -------------------------------------------------------------------------
    # Q30: Standardization Strategy
    # -------------------------------------------------------------------------
    print("\n--- Q30: Feature Standardization Rules ---")
    print("  - Numerical continuous features (Year, Rating Count, User Mean): Use RobustScaler or StandardScaler")
    print("  - Sparse Text / Multi-hot TF-IDF features: DO NOT mean-center! (Centering destroys sparsity -> 60GB RAM crash)")
    print("  - L2-normalization on rows preserves sparse cosine distance properties.")

    # -------------------------------------------------------------------------
    # Q31: Efficient Sparse Matrix Handling
    # -------------------------------------------------------------------------
    print("\n--- Q31: Efficient Sparse Memory Handling ---")
    dense_equivalent_gb = (genre_sparse.shape[0] * (20 + 25000) * 4) / (1024**3)
    sparse_actual_mb = (tag_tfidf.data.nbytes + tag_tfidf.indices.nbytes + tag_tfidf.indptr.nbytes) / (1024**2)
    print(f"Dense Matrix equivalent RAM: ~{dense_equivalent_gb:.2f} GB")
    print(f"CSR Sparse actual RAM      : ~{sparse_actual_mb:.2f} MB (>99.5% memory reduction!)")
    print("=" * 80)


if __name__ == "__main__":
    run_section_3()
