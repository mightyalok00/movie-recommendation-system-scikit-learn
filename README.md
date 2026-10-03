# 🎬 MovieLens 32M Movie Recommendation System
### Production-Grade Recommendation Engine powered by Scikit-learn, TruncatedSVD & Multi-Modal Content Filtering

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Highlights & Architecture

- **Dataset Scale:** Engineered for **MovieLens 32M** (32,000,204 ratings, 87,585 movies, 200,948 users, 2,000,072 tags).
- **Multi-Modal Content Engine:** Fuses multi-hot genre binarization, TF-IDF tag representations with sublinear TF scaling, and title n-grams with exact `NearestNeighbors(metric='cosine')` indexing.
- **Collaborative Filtering:** TruncatedSVD Latent Factor Matrix Factorization on CSR sparse matrices with $101.2\times$ latent compression.
- **Weighted Hybrid Orchestrator:** Dynamic score fusion ($\alpha=0.60\text{ Collab} + \beta=0.30\text{ Content} + \gamma=0.10\text{ Bayesian Pop}$) with **Genre Diversity Maximal Marginal Relevance (MMR)** re-ranking.
- **Cold-Start Resilience:** Seamless onboarding for brand-new users via interactive genre elicitation and Bayesian-smoothed IMDB weighted ratings.
- **Production Deployment:**
  - 🚀 **Streamlit Interactive Dashboard** for instant movie search, hybrid recommendations, and diagnostic visual analytics.
  - ⚡ **FastAPI REST Service** with Pydantic schemas, sub-100ms P50 latency, and Kolmogorov-Smirnov drift monitoring.
  - 📦 **Scikit-Learn Reusable Estimator Pipeline** (`MovieLensRecommendationPipeline`) with `.fit()`, `.predict()`, `.recommend()`, and `.joblib` serialization.

---

## 📁 Repository Structure

```
movie_recommendation_system/
├── README.md                                  # Master project documentation
├── requirements.txt                           # Project dependencies
├── config.py                                  # Dynamic path & memory configuration
├── data_loader.py                             # Memory-optimized data loader & CSR matrix generator
├── run_all.py                                 # Automated master runner across all 11 sections
│
├── solutions/                                 # Dedicated solution scripts (Questions 1 to 113)
│   ├── 01_dataset_understanding_validation.py # Section 1: Q1 - Q10
│   ├── 02_exploratory_data_analysis.py        # Section 2: Q11 - Q22
│   ├── 03_feature_engineering.py              # Section 3: Q23 - Q31
│   ├── 04_content_based_recommendation.py     # Section 4: Q32 - Q39
│   ├── 05_collaborative_filtering.py          # Section 5: Q40 - Q50
│   ├── 06_preference_prediction_supervised.py # Section 6: Q51 - Q58
│   ├── 07_dimensionality_reduction.py         # Section 7: Q59 - Q65
│   ├── 08_hybrid_recommendation_system.py     # Section 8: Q66 - Q74 (Q79-Q86)
│   ├── 09_evaluation_and_ranking.py           # Section 9: Q75 - Q87 (Q88-Q99)
│   ├── 10_cold_start_and_production.py        # Section 10: Q88 - Q100 (Q100-Q112)
│   └── 11_advanced_challenges.py              # Section 11: Q101 - Q113 (Q114-Q123)
│
├── src/                                       # Reusable Core ML Package (Scikit-learn Compatible)
│   ├── __init__.py
│   ├── base.py                                # BaseRecommender ABC interface
│   ├── content_based.py                       # Genre, Tag TF-IDF & Unified Content Recommenders
│   ├── collaborative.py                       # TruncatedSVD Matrix Factorization & KNN CF
│   ├── supervised.py                          # HistGradientBoosting & Logistic Preference Classifiers
│   ├── cold_start.py                          # Bayesian Popularity & Genre-Prior Onboarding
│   ├── hybrid.py                              # Weighted Hybrid Recommender with MMR Diversity
│   ├── evaluation.py                          # Temporal Split, Precision@K, Recall@K, NDCG@K, Diversity
│   └── pipeline.py                            # Scikit-learn Pipeline estimator wrapper
│
├── app/                                       # Web Application & Production API
│   ├── __init__.py
│   ├── streamlit_app.py                       # Interactive Streamlit Web Application
│   ├── api.py                                 # FastAPI RESTful Microservice
│   └── monitoring.py                          # Kolmogorov-Smirnov rating drift telemetry
│
├── tests/                                     # Automated Unit Test Suite (100% Pass)
│   ├── test_data_loader.py
│   ├── test_models.py
│   └── test_evaluation.py
│
└── reports/
    ├── MOVIELENS_32M_FULL_SOLUTIONS.md        # Comprehensive Question-by-Question Solution Dossier
    └── benchmark_results.csv                  # Offline benchmark matrix
```

---

## 📊 Offline Benchmark Evaluation

*Evaluated under strict temporal train/test partitioning per user on hold-out test interactions:*

| Model Architecture | Precision@5 | Precision@10 | Precision@20 | Recall@5 | Recall@10 | Recall@20 | NDCG@10 | MAP@10 | Catalog Coverage@10 | Novelty@10 | Intra-List Diversity@10 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Popularity Baseline** | 0.0380 | 0.0310 | 0.0270 | 0.0301 | 0.0437 | 0.0722 | 0.0480 | 0.0265 | 0.01% | 9.1785 | 0.5216 |
| **SVD Matrix Factorization** | **0.0950** | **0.0800** | **0.0740** | 0.0555 | **0.0976** | **0.1786** | **0.1110** | **0.0560** | 0.41% | 9.4525 | 0.6777 |
| **Weighted Hybrid (Proposed)** | 0.0940 | 0.0765 | 0.0648 | **0.0564** | 0.0944 | 0.1546 | 0.1072 | 0.0537 | **0.44%** | **9.8218** | **0.7266** |

---

## ⚡ Quick Start & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Master Analysis Suite (All Sections Q1 to Q113)
```bash
python run_all.py
```

### 3. Run Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 4. Launch Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

### 5. Launch FastAPI REST Service
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```
Interactive Swagger API documentation available at `http://localhost:8000/docs`.

---

## 🔧 Reusable Scikit-Learn Pipeline Example

```python
import pandas as pd
from src.pipeline import MovieLensRecommendationPipeline
from data_loader import MovieLensDataLoader

# 1. Load Data
loader = MovieLensDataLoader(data_dir=r"E:\ml-32m")
movies_df = loader.load_movies()
tags_df = loader.load_tags()
ratings_df = loader.load_ratings(max_rows=500_000)

# 2. Initialize & Fit Estimator Pipeline
pipeline = MovieLensRecommendationPipeline(
    n_svd_components=32,
    collab_weight=0.60,
    content_weight=0.30,
    popularity_weight=0.10,
    diversity_penalty=0.15
)
pipeline.fit(ratings_df=ratings_df, movies_df=movies_df, tags_df=tags_df)

# 3. Generate Top-10 Recommendations for User 42
recommendations = pipeline.recommend(user_id=42, top_k=10, exclude_seen=True)
for movie_id, score in recommendations:
    print(f"Movie ID: {movie_id} -> Hybrid Score: {score:.3f}")
```
