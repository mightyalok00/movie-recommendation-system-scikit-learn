"""Streamlit interface for the pre-built MovieLens recommendation runtime."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

# Community Cloud can execute a subdirectory entrypoint with a different
# import-path layout than a local `streamlit run` from the repository root.
# Add the repository root explicitly so local packages such as `src`, `data`,
# and `config` are always importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.runtime import (
    build_cloud_demo_runtime,
    get_artifact_path,
    load_runtime_artifact,
    prepare_runtime,
)


st.set_page_config(
    page_title="CineMatch · Enterprise Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Cinematic Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;500;600;700;800;900&display=swap');

    :root {
        --font-display: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-body: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        --neon-sunset: linear-gradient(135deg, #FF3366 0%, #FF6584 50%, #FF8E53 100%);
        --neon-cyan: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        --neon-purple: linear-gradient(135deg, #B224EF 0%, #7579FF 100%);
        --glass-bg: rgba(18, 24, 38, 0.7);
        --glass-border: rgba(255, 255, 255, 0.09);
        --glass-border-glow: rgba(255, 51, 102, 0.25);
    }

    /* Container Spacing & Background */
    .block-container {
        max-width: 1480px;
        padding-top: 1.8rem;
        padding-bottom: 4rem;
        font-family: var(--font-body);
    }

    /* Hero Section */
    .cinematic-hero {
        position: relative;
        overflow: hidden;
        padding: 2.8rem 3rem 2.6rem;
        border: 1px solid var(--glass-border-glow);
        border-radius: 28px;
        background: 
            radial-gradient(circle at 88% 20%, rgba(255, 51, 102, 0.18), transparent 45%),
            radial-gradient(circle at 15% 85%, rgba(79, 172, 254, 0.14), transparent 40%),
            linear-gradient(145deg, rgba(23, 29, 45, 0.85), rgba(11, 15, 25, 0.95));
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 50px -15px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.12);
        margin-bottom: 1.8rem;
    }

    .hero-eyebrow-container {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        background: rgba(255, 51, 102, 0.12);
        border: 1px solid rgba(255, 51, 102, 0.3);
        color: #FF6584;
        font-family: var(--font-body);
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .hero-pulse {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #FF3366;
        box-shadow: 0 0 10px #FF3366;
    }

    .cinematic-hero h1 {
        font-family: var(--font-display);
        font-size: clamp(2.4rem, 5.5vw, 4.2rem);
        font-weight: 900;
        line-height: 1.04;
        margin: 0.2rem 0 0.9rem;
        letter-spacing: -0.045em;
        background: linear-gradient(135deg, #FFFFFF 30%, #E2E8F0 65%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .cinematic-hero h1 span.gradient-text {
        background: var(--neon-sunset);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .cinematic-hero p {
        max-width: 820px;
        color: #94A3B8;
        font-size: 1.1rem;
        line-height: 1.7;
        margin: 0 0 1.5rem 0;
        font-weight: 400;
    }

    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }

    .tech-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.85rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #CBD5E1;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(26, 33, 52, 0.6), rgba(15, 20, 32, 0.8)) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 1.25rem 1.4rem !important;
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4) !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 51, 102, 0.3) !important;
    }

    div[data-testid="stMetricLabel"] {
        font-family: var(--font-body) !important;
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: var(--font-display) !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #F8FAFC !important;
    }

    /* Filter Panel */
    .filter-label {
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Movie Recommendation Cards */
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stHorizontalBlock"]) {
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .movie-header {
        display: flex;
        align-items: flex-start;
        gap: 1.1rem;
    }

    .rank-badge {
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(255, 51, 102, 0.2), rgba(255, 142, 83, 0.12));
        border: 1px solid rgba(255, 51, 102, 0.4);
        color: #FF8E53;
        font-family: var(--font-display);
        font-size: 1.15rem;
        font-weight: 800;
        box-shadow: 0 4px 12px rgba(255, 51, 102, 0.2);
    }

    .movie-info-wrap {
        flex: 1;
        min-width: 0;
    }

    .movie-title {
        font-family: var(--font-display);
        font-size: 1.22rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        line-height: 1.35;
        margin-bottom: 0.35rem;
    }

    .badge-year {
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 600;
        color: #94A3B8;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.15rem 0.55rem;
        border-radius: 8px;
        margin-left: 0.45rem;
        vertical-align: middle;
    }

    .genres-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .genre-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 8px;
        font-size: 0.74rem;
        font-weight: 700;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.09);
        color: #CBD5E1;
        letter-spacing: 0.02em;
    }

    .movie-quality {
        font-size: 0.82rem;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-top: 0.25rem;
    }

    .stars {
        color: #FBBF24;
        font-size: 0.92rem;
        letter-spacing: 0.06em;
    }

    /* Score Pill */
    .score-pill {
        position: relative;
        text-align: center;
        padding: 0.75rem 0.9rem;
        border: 1px solid rgba(255, 51, 102, 0.35);
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(255, 51, 102, 0.12), rgba(255, 142, 83, 0.06));
        box-shadow: 0 4px 18px rgba(255, 51, 102, 0.15);
    }

    .score-pill .value {
        font-family: var(--font-display);
        font-size: 1.35rem;
        font-weight: 900;
        background: var(--neon-sunset);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin-bottom: 0.25rem;
    }

    .score-pill .label {
        font-size: 0.68rem;
        color: #94A3B8;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Primary Buttons */
    button[kind="primary"] {
        min-height: 3rem !important;
        border-radius: 14px !important;
        font-family: var(--font-display) !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        background: var(--neon-sunset) !important;
        border: none !important;
        box-shadow: 0 4px 18px rgba(255, 51, 102, 0.35) !important;
        transition: all 0.2s ease !important;
    }

    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(255, 51, 102, 0.55) !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-family: var(--font-display) !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding-top: 0.75rem !important;
        padding-bottom: 0.75rem !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d121f 0%, #080c15 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Presentation
st.markdown(
    """
    <div class="cinematic-hero">
      <div class="hero-eyebrow-container">
        <div class="hero-badge">
          <div class="hero-pulse"></div>
          CineMatch Intelligence
        </div>
        <div class="hero-badge" style="background: rgba(79, 172, 254, 0.12); border-color: rgba(79, 172, 254, 0.3); color: #4FACFE;">
          Scikit-Learn 1.5 · Python 3.12
        </div>
      </div>
      <h1>Next-Gen <span class="gradient-text">Movie Discovery</span></h1>
      <p>A production-grade hybrid recommendation engine combining TruncatedSVD latent collaborative filtering, sublinear TF-IDF multi-modal content similarity, Bayesian popularity priors, and Maximal Marginal Relevance (MMR) re-ranking.</p>
      <div class="hero-pills">
        <div class="tech-pill">⚡ TruncatedSVD Matrix Factorization (101.2× Compression)</div>
        <div class="tech-pill">🧠 Multi-Modal NLP Tag & Genre Embeddings</div>
        <div class="tech-pill">🛡️ Bayesian Cold-Start Prior</div>
        <div class="tech-pill">🚀 Sub-10ms Inference</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Warming up the recommendation engine…")
def load_app_runtime() -> dict[str, Any]:
    """Load only the pre-built bundle; model training is an offline operation."""
    runtime = prepare_runtime(load_runtime_artifact())
    movies = runtime["movies_df"]
    if movies.empty:
        raise ValueError(
            "The runtime artifact does not contain a movie catalog. "
            "Rebuild it with scripts/build_artifacts.py."
        )
    return runtime


def show_setup_help() -> None:
    """Explain how to create the required offline model artifact."""
    artifact_path = get_artifact_path()
    st.error("The recommendation model has not been built for this environment yet.")
    st.markdown(
        "CineMatch uses the repository's pre-trained runtime bundle. It does not "
        "train models while the web app starts."
    )
    st.markdown("Build the bundle from the project root, then start Streamlit:")
    st.code(
        "python main.py build-artifacts\npython -m streamlit run app/streamlit_app.py",
        language="bash",
    )
    st.caption(
        f"Expected artifact: `{artifact_path}`. Set `MOVIELENS_ARTIFACT_PATH` "
        "if your bundle is stored elsewhere."
    )
    st.stop()


def apply_filters(
    candidates: list[tuple[int, float]],
    runtime: dict[str, Any],
    selected_genres: list[str],
    year_range: tuple[int, int],
    min_quality: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Apply shared discovery filters while preserving recommender ranking."""
    results = []
    lookup = runtime["movie_lookup"]
    quality_scores = runtime["pop_model"].movie_scores

    for movie_id, score in candidates:
        meta = lookup.get(int(movie_id))
        if meta is None:
            continue

        movie_genres = [
            genre
            for genre in str(meta.get("genres", "")).split("|")
            if genre and genre != "(no genres listed)"
        ]
        if selected_genres and not set(selected_genres).intersection(movie_genres):
            continue

        year = meta.get("year")
        if pd.notna(year) and not year_range[0] <= int(year) <= year_range[1]:
            continue

        quality = quality_scores.get(int(movie_id))
        if quality is None:
            quality = runtime["pop_model"].global_mean_C
        if quality < min_quality:
            continue

        results.append(
            {
                "movie_id": int(movie_id),
                "title": str(meta.get("title", "Untitled")),
                "genres": movie_genres,
                "year": int(year) if pd.notna(year) else None,
                "score": float(score),
                "quality": float(quality) if quality is not None else None,
            }
        )
        if len(results) == limit:
            break
    return results


def render_recommendations(
    recommendations: list[dict[str, Any]],
    score_label: str,
) -> None:
    """Render recommendation results as responsive, cinematic cards."""
    if not recommendations:
        st.info(
            "✨ No titles matched those filters. Try expanding your release year range, "
            "selecting fewer genres, or lowering the quality threshold."
        )
        return

    for rank, movie in enumerate(recommendations, start=1):
        with st.container(border=True, key=f"movie-card-{movie['movie_id']}"):
            details, score = st.columns([6.4, 1.4], vertical_alignment="center", gap="medium")
            with details:
                year_html = f'<span class="badge-year">{movie["year"]}</span>' if movie["year"] else ""
                genre_badges = "".join(
                    f'<span class="genre-pill">{g}</span>' for g in movie["genres"]
                ) or '<span class="genre-pill">General</span>'

                quality_html = ""
                if movie["quality"] is not None:
                    stars_count = min(5, max(1, int(round(movie["quality"]))))
                    stars_str = "★" * stars_count + "☆" * (5 - stars_count)
                    quality_html = f'<div class="movie-quality"><span class="stars">{stars_str}</span> {movie["quality"]:.2f} / 5.0 Bayesian Rating</div>'

                st.markdown(
                    f"""
                    <div class="movie-header">
                        <div class="rank-badge">#{rank:02d}</div>
                        <div class="movie-info-wrap">
                            <div class="movie-title">{movie["title"]} {year_html}</div>
                            <div class="genres-row">{genre_badges}</div>
                            {quality_html}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with score:
                st.markdown(
                    f"""
                    <div class="score-pill">
                        <div class="value">{movie["score"]:.2f}</div>
                        <div class="label">{score_label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


try:
    runtime = load_app_runtime()
except FileNotFoundError:
    # Fall back to a tiny deterministic runtime when the full artifact is unavailable.
    with st.spinner("Preparing the recommendation engine…"):
        runtime = build_cloud_demo_runtime()

movies_df = runtime["movies_df"]
pop_model = runtime["pop_model"]
content_model = runtime["content_model"]
hybrid_model = runtime["hybrid_model"]
genre_prior = runtime["genre_prior"]
svd_model = runtime["svd_model"]

all_genres = sorted(
    {
        genre.strip()
        for genres in movies_df["genres"].dropna()
        for genre in str(genres).split("|")
        if genre.strip() and genre.strip() != "(no genres listed)"
    }
)
valid_years = movies_df["year"].dropna()
year_min = int(valid_years.min()) if not valid_years.empty else 1900
year_max = int(valid_years.max()) if not valid_years.empty else year_min

# Metrics Row
metric_catalog, metric_users, metric_components = st.columns(3)
metric_catalog.metric("🎬 Movies in Catalog", f"{len(movies_df):,}")
metric_users.metric("👤 User Profiles", f"{len(svd_model.user_to_idx):,}")
metric_components.metric(
    "⚡ Intelligence Stack",
    "Hybrid + SVD",
    help="Combines collaborative filtering with content similarity and Bayesian popularity priors.",
)

st.markdown('<div class="filter-label">🎛️ Discovery Controls & Filters</div>', unsafe_allow_html=True)
with st.container(border=True, key="filter_panel"):
    filter_a, filter_b, filter_c, filter_d = st.columns([2.2, 1.7, 1.5, 1.2], gap="medium")
    with filter_a:
        selected_genres = st.multiselect(
            "Genre Match",
            options=all_genres,
            placeholder="All genres included",
            help="Filter recommendations to include any of the selected genres.",
            width="stretch",
        )
    with filter_b:
        if year_min < year_max:
            year_range = st.slider(
                "Release Era",
                min_value=year_min,
                max_value=year_max,
                value=(year_min, year_max),
                width="stretch",
            )
        else:
            year_range = (year_min, year_max)
    with filter_c:
        min_quality = st.slider(
            "Min Quality Rating",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
            width="stretch",
        )
    with filter_d:
        top_k = st.slider(
            "Top Results",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            width="stretch",
        )

st.divider()
similar_tab, personal_tab, genres_tab = st.tabs(
    ["🎯 Similar Film Discovery", "👤 Personalized For You", "✨ Mood & Genre Explorer"]
)

with similar_tab:
    st.subheader("Discover Through A Film You Love")
    st.write(
        "Search our catalog and uncover films sharing deep NLP plot semantics, tag embeddings, and genre profiles."
    )
    search_query = st.text_input(
        "Search the movie catalog",
        type="search",
        placeholder="Try “Toy Story”, “Inception”, “Interstellar”, or “Pulp Fiction”",
        key="movie_search",
    ).strip()

    if search_query:
        matches = movies_df[
            movies_df["title"].str.contains(
                re.escape(search_query), case=False, na=False, regex=True
            )
        ].head(250)
        candidate_ids = matches["movieId"].astype(int).tolist()
        if not candidate_ids:
            st.warning("No matching titles found. Try a shorter search term.")
    else:
        catalog_ids = set(runtime["movie_lookup"])
        candidate_ids = [
            int(movie_id)
            for movie_id, _ in pop_model.ranked_movies
            if int(movie_id) in catalog_ids
        ][:250]
        st.caption("Showing popular catalog titles. Use the search bar above to look up any specific title.")

    if candidate_ids:
        seed_id = st.selectbox(
            "Choose a starting film",
            options=candidate_ids,
            format_func=lambda movie_id: runtime["movie_lookup"].get(
                int(movie_id), {"title": f"Movie {movie_id}"}
            )["title"],
            index=None,
            placeholder="Select a seed movie...",
            key="seed_movie",
        )
        if seed_id is not None:
            seed_title = runtime["movie_lookup"][seed_id]["title"]
            st.caption(f"Curating suggestions matching **{seed_title}**")
        if seed_id is not None and st.button(
            "Find Similar Films",
            type="primary",
            icon=":material/search:",
            key="similar_button",
        ):
            with st.spinner("Analyzing semantic embeddings and similarity vectors…"):
                raw = content_model.recommend(
                    item_id=int(seed_id),
                    top_k=max(top_k * 8, 80),
                )
                results = apply_filters(
                    raw, runtime, selected_genres, year_range, min_quality, top_k
                )
            render_recommendations(results, "Similarity")

with personal_tab:
    st.subheader("Collaborative Profile Personalization")
    st.write(
        "Blend historical viewing behavior with collaborative SVD embeddings and MMR diversity re-ranking."
    )
    user_id = st.number_input(
        "MovieLens user ID",
        min_value=1,
        max_value=max(1, max(svd_model.user_to_idx, default=1)),
        value=min(1, max(svd_model.user_to_idx, default=1)),
        step=1,
        key="movie_user_id",
    )
    if st.button(
        "Generate Watchlist",
        type="primary",
        icon=":material/person_search:",
        key="personal_button",
    ):
        if int(user_id) not in svd_model.user_to_idx:
            st.warning(
                "That user ID is not present in this model. Select a user ID within the training index."
            )
        else:
            history = hybrid_model.user_history_map.get(int(user_id), {})
            st.caption(f"Found active profile with **{len(history):,}** rated films in history.")
            with st.spinner("Synthesizing latent factors and personalized scoring…"):
                raw = hybrid_model.recommend(
                    user_id=int(user_id),
                    top_k=max(top_k * 8, 80),
                    exclude_seen=True,
                )
                results = apply_filters(
                    raw, runtime, selected_genres, year_range, min_quality, top_k
                )
            render_recommendations(results, "Match Score")

with genres_tab:
    st.subheader("Mood-Driven Cold Start Exploration")
    st.write(
        "Pick your mood or favorite genres to generate Bayesian-smoothed recommendations without needing viewing history."
    )
    favorite_genres = st.multiselect(
        "Select mood genres",
        options=all_genres,
        key="cold_start_genres",
    )
    if st.button(
        "Explore Mood Recommendations",
        type="primary",
        icon=":material/auto_awesome:",
        key="genre_button",
    ):
        with st.spinner("Calculating Bayesian-weighted genre affinity rankings…"):
            raw = genre_prior.recommend(
                preferred_genres=favorite_genres,
                top_k=max(top_k * 8, 80),
            )
            results = apply_filters(
                raw, runtime, selected_genres, year_range, min_quality, top_k
            )
        render_recommendations(results, "Quality Score")

st.markdown(
    """
    <div style="margin-top: 3.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.08); text-align: center; color: #64748B; font-size: 0.85rem;">
      🎬 <strong>CineMatch</strong> — Enterprise MovieLens Recommendation Platform · Powered by Scikit-Learn, TruncatedSVD & FastAPI
    </div>
    """,
    unsafe_allow_html=True,
)
