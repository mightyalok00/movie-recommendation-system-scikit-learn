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
    page_title="CineMatch · Movie discovery",
    page_icon=":material/movie:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1480px; padding-top: 2.2rem; padding-bottom: 3rem;}
    .hero {padding: 2rem 2.2rem 1.8rem; border: 1px solid rgba(249,115,91,.22); border-radius: 24px; background: radial-gradient(circle at 85% 10%, rgba(249,115,91,.16), transparent 34%), linear-gradient(135deg, rgba(255,255,255,.045), rgba(255,255,255,.015)); margin-bottom: 1.25rem;}
    .eyebrow {color:#f9735b; font-size:.78rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase;}
    .hero h1 {font-size:clamp(2.2rem,5vw,4.5rem); line-height:.98; margin:.35rem 0 .8rem; letter-spacing:-.055em;}
    .hero p {max-width:760px; color:#aeb4c0; font-size:1.05rem; line-height:1.65; margin:0;}
    div[data-testid="stMetric"] {background:rgba(255,255,255,.025); border:1px solid rgba(255,255,255,.07); border-radius:16px; padding:1rem 1.1rem;}
    div[data-testid="stMetricLabel"] {color:#8f96a3;}
    .movie-title {font-size:1.08rem; font-weight:750; letter-spacing:-.01em;}
    .movie-meta {color:#8f96a3; font-size:.86rem; margin-top:.15rem;}
    .score-pill {text-align:center; padding:.55rem .7rem; border:1px solid rgba(249,115,91,.22); border-radius:14px; background:rgba(249,115,91,.07);}
    .score-pill .value {font-size:1.18rem; font-weight:800; color:#ff9a86;}
    .score-pill .label {font-size:.68rem; color:#8f96a3; text-transform:uppercase; letter-spacing:.08em;}
    button[kind="primary"] {min-height:2.8rem; border-radius:12px; font-weight:750;}
    [data-testid="stSidebar"] {border-right:1px solid rgba(255,255,255,.08);}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">CineMatch · MovieLens Intelligence</div>
      <h1>Find your next<br>favorite film.</h1>
      <p>Discover smarter with content similarity, collaborative personalization,
      and genre-led recommendations — wrapped in one clean movie discovery experience.</p>
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

    movies["movieId"] = movies["movieId"].astype(int)
    movies["year"] = (
        movies["title"].astype(str).str.extract(r"\((\d{4})\)\s*$", expand=False)
    )
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce").astype("Int64")
    runtime["movies_df"] = movies
    runtime["movie_lookup"] = movies.set_index("movieId").to_dict(orient="index")
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
    """Render recommendation results as responsive, keyboard-friendly cards."""
    if not recommendations:
        st.info(
            "No titles matched those filters. Try a wider year range, "
            "fewer genres, or a lower quality threshold."
        )
        return

    for rank, movie in enumerate(recommendations, start=1):
        with st.container(border=True, key=f"movie-card-{movie['movie_id']}"):
            details, score = st.columns([6.4, 1.2], vertical_alignment="center", gap="medium")
            with details:
                year = f" · {movie['year']}" if movie["year"] else ""
                st.markdown(f'<div class="movie-title">{rank:02d} · {movie["title"]}{year}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="movie-meta">{" · ".join(movie["genres"]) or "Genres not listed"}</div>', unsafe_allow_html=True)
                if movie["quality"] is not None:
                    st.caption(f"Quality estimate · {movie['quality']:.2f} / 5")
            with score:
                st.markdown(f'<div class="score-pill"><div class="value">{movie["score"]:.2f}</div><div class="label">{score_label}</div></div>', unsafe_allow_html=True)


try:
    runtime = load_app_runtime()
    cloud_demo_mode = False
except FileNotFoundError:
    # Streamlit Community Cloud does not have the local joblib artifact.
    # Fall back to a tiny deterministic runtime so the hosted UI remains usable.
    with st.spinner("Preparing the lightweight hosted demo…"):
        runtime = build_cloud_demo_runtime()
    cloud_demo_mode = True

movies_df = runtime["movies_df"]
pop_model = runtime["pop_model"]
content_model = runtime["content_model"]
hybrid_model = runtime["hybrid_model"]
genre_prior = runtime["genre_prior"]
svd_model = runtime["svd_model"]

if cloud_demo_mode:
    st.info(
        "Hosted demo mode · using a lightweight MovieLens sample because the full "
        "32M runtime artifact is not bundled with the public repository. The same "
        "recommendation pipeline powers this demo; production runs use the pre-built artifact.",
        icon=":material/cloud_done:",
    )

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

st.markdown('<div style="font-size:.78rem;font-weight:800;letter-spacing:.11em;text-transform:uppercase;color:#8f96a3;margin-bottom:.45rem;">Discovery filters</div>', unsafe_allow_html=True)
with st.container(border=True, key="filter_panel"):
    filter_a, filter_b, filter_c, filter_d = st.columns([2.2, 1.7, 1.5, 1.2], gap="medium")
    with filter_a:
        selected_genres = st.multiselect(
            "Genres",
            options=all_genres,
            placeholder="All genres",
            help="Select one or more genres. A title matching any selected genre is included.",
            width="stretch",
        )
    with filter_b:
        if year_min < year_max:
            year_range = st.slider(
                "Release years",
                min_value=year_min,
                max_value=year_max,
                value=(year_min, year_max),
                width="stretch",
            )
        else:
            year_range = (year_min, year_max)
    with filter_c:
        min_quality = st.slider(
            "Min. quality",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
            width="stretch",
        )
    with filter_d:
        top_k = st.slider(
            "Results",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            width="stretch",
        )

metric_catalog, metric_users, metric_components = st.columns(3)
metric_catalog.metric("Movies in catalog", f"{len(movies_df):,}")
metric_users.metric("Known user profiles", f"{len(svd_model.user_to_idx):,}")
metric_components.metric(
    "Personalization",
    "Hybrid + SVD",
    help="Combines collaborative filtering with content and popularity signals.",
)

st.divider()
similar_tab, personal_tab, genres_tab = st.tabs(
    ["Explore similar films", "For a returning viewer", "Start with a genre"]
)

with similar_tab:
    st.subheader("A film you love, a path to the next one")
    st.write(
        "Search the catalog, then discover films with similar genres, tags, and titles."
    )
    search_query = st.text_input(
        "Search the movie catalog",
        type="search",
        placeholder="Try “Toy Story”, “Inception”, or another title",
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
            st.warning("No matching titles found. Try a shorter search.")
    else:
        candidate_ids = [movie_id for movie_id, _ in pop_model.ranked_movies[:250]]
        st.caption("Popular catalog titles are shown until you search.")

    if candidate_ids:
        seed_id = st.selectbox(
            "Choose a starting film",
            options=candidate_ids,
            format_func=lambda movie_id: runtime["movie_lookup"][movie_id]["title"],
            index=None,
            placeholder="Select a movie",
            key="seed_movie",
        )
        if seed_id is not None:
            seed_title = runtime["movie_lookup"][seed_id]["title"]
            st.caption(f"Starting from **{seed_title}**")
        if seed_id is not None and st.button(
            "Find similar films",
            type="primary",
            icon=":material/search:",
            key="similar_button",
        ):
            with st.spinner("Finding a good next watch…"):
                raw = content_model.recommend(
                    item_id=int(seed_id),
                    top_k=max(top_k * 8, 80),
                )
                results = apply_filters(
                    raw, runtime, selected_genres, year_range, min_quality, top_k
                )
            render_recommendations(results, "Match")

with personal_tab:
    st.subheader("Recommendations shaped around a viewer")
    st.write(
        "Enter a user ID present in the model to blend viewing history with collaborative signals."
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
        "Build my recommendations",
        type="primary",
        icon=":material/person_search:",
        key="personal_button",
    ):
        if int(user_id) not in svd_model.user_to_idx:
            st.warning(
                "That user ID is not in this model. Try an ID included in the "
                "ratings used when the runtime artifact was built."
            )
        else:
            history = hybrid_model.user_history_map.get(int(user_id), {})
            st.caption(f"Using a profile with {len(history):,} rated films.")
            with st.spinner("Personalizing your watchlist…"):
                raw = hybrid_model.recommend(
                    user_id=int(user_id),
                    top_k=max(top_k * 8, 80),
                    exclude_seen=True,
                )
                results = apply_filters(
                    raw, runtime, selected_genres, year_range, min_quality, top_k
                )
            render_recommendations(results, "For you")

with genres_tab:
    st.subheader("Pick a mood, find a film")
    st.write(
        "Choose genres to discover highly rated films, even without a viewing history."
    )
    favorite_genres = st.multiselect(
        "Genres you’re in the mood for",
        options=all_genres,
        key="cold_start_genres",
    )
    if st.button(
        "Explore this mood",
        type="primary",
        icon=":material/auto_awesome:",
        key="genre_button",
    ):
        with st.spinner("Curating your genre-led picks…"):
            raw = genre_prior.recommend(
                preferred_genres=favorite_genres,
                top_k=max(top_k * 8, 80),
            )
            results = apply_filters(
                raw, runtime, selected_genres, year_range, min_quality, top_k
            )
        render_recommendations(results, "Community")

st.caption(
    "CineMatch is a discovery interface for the MovieLens recommendation project. "
    "Scores are model signals, not guarantees of personal preference."
)
