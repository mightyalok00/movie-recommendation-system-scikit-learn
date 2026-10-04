"""Streamlit interface for the pre-built MovieLens recommendation runtime."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import streamlit as st

from src.runtime import get_artifact_path, load_runtime_artifact, prepare_runtime


st.set_page_config(
    page_title="CineMatch · Movie discovery",
    page_icon=":material/movie:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.badge(
    "MovieLens · intelligent discovery",
    color="orange",
    icon=":material/movie_filter:",
)
st.title("Find your next favorite film.")
st.write(
    "Explore films through content similarity, personalized collaborative "
    "recommendations, and genre-led discovery—all powered by the project's "
    "Scikit-learn recommendation runtime."
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
        with st.container(border=True):
            details, score = st.columns([5, 1], vertical_alignment="center")
            with details:
                year = f" · {movie['year']}" if movie["year"] else ""
                st.subheader(f"{rank:02d} · {movie['title']}{year}")
                st.caption("  ·  ".join(movie["genres"]) or "Genres not listed")
            with score:
                st.metric(score_label, f"{movie['score']:.2f}")
                if movie["quality"] is not None:
                    st.caption(f"Quality estimate · {movie['quality']:.2f} / 5")


try:
    runtime = load_app_runtime()
except FileNotFoundError:
    show_setup_help()

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

st.sidebar.markdown("### Your discovery settings")
selected_genres = st.sidebar.multiselect(
    "Include any of these genres",
    options=all_genres,
    help="Leave empty to include every genre.",
)
if year_min < year_max:
    year_range = st.sidebar.slider(
        "Release years",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )
else:
    year_range = (year_min, year_max)
min_quality = st.sidebar.slider(
    "Minimum quality estimate",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.1,
)
top_k = st.sidebar.slider("Titles per view", min_value=5, max_value=20, value=10)
st.sidebar.divider()
st.sidebar.caption(
    "Recommendations run locally from a pre-built model bundle. "
    "Your filters are applied in this session."
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
