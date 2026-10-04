"""CineMatch Streamlit application: discovery, personalization, cold start and model transparency."""
from __future__ import annotations
import html, os, re, sys
from pathlib import Path
from urllib.parse import quote
from typing import Any
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import build_cloud_demo_runtime, get_artifact_path, load_runtime_artifact, prepare_runtime

st.set_page_config(page_title="CineMatch · Movie Intelligence", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800;900&display=swap');
:root{--body:'Plus Jakarta Sans',sans-serif;--display:'Outfit',sans-serif}
.block-container{max-width:1480px;padding:1.5rem 2rem 4rem;font-family:var(--body)}
.hero{padding:2.7rem 3rem;border:1px solid rgba(255,80,120,.28);border-radius:28px;background:radial-gradient(circle at 90% 15%,rgba(255,51,102,.18),transparent 38%),radial-gradient(circle at 10% 90%,rgba(79,172,254,.13),transparent 38%),linear-gradient(145deg,rgba(23,29,45,.9),rgba(8,12,21,.96));box-shadow:0 20px 55px -25px #000;margin-bottom:1.4rem}
.hero h1{font-family:var(--display);font-size:clamp(2.4rem,5vw,4.4rem);line-height:1.02;margin:.4rem 0 1rem;color:#f8fafc;letter-spacing:-.045em}
.gradient{background:linear-gradient(135deg,#ff3366,#ff8e53);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{max-width:900px;color:#94a3b8;font-size:1.05rem;line-height:1.7}
.pill{display:inline-block;padding:.42rem .75rem;margin:.2rem;border-radius:999px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.09);color:#cbd5e1;font-size:.78rem;font-weight:700}
.section-kicker{color:#94a3b8;font-size:.75rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;margin:1rem 0 .55rem}
.movie-title{font-family:var(--display);font-size:1.1rem;font-weight:800;color:#f8fafc}.muted{color:#94a3b8;font-size:.82rem}.reason{color:#cbd5e1;font-size:.86rem;line-height:1.55;margin-top:.45rem}
.tag{display:inline-block;padding:.2rem .5rem;margin:.15rem;border-radius:7px;background:rgba(255,255,255,.05);color:#cbd5e1;font-size:.7rem;font-weight:700}.score{font-family:var(--display);font-size:1.35rem;font-weight:900;color:#ff6584}
.insight{padding:1rem;border:1px solid rgba(255,255,255,.08);border-radius:16px;background:rgba(255,255,255,.025);height:100%}
div[data-testid="stMetric"]{border-radius:18px!important;background:linear-gradient(145deg,rgba(26,33,52,.6),rgba(15,20,32,.8))!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d121f,#080c15)!important}
@media(max-width:800px){.block-container{padding:.8rem}.hero{padding:1.8rem 1.2rem}.hero h1{font-size:2.5rem}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<span class="pill">● CineMatch Intelligence</span><span class="pill">Scikit-Learn 1.5 · Python 3.12</span>
<h1>Next-Gen <span class="gradient">Movie Discovery</span></h1>
<p>Hybrid recommendation powered by TruncatedSVD collaborative filtering, TF-IDF content similarity, Bayesian popularity priors and diversity-aware re-ranking — with explainable results, discovery surfaces, cold-start onboarding and model transparency.</p>
<span class="pill">⚡ Hybrid ranking</span><span class="pill">🧠 Content + tags</span><span class="pill">🛡️ Bayesian cold start</span><span class="pill">🎯 Explainable</span><span class="pill">📈 Model insights</span>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Warming up the recommendation engine…")
def load_app_runtime() -> dict[str, Any]:
    """Load the offline artifact without full-scale training at app startup."""
    return prepare_runtime(load_runtime_artifact())

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tmdb_poster(tmdb_id: int) -> str | None:
    """Return a TMDB poster URL when an optional API key is configured."""
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key or not tmdb_id:
        return None
    try:
        from urllib.request import Request, urlopen
        import json
        request = Request(f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={quote(key)}", headers={"User-Agent":"CineMatch/1.0"})
        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        poster_path = payload.get("poster_path")
        return f"https://image.tmdb.org/t/p/w342{poster_path}" if poster_path else None
    except Exception:
        return None

def movie_meta(movie_id: int) -> dict[str, Any]:
    return runtime["movie_lookup"].get(int(movie_id), {})

def movie_stats(movie_id: int) -> tuple[int, float, float]:
    mid = int(movie_id)
    count = int(runtime["pop_model"].movie_counts.get(mid, 0))
    mean = float(runtime["pop_model"].movie_means.get(mid, runtime["pop_model"].global_mean_C))
    bayes = float(runtime["pop_model"].movie_scores.get(mid, runtime["pop_model"].global_mean_C))
    return count, mean, bayes

def poster_for(movie_id: int) -> str | None:
    links = runtime.get("links_df")
    if links is None or links.empty or "tmdbId" not in links.columns:
        return None
    rows = links[links["movieId"].astype(int) == int(movie_id)]
    if rows.empty or pd.isna(rows.iloc[0]["tmdbId"]):
        return None
    return fetch_tmdb_poster(int(rows.iloc[0]["tmdbId"]))

def apply_filters(candidates, selected_genres, year_range, min_quality, limit, reason=""):
    results = []
    for movie_id, score in candidates:
        meta = movie_meta(movie_id)
        if not meta:
            continue
        genres = [g for g in str(meta.get("genres","")).split("|") if g and g != "(no genres listed)"]
        if selected_genres and not set(selected_genres).intersection(genres):
            continue
        year = meta.get("year")
        if pd.notna(year) and not year_range[0] <= int(year) <= year_range[1]:
            continue
        count, mean, bayes = movie_stats(movie_id)
        if bayes < min_quality:
            continue
        results.append({"movie_id":int(movie_id),"title":str(meta.get("title","Untitled")),"genres":genres,
                        "year":int(year) if pd.notna(year) else None,"score":float(score),
                        "quality":bayes,"mean":mean,"votes":count,"reason":reason})
        if len(results) >= limit:
            break
    return results

def render_movie_cards(items, score_label, show_posters=False, key_prefix="results"):
    if not items:
        st.info("No titles matched these controls. Expand the filters and try again.")
        return
    for rank, movie in enumerate(items, 1):
        with st.container(border=True):
            cols = st.columns([1.0,4.9,1.1,1.1] if show_posters else [5.9,1.1,1.1], gap="medium")
            offset = 0
            if show_posters:
                with cols[0]:
                    poster = poster_for(movie["movie_id"])
                    if poster:
                        st.image(poster, width="stretch")
                    else:
                        st.markdown(f"<div class='insight'><b>#{rank:02d}</b><br><br>🎬<br><span class='muted'>Poster</span></div>", unsafe_allow_html=True)
                offset = 1
            with cols[offset]:
                tags = "".join(f"<span class='tag'>{html.escape(g)}</span>" for g in movie["genres"]) or "<span class='tag'>General</span>"
                st.markdown(f"<div class='movie-title'>#{rank:02d} {html.escape(movie['title'])}{' · '+str(movie['year']) if movie['year'] else ''}</div><div>{tags}</div><div class='muted'>⭐ {movie['mean']:.2f}/5 · Bayesian {movie['quality']:.2f} · {movie['votes']:,} ratings</div><div class='reason'>💡 {html.escape(movie['reason'] or 'Ranked by the selected recommendation strategy.')}</div>", unsafe_allow_html=True)
                with st.expander("Why this movie?", key=f"why-{key_prefix}-{movie['movie_id']}"):
                    st.write(movie["reason"] or "Selected from the active candidate set.")
                    a,b,c=st.columns(3)
                    a.metric("Recommendation score", f"{movie['score']:.3f}")
                    b.metric("Bayesian quality", f"{movie['quality']:.2f}")
                    c.metric("Rating support", f"{movie['votes']:,}")
            with cols[offset+1]:
                st.markdown(f"<div class='score'>{movie['score']:.2f}<br><span class='muted'>{score_label}</span></div>", unsafe_allow_html=True)
            with cols[offset+2]:
                if st.button("Save", key=f"save-{key_prefix}-{movie['movie_id']}", use_container_width=True):
                    st.session_state.setdefault("watchlist", [])
                    if movie["movie_id"] not in st.session_state["watchlist"]:
                        st.session_state["watchlist"].append(movie["movie_id"])
                    st.toast(f"Saved {movie['title']}")

try:
    runtime = load_app_runtime()
    cloud_demo = False
except FileNotFoundError:
    with st.spinner("Preparing the lightweight hosted demo runtime…"):
        runtime = build_cloud_demo_runtime()
    cloud_demo = True

movies_df = runtime["movies_df"]
pop_model = runtime["pop_model"]
content_model = runtime["content_model"]
hybrid_model = runtime["hybrid_model"]
genre_prior = runtime["genre_prior"]
svd_model = runtime["svd_model"]

all_genres = sorted({g.strip() for values in movies_df["genres"].dropna() for g in str(values).split("|") if g.strip() and g.strip() != "(no genres listed)"})
valid_years = movies_df["year"].dropna()
year_min = int(valid_years.min()) if not valid_years.empty else 1900
year_max = int(valid_years.max()) if not valid_years.empty else year_min

m1,m2,m3=st.columns(3)
m1.metric("🎬 Movies in Catalog",f"{len(movies_df):,}")
m2.metric("👤 User Profiles",f"{len(svd_model.user_to_idx):,}")
m3.metric("⚡ Intelligence Stack","Hybrid + SVD",help="Collaborative filtering + content similarity + Bayesian popularity.")

with st.container(border=True):
    st.markdown("<div class='section-kicker'>Discovery controls</div>",unsafe_allow_html=True)
    f1,f2,f3,f4=st.columns([2.2,1.7,1.5,1.2])
    with f1: selected_genres=st.multiselect("Genre Match",all_genres,placeholder="All genres")
    with f2: year_range=st.slider("Release Era",year_min,year_max,(year_min,year_max)) if year_min<year_max else (year_min,year_max)
    with f3: min_quality=st.slider("Min Quality",0.0,5.0,0.0,.1)
    with f4: top_k=st.slider("Top Results",5,20,10)

similar_tab,personal_tab,cold_tab,discover_tab,insights_tab=st.tabs(["🎯 Similar Film","👤 For You","✨ Cold Start","🔥 Trending & Gems","📊 Model Insights"])

with similar_tab:
    st.subheader("Discover Through a Film You Love")
    query=st.text_input("Search the catalog",placeholder="Try Inception, Toy Story, Interstellar…",key="movie_search").strip()
    if query:
        matches=movies_df[movies_df["title"].str.contains(re.escape(query),case=False,na=False,regex=True)].head(250)
    else:
        catalog_ids=set(runtime["movie_lookup"])
        matches=movies_df[movies_df["movieId"].isin([mid for mid,_ in pop_model.ranked_movies if mid in catalog_ids][:250])]
        st.caption("Showing popular titles. Search above to find a specific seed film.")
    ids=matches["movieId"].astype(int).tolist()
    if ids:
        seed=st.selectbox("Starting film",ids,index=None,placeholder="Select a movie",format_func=lambda x:movie_meta(x).get("title",f"Movie {x}"))
        if seed is not None and st.button("Find Similar Films",type="primary",use_container_width=True):
            raw=content_model.recommend(item_id=int(seed),top_k=max(top_k*8,80))
            reason=f"Content similarity to {movie_meta(seed).get('title','your seed film')}, emphasizing shared genre and tag signals."
            render_movie_cards(apply_filters(raw,selected_genres,year_range,min_quality,top_k,reason),"Similarity",True,"similar")

with personal_tab:
    st.subheader("Personalized For You")
    user_max=max(svd_model.user_to_idx,default=1)
    user_id=st.number_input("MovieLens user ID",1,max(1,user_max),min(1,user_max),key="movie_user_id")
    if st.button("Generate Watchlist",type="primary",use_container_width=True):
        uid=int(user_id)
        if uid not in svd_model.user_to_idx:
            st.warning("That user ID is not present in this runtime.")
        else:
            history=hybrid_model.user_history_map.get(uid,{})
            raw=hybrid_model.recommend(user_id=uid,top_k=max(top_k*8,80),exclude_seen=True)
            reason=f"Hybrid ranking from {len(history):,} rated films; collaborative taste signals are blended with content and popularity."
            st.caption(f"Profile has **{len(history):,}** rated films.")
            render_movie_cards(apply_filters(raw,selected_genres,year_range,min_quality,top_k,reason),"Match",True,"personal")

with cold_tab:
    st.subheader("Cold-Start Onboarding")
    st.write("No viewing history required. Pick genres and get Bayesian-smoothed recommendations.")
    favorite=st.multiselect("Your favorite genres",all_genres,key="cold_start_genres")
    if favorite and st.button("Build My Starter List",type="primary",use_container_width=True):
        raw=genre_prior.recommend(preferred_genres=favorite,top_k=max(top_k*8,80))
        reason=f"Cold-start ranking for {', '.join(favorite)} using Bayesian popularity and genre affinity."
        render_movie_cards(apply_filters(raw,selected_genres,year_range,min_quality,top_k,reason),"Quality",True,"cold")
    else:
        st.info("Choose one or more genres to start your personalized onboarding.")

with discover_tab:
    st.subheader("Discovery Radar")
    popular=[(mid,score) for mid,score in pop_model.ranked_movies if mid in runtime["movie_lookup"]]
    trend_results=apply_filters(popular,selected_genres,year_range,min_quality,top_k,"High Bayesian quality among the strongest catalog-wide popularity signals.")
    counts=pop_model.movie_counts
    threshold=max(100,int(pd.Series(list(counts.values())).quantile(.35))) if counts else 100
    gems=sorted([(mid,score) for mid,score in pop_model.ranked_movies if mid in runtime["movie_lookup"] and 3<=counts.get(mid,0)<=threshold],key=lambda x:x[1],reverse=True)
    gem_results=apply_filters(gems,selected_genres,year_range,min_quality,top_k,"Hidden gem: strong Bayesian quality with lighter rating support than the catalog's most popular titles.")
    a,b=st.columns(2)
    with a:
        st.markdown("### 🔥 Trending / Popular")
        render_movie_cards(trend_results,"Quality",False,"trending")
    with b:
        st.markdown("### 💎 Hidden Gems")
        render_movie_cards(gem_results,"Quality",False,"gems")

with insights_tab:
    st.subheader("Model & Data Insights")
    total_ratings=int(sum(pop_model.movie_counts.values()))
    stats=pd.Series(list(pop_model.movie_means.values()),dtype="float64")
    i1,i2,i3,i4=st.columns(4)
    i1.metric("Ratings indexed",f"{total_ratings:,}")
    i2.metric("Global mean",f"{pop_model.global_mean_C:.2f}/5")
    i3.metric("Rated titles",f"{len(pop_model.movie_counts):,}")
    i4.metric("SVD components",f"{getattr(svd_model,'n_components','—')}")
    st.markdown("### Recommendation architecture")
    c1,c2,c3=st.columns(3)
    c1.markdown("<div class='insight'><b>🧠 Content engine</b><br><span class='muted'>TF-IDF over title, genre and aggregated tags for item similarity and sparse histories.</span></div>",unsafe_allow_html=True)
    c2.markdown("<div class='insight'><b>⚡ Collaborative engine</b><br><span class='muted'>TruncatedSVD over the sparse user-item matrix learns latent preference structure.</span></div>",unsafe_allow_html=True)
    c3.markdown("<div class='insight'><b>🛡️ Ranking layer</b><br><span class='muted'>Hybrid score fusion, Bayesian quality prior and genre-aware diversity re-ranking.</span></div>",unsafe_allow_html=True)
    if not stats.empty:
        q1,q2,q3=st.columns(3)
        q1.metric("Median rating",f"{stats.median():.2f}")
        q2.metric("Top 10% threshold",f"{stats.quantile(.9):.2f}")
        q3.metric("Rating std. dev.",f"{stats.std():.2f}")
        st.bar_chart(stats.round(2).value_counts().sort_index().head(20))

with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("Recommendation control center")
    st.markdown("### Your Watchlist")
    watchlist=st.session_state.get("watchlist",[])
    if not watchlist:
        st.info("Use Save on any recommendation.")
    else:
        for mid in watchlist:
            st.write(f"• {movie_meta(mid).get('title',f'Movie {mid}')}")
        if st.button("Clear watchlist",use_container_width=True):
            st.session_state["watchlist"]=[]
            st.rerun()
    st.divider()
    st.markdown("### Poster enrichment")
    if os.environ.get("TMDB_API_KEY"):
        st.success("TMDB poster enrichment enabled.")
    else:
        st.caption("Optional: set TMDB_API_KEY to show live poster artwork. The app remains fully functional without it.")
    st.divider()
    st.caption(f"Runtime artifact: {get_artifact_path()}")
    if cloud_demo:
        st.caption("Hosted lightweight demo runtime is active; the full artifact remains artifact-first.")

st.markdown("<div style='margin-top:3rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,.08);text-align:center;color:#64748b;font-size:.82rem'>🎬 <b>CineMatch</b> · MovieLens recommendation platform · Scikit-Learn + TruncatedSVD + FastAPI</div>",unsafe_allow_html=True)
