"""
MovieLens 32M Data Ingestion & Sparse Matrix Generation Module
==============================================================
Provides high-performance, memory-efficient data loading, downcasting,
and sparse matrix generation for the 32-million interaction MovieLens dataset.
"""

import os
import gc
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from scipy.sparse import csr_matrix
from config.settings import (
    MOVIES_FILE, RATINGS_FILE, TAGS_FILE, LINKS_FILE,
    DTYPE_MAPPINGS, RANDOM_SEED
)


class MovieLensDataLoader:
    """
    High-performance data loader tailored for the MovieLens 32M dataset.
    Implements memory-safe loading, downcasting, chunking, and sparse matrix generation.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir:
            self.movies_path = os.path.join(data_dir, "movies.csv")
            self.ratings_path = os.path.join(data_dir, "ratings.csv")
            self.tags_path = os.path.join(data_dir, "tags.csv")
            self.links_path = os.path.join(data_dir, "links.csv")
        else:
            self.movies_path = str(MOVIES_FILE)
            self.ratings_path = str(RATINGS_FILE)
            self.tags_path = str(TAGS_FILE)
            self.links_path = str(LINKS_FILE)

    def load_movies(self) -> pd.DataFrame:
        """
        Loads the movies dataset with clean title parsing and release year extraction.
        
        Returns:
            pd.DataFrame: DataFrame containing movieId, title, clean_title, year, and genres.
        """
        target_path = self.movies_path
        if not os.path.exists(target_path):
            from config.settings import BASE_DIR
            fallback = BASE_DIR / "data" / "movies.csv"
            if fallback.exists():
                target_path = str(fallback)

        df = pd.read_csv(
            target_path,
            dtype=DTYPE_MAPPINGS["movies"]
        )
        # Extract release year from title string (e.g., 'Toy Story (1995)' -> 1995)
        df["year"] = df["title"].str.extract(r"\((\d{4})\)$", expand=False).astype("float32")
        # Clean title without trailing year for cleaner TF-IDF and keyword matching
        df["clean_title"] = df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()
        return df

    def load_tags(self, min_tag_freq: int = 2) -> pd.DataFrame:
        """
        Loads user-assigned tags and aggregates tags per movie into a clean text representation.
        
        Args:
            min_tag_freq: Minimum frequency for a tag across all movies to be retained.
            
        Returns:
            pd.DataFrame: Aggregated tags per movie (movieId, combined_tags).
        """
        df = pd.read_csv(
            self.tags_path,
            dtype=DTYPE_MAPPINGS["tags"]
        )
        # Drop missing tag records
        df = df.dropna(subset=["tag"])
        # Standardize tag strings: lowercase and strip whitespace
        df["tag_clean"] = df["tag"].str.lower().str.strip()
        
        # Filter rare tags to reduce noise and vocabulary size
        if min_tag_freq > 1:
            tag_counts = df["tag_clean"].value_counts()
            valid_tags = set(tag_counts[tag_counts >= min_tag_freq].index)
            df = df[df["tag_clean"].isin(valid_tags)]

        # Group and concatenate tags per movie
        aggregated_tags = (
            df.groupby("movieId")["tag_clean"]
            .apply(lambda tags: " ".join(tags))
            .reset_index()
            .rename(columns={"tag_clean": "combined_tags"})
        )
        return aggregated_tags

    def load_ratings(
        self,
        sample_fraction: Optional[float] = None,
        max_rows: Optional[int] = None,
        min_user_ratings: int = 0,
        min_movie_ratings: int = 0
    ) -> pd.DataFrame:
        """
        Loads rating records with options for memory-safe sampling and activity filtering.
        
        Args:
            sample_fraction: Optional fraction of ratings to load (e.g. 0.1 for 10%).
            max_rows: Maximum number of rows to load directly.
            min_user_ratings: Filter out users with fewer than this number of interactions.
            min_movie_ratings: Filter out movies with fewer than this number of interactions.
            
        Returns:
            pd.DataFrame: Ratings dataframe with optimal memory dtypes.
        """
        if sample_fraction is not None and sample_fraction < 1.0:
            chunks = []
            chunk_size = 2_000_000
            for chunk in pd.read_csv(self.ratings_path, chunksize=chunk_size, dtype=DTYPE_MAPPINGS["ratings"]):
                sampled_chunk = chunk.sample(frac=sample_fraction, random_state=RANDOM_SEED)
                chunks.append(sampled_chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(
                self.ratings_path,
                nrows=max_rows,
                dtype=DTYPE_MAPPINGS["ratings"]
            )

        # Apply k-core activity filters if specified
        if min_user_ratings > 0 or min_movie_ratings > 0:
            user_counts = df["userId"].value_counts()
            movie_counts = df["movieId"].value_counts()
            
            valid_users = user_counts[user_counts >= min_user_ratings].index
            valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
            
            df = df[df["userId"].isin(valid_users) & df["movieId"].isin(valid_movies)].reset_index(drop=True)

        return df

    def get_user_movie_sparse_matrix(
        self,
        ratings_df: pd.DataFrame
    ) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
        """
        Constructs a compressed sparse row (CSR) user-item matrix from ratings DataFrame.
        
        Returns:
            matrix: scipy.sparse.csr_matrix (users x items)
            user_to_idx: dict mapping userId -> matrix row index
            idx_to_user: dict mapping matrix row index -> userId
            movie_to_idx: dict mapping movieId -> matrix column index
            idx_to_movie: dict mapping matrix column index -> movieId
        """
        unique_users = np.sort(ratings_df["userId"].unique())
        unique_movies = np.sort(ratings_df["movieId"].unique())

        user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}

        movie_to_idx = {mid: idx for idx, mid in enumerate(unique_movies)}
        idx_to_movie = {idx: mid for mid, idx in movie_to_idx.items()}

        rows = ratings_df["userId"].map(user_to_idx).to_numpy()
        cols = ratings_df["movieId"].map(movie_to_idx).to_numpy()
        values = ratings_df["rating"].to_numpy(dtype=np.float32)

        n_users = len(unique_users)
        n_movies = len(unique_movies)

        matrix = csr_matrix((values, (rows, cols)), shape=(n_users, n_movies), dtype=np.float32)
        return matrix, user_to_idx, idx_to_user, movie_to_idx, idx_to_movie

    def load_full_content_metadata(self) -> pd.DataFrame:
        """
        Merges movies metadata with aggregated tags into a unified content representation.
        
        Returns:
            pd.DataFrame: Merged metadata with rich text representations.
        """
        movies = self.load_movies()
        tags = self.load_tags(min_tag_freq=2)
        
        # Left join so that movies without tags are preserved
        merged = pd.merge(movies, tags, on="movieId", how="left")
        merged["combined_tags"] = merged["combined_tags"].fillna("")
        
        # Clean genre string (replace '|' with spaces for tokenization)
        merged["genre_tokens"] = merged["genres"].str.replace("|", " ", regex=False).str.replace("(no genres listed)", "", regex=False)
        
        # Combined content document for unified content-based modeling
        merged["content_soup"] = (
            merged["clean_title"] + " " +
            merged["genre_tokens"] + " " +
            merged["combined_tags"]
        ).str.strip()

        return merged
