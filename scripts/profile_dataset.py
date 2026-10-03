"""Generate YData Profiling reports for MovieLens data.

The full MovieLens 32M ratings file is intentionally not profiled in-memory.
Instead, this script profiles representative bounded samples and the smaller
catalog tables, making it practical on developer machines and CI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport


def read_sample(path: Path, sample_rows: int) -> pd.DataFrame:
    """Read a bounded sample from a CSV without loading the full file."""
    return pd.read_csv(path, nrows=sample_rows)


def build_profile(df: pd.DataFrame, title: str, output_path: Path) -> None:
    """Build and save a YData Profiling HTML report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = ProfileReport(
        df,
        title=title,
        minimal=True,
        progress_bar=False,
    )
    profile.to_file(output_path)


def generate_profiles(
    data_dir: Path,
    output_dir: Path,
    ratings_sample: int,
    tags_sample: int,
) -> list[Path]:
    """Generate bounded MovieLens profiling reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    movies_path = data_dir / "movies.csv"
    if movies_path.exists():
        movies = pd.read_csv(movies_path)
        target = output_dir / "movies_profile.html"
        build_profile(movies, "MovieLens Movies Profile", target)
        generated.append(target)

    links_path = data_dir / "links.csv"
    if links_path.exists():
        links = pd.read_csv(links_path)
        target = output_dir / "links_profile.html"
        build_profile(links, "MovieLens Links Profile", target)
        generated.append(target)

    ratings_path = data_dir / "ratings.csv"
    if ratings_path.exists():
        ratings = read_sample(ratings_path, ratings_sample)
        target = output_dir / "ratings_sample_profile.html"
        build_profile(
            ratings,
            f"MovieLens Ratings Sample Profile ({len(ratings):,} rows)",
            target,
        )
        generated.append(target)

    tags_path = data_dir / "tags.csv"
    if tags_path.exists():
        tags = read_sample(tags_path, tags_sample)
        target = output_dir / "tags_sample_profile.html"
        build_profile(
            tags,
            f"MovieLens Tags Sample Profile ({len(tags):,} rows)",
            target,
        )
        generated.append(target)

    if not generated:
        raise FileNotFoundError(
            f"No MovieLens CSV files were found in {data_dir}. "
            "Point --data-dir to the extracted dataset."
        )

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate bounded YData Profiling HTML reports for MovieLens."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing MovieLens CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/ydata"),
        help="Directory for generated HTML reports.",
    )
    parser.add_argument(
        "--ratings-sample",
        type=int,
        default=100_000,
        help="Maximum number of rating rows to profile.",
    )
    parser.add_argument(
        "--tags-sample",
        type=int,
        default=100_000,
        help="Maximum number of tag rows to profile.",
    )
    args = parser.parse_args()

    generated = generate_profiles(
        args.data_dir,
        args.output_dir,
        args.ratings_sample,
        args.tags_sample,
    )
    print("Generated YData Profiling reports:")
    for path in generated:
        print(f" - {path}")


if __name__ == "__main__":
    main()
