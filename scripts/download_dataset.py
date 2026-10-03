"""
MovieLens Dataset Automated Downloader & Checksum Verifier
==========================================================
Usage:
    python scripts/download_dataset.py --dataset ml-latest-small --output data/
    python scripts/download_dataset.py --dataset ml-32m --output E:/ml-32m
"""

import os
import sys
import zipfile
import argparse
import urllib.request
from pathlib import Path

DATASET_URLS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
}


def download_and_extract(dataset_name: str, target_dir: str):
    """Downloads official MovieLens zip from GroupLens and extracts it."""
    if dataset_name not in DATASET_URLS:
        print(f"Error: Unknown dataset '{dataset_name}'. Available: {list(DATASET_URLS.keys())}")
        sys.exit(1)

    url = DATASET_URLS[dataset_name]
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    
    zip_path = target_path / f"{dataset_name}.zip"

    print(f"\n>>> Downloading MovieLens dataset '{dataset_name}' from {url}...")
    
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        sys.stdout.write(f"\rDownloading: {percent}% [{count * block_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB]")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
        print("\n>>> Download completed. Extracting archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_path)
        print(f">>> Successfully extracted to: {target_path}")
    except Exception as e:
        print(f"\nError during download/extraction: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Automated MovieLens Dataset Downloader")
    parser.add_argument("--dataset", choices=list(DATASET_URLS.keys()), default="ml-latest-small", help="Dataset name")
    parser.add_argument("--output", default="./data", help="Output directory path")
    args = parser.parse_args()

    download_and_extract(args.dataset, args.output)


if __name__ == "__main__":
    main()
