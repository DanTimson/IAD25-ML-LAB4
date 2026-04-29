"""
download_data.py — download competition files from Kaggle and place them
under data/raw/ so the rest of the codebase can find them by convention.

Requirements:
    pip install kagglehub --break-system-packages

Kaggle credentials (~/.kaggle/kaggle.json) must be present.
Create them at https://www.kaggle.com/settings → API → Create New Token.

Usage:
    python download_data.py
    python download_data.py --dest data/raw   # override destination
"""

import argparse
import os
import shutil
import sys


COMPETITION = "dm-2026-ml-lab-4-vqa"

EXPECTED_FILES = [
    "scienceQA_train.parquet",
    "scienceQA_val.parquet",
    "scienceQA_test_set.parquet",
]


def download(dest: str):
    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "kagglehub not installed.\n"
            "Run: pip install kagglehub --break-system-packages"
        )

    creds = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(creds):
        sys.exit(
            "Kaggle credentials not found at ~/.kaggle/kaggle.json.\n"
            "Go to https://www.kaggle.com/settings → API → Create New Token."
        )

    print(f"Downloading competition '{COMPETITION}' …")
    raw_path = kagglehub.competition_download(COMPETITION)
    print(f"Downloaded to: {raw_path}")

    os.makedirs(dest, exist_ok=True)

    # kagglehub may return a directory or a single file path
    if os.path.isdir(raw_path):
        for fname in os.listdir(raw_path):
            src = os.path.join(raw_path, fname)
            dst = os.path.join(dest, fname)
            shutil.copy2(src, dst)
            print(f"  copied {fname} → {dst}")
    else:
        fname = os.path.basename(raw_path)
        dst   = os.path.join(dest, fname)
        shutil.copy2(raw_path, dst)
        print(f"  copied {fname} → {dst}")

    # Verify expected files are present
    missing = [f for f in EXPECTED_FILES if not os.path.exists(os.path.join(dest, f))]
    if missing:
        print(f"\nWARNING: expected files not found in {dest}: {missing}")
        print("Files present:", os.listdir(dest))
    else:
        print(f"\nAll expected files present in {dest}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/raw",
                        help="Directory to place the parquet files (default: data/raw)")
    args = parser.parse_args()
    download(args.dest)
