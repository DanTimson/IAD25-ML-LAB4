"""
download_data.py — fetch competition files from Kaggle into data/raw/.

Requires ~/.kaggle/kaggle.json (kaggle.com/settings → API → Create New Token).

Usage:
    python download_data.py
    python download_data.py --dest data/raw
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
        sys.exit("kagglehub not installed: pip install kagglehub")

    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        sys.exit("Kaggle credentials not found at ~/.kaggle/kaggle.json")

    print(f"Downloading {COMPETITION} ...")
    raw_path = kagglehub.competition_download(COMPETITION)
    os.makedirs(dest, exist_ok=True)

    if os.path.isdir(raw_path):
        for fname in os.listdir(raw_path):
            shutil.copy2(os.path.join(raw_path, fname), os.path.join(dest, fname))
            print(f"  {fname}")
    else:
        fname = os.path.basename(raw_path)
        shutil.copy2(raw_path, os.path.join(dest, fname))
        print(f"  {fname}")

    missing = [f for f in EXPECTED_FILES if not os.path.exists(os.path.join(dest, f))]
    if missing:
        print(f"WARNING: expected files not found: {missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/raw")
    download(parser.parse_args().dest)