"""
inspect_parquet.py — verify column schema before training.

Usage:
    python inspect_parquet.py                        # checks all three splits
    python inspect_parquet.py --path data/raw/scienceQA_train.parquet
"""

import argparse
import os
import pandas as pd


def inspect(path: str):
    print(f"\n{'─' * 60}")
    print(f"File : {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    df = pd.read_parquet(path)
    print(f"Rows : {len(df)}")
    print(f"Cols : {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes}\n")

    row = df.iloc[0]
    print("First row:")
    for col in df.columns:
        val = row[col]
        if isinstance(val, (bytes, bytearray)):
            print(f"  {col}: <bytes, len={len(val)}>")
        elif isinstance(val, dict):
            print(f"  {col}: dict with keys {list(val.keys())}")
        elif hasattr(val, "__len__") and not isinstance(val, str) and len(str(val)) > 100:
            print(f"  {col}: <{type(val).__name__}, len={len(val)}>")
        else:
            print(f"  {col}: {val!r}")

    if "image" in df.columns:
        null_frac = df["image"].isna().mean()
        print(f"\nimage null fraction: {null_frac:.3%}  "
              f"({'expected ~51%' if 0.4 < null_frac < 0.65 else 'unusual — check'})")

    if "answer" in df.columns:
        print(f"answer range: {df['answer'].min()} – {df['answer'].max()}")
    else:
        print("answer column: absent (expected for test set)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None,
                        help="Single parquet file to inspect (default: all three splits)")
    args = parser.parse_args()

    if args.path:
        inspect(args.path)
    else:
        for fname in [
            "data/raw/scienceQA_train.parquet",
            "data/raw/scienceQA_val.parquet",
            "data/raw/scienceQA_test_set.parquet",
        ]:
            if os.path.exists(fname):
                inspect(fname)
            else:
                print(f"\n[missing] {fname}")