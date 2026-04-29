"""
inspect_parquet.py

Usage:
    python inspect_parquet.py
    python inspect_parquet.py --path data/raw/scienceQA_train.parquet
"""

import argparse
import os
import pandas as pd


def inspect(path: str):
    df = pd.read_parquet(path)
    print(f"\n{path}  ({os.path.getsize(path) / 1e6:.0f} MB)")
    print(f"rows={len(df)}  cols={list(df.columns)}")
    print(df.dtypes.to_string())

    row = df.iloc[0]
    print("\nFirst row:")
    for col in df.columns:
        val = row[col]
        if isinstance(val, (bytes, bytearray)):
            print(f"  {col}: <bytes len={len(val)}>")
        elif isinstance(val, dict):
            print(f"  {col}: dict{list(val.keys())}")
        elif hasattr(val, "__len__") and not isinstance(val, str) and len(str(val)) > 80:
            print(f"  {col}: <{type(val).__name__} len={len(val)}>")
        else:
            print(f"  {col}: {val!r}")

    if "image" in df.columns:
        print(f"\nimage null fraction: {df['image'].isna().mean():.1%}")
    if "answer" in df.columns:
        print(f"answer range: {df['answer'].min()}–{df['answer'].max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None)
    args = parser.parse_args()

    if args.path:
        inspect(args.path)
    else:
        for f in ["data/raw/scienceQA_train.parquet",
                  "data/raw/scienceQA_val.parquet",
                  "data/raw/scienceQA_test_set.parquet"]:
            if os.path.exists(f):
                inspect(f)
            else:
                print(f"\n[missing] {f}")