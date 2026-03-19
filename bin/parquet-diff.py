#!/usr/bin/env python
import sys
import pandas as pd

"""Checks for content differences of .parquet files not encoding differences"""
f1 = sys.argv[2]  # old-file
f2 = sys.argv[5]  # new-file

df1 = pd.read_parquet(f1)
df2 = pd.read_parquet(f2)

diffs = []

if df1.shape != df2.shape:
    diffs.append(f"Shape: {df1.shape} vs {df2.shape}")

if list(df1.columns) != list(df2.columns):
    diffs.append(f"Columns: {list(df1.columns)} vs {list(df2.columns)}")

if not (df1.dtypes == df2.dtypes).all():
    for c in df1.columns:
        if df1[c].dtype != df2[c].dtype:
            diffs.append(f"Dtype {c}: {df1[c].dtype} vs {df2[c].dtype}")

for c in df1.columns:
    if c not in df2.columns:
        continue
    if pd.api.types.is_float_dtype(df1[c]):
        max_diff = (df1[c] - df2[c]).abs().max()
        if max_diff > 0:
            diffs.append(f"{c}: max abs diff = {max_diff}")
    else:
        mismatches = (df1[c] != df2[c]).sum()
        if mismatches > 0:
            diffs.append(f"{c}: {mismatches} mismatched values")

if not df1.index.equals(df2.index):
    diffs.append("Index differs")

if diffs:
    print("\n".join(diffs), file=sys.stderr)
    sys.exit(1)

sys.exit(0)