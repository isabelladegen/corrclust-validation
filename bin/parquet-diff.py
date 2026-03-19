#!/usr/bin/env python
import sys
import pandas as pd

"""Checks for content differences of .parquet files not encoding differences"""
f1 = sys.argv[2]  # old-file
f2 = sys.argv[5]  # new-file

df1 = pd.read_parquet(f1)
df2 = pd.read_parquet(f2)

diffs = []
checks_passed = []

if df1.shape != df2.shape:
    diffs.append(f"Shape: {df1.shape} vs {df2.shape}")
    print(f"\n{sys.argv[1]}:")
    print("\n".join(diffs))
    sys.exit(0)
checks_passed.append(f"shape: {df1.shape}")

if list(df1.columns) != list(df2.columns):
    diffs.append(f"Columns: {list(df1.columns)} vs {list(df2.columns)}")
    print(f"\n{sys.argv[1]}:")
    print("\n".join(diffs))
    sys.exit(0)
checks_passed.append(f"columns: {list(df1.columns)}")

dtype_ok = True
if not (df1.dtypes == df2.dtypes).all():
    dtype_ok = False
    for c in df1.columns:
        if df1[c].dtype != df2[c].dtype:
            diffs.append(f"Dtype {c}: {df1[c].dtype} vs {df2[c].dtype}")
if dtype_ok:
    checks_passed.append("dtypes: all match")

values_checked = 0
for c in df1.columns:
    if c not in df2.columns:
        continue
    if df1.shape[0] != df2.shape[0]:
        continue
    s1 = df1[c].reset_index(drop=True)
    s2 = df2[c].reset_index(drop=True)
    values_checked += len(s1)
    if pd.api.types.is_float_dtype(df1[c]):
        max_diff = (s1 - s2).abs().max()
        if max_diff > 0:
            diffs.append(f"{c}: max abs diff = {max_diff}")
    else:
        mismatches = (s1 != s2).sum()
        if mismatches > 0:
            diffs.append(f"{c}: {mismatches} mismatched values")
            mask = s1 != s2
            for i in mask[mask].index[:3]:
                diffs.append(f"  row {i}: '{s1[i]}' -> '{s2[i]}'")

if not df1.index.equals(df2.index):
    diffs.append("Index differs")

if not diffs:
    checks_passed.append(f"values: {values_checked} cells compared, all identical")

print(f"\n{sys.argv[1]}:")
if diffs:
    print("\n".join(diffs))
else:
    print("\n".join(checks_passed))

sys.exit(0)