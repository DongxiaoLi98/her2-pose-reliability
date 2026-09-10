"""Print the shape of a downloaded benchmark table so you can fill in the mapping.

    python -m scripts.inspect_dataset data/benchmark_raw.csv
"""
from __future__ import annotations

import sys

import pandas as pd

SUGGEST = {
    "group":   ["pdb_id", "pdb_code", "target", "complex_id", "system", "id"],
    "pose":    ["file", "pose_id", "model", "rank", "prediction"],
    "rmsd":    ["rmsd", "lig_rmsd", "ligand_rmsd", "symrmsd"],
    "score":   ["score", "vina_score", "docking_score", "confidence", "affinity"],
    "clash":   ["number_clashes", "clash_count", "n_clashes"],
}


def main(path: str) -> None:
    df = pd.read_csv(path, nrows=2000)
    print(f"{path}: {len(df)} sampled rows, {df.shape[1]} columns\n")
    print("--- columns -------------------------------------------------------")
    for c in df.columns:
        nn = df[c].notna().sum()
        print(f"  {c:<50} {str(df[c].dtype):<10} non-null {nn}/{len(df)}")

    print("\n--- likely mapping targets ---------------------------------------")
    lower = {c.lower(): c for c in df.columns}
    for role, names in SUGGEST.items():
        hits = [lower[n] for n in names if n in lower]
        print(f"  {role:<8} -> {hits if hits else 'NOT FOUND — inspect manually'}")

    print("\n--- boolean columns (candidates for qc_boolean_columns) ----------")
    bools = [c for c in df.columns
             if df[c].dropna().isin([True, False, "True", "False"]).all() and df[c].notna().any()]
    for c in bools:
        print(f"  - {c}")

    print("\nNow edit config/benchmark_mapping.yaml so every entry under "
          "`columns` names a real column, then run: python run_benchmark.py")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m scripts.inspect_dataset <table.csv>")
    main(sys.argv[1])
