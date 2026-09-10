"""Freeze a benchmark table into contract form, ready to upload to S3.

    python -m scripts.export_contract_table \
        --table data/benchmark_raw.csv \
        --mapping config/benchmark_mapping_local.yaml \
        --out data/pose_contract_table.csv

WHY THIS EXISTS
---------------
The benchmark loader needs a mapping file to interpret a raw tool output. A
SageMaker pipeline step should not have to carry that mapping around: the
mapping is a local, dataset-specific concern, while the pipeline consumes the
contract. So we apply the mapping once, here, and hand the pipeline a table that
already satisfies the contract.

The provenance record travels alongside the table so the substitutions made
during mapping are still visible from the cloud run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src import io_utils
from src.benchmark_loader import load_benchmark


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--mapping", default="config/benchmark_mapping_local.yaml")
    ap.add_argument("--out", default="data/pose_contract_table.csv")
    a = ap.parse_args()

    df, provenance = load_benchmark(a.mapping, table_path=a.table)
    io_utils.write_table(df, a.out)
    io_utils.write_json(provenance, str(Path(a.out).with_suffix(".provenance.json")))

    print(f"wrote {a.out}")
    print(f"  rows        {len(df)}")
    print(f"  structures  {df['structure_id'].nunique()}")
    print(f"  labels      {df['label'].value_counts(dropna=False).to_dict()}")
    print(f"  failed rows {int((df['runtime_status'] == 'failed').sum())}")
    print("\nsubstitutions recorded in the provenance file:")
    for s in provenance.get("substitutions", []):
        print(f"  - {s}")
    print(f"\nupload with:\n  aws s3 cp {a.out} s3://YOUR-BUCKET/her2/pose_contract_table.csv")


if __name__ == "__main__":
    main()
