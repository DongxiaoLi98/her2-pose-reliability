"""Thin storage abstraction.

Every step reads and writes through these two functions. Locally they hit the
filesystem; given an s3:// URI they hit S3 (pandas + s3fs / fsspec). Nothing
else in the codebase knows where the bytes live -- that is what makes the same
code run on a laptop and inside a SageMaker job unchanged.
"""
from __future__ import annotations

import json
import os

import pandas as pd


def _is_s3(uri: str) -> bool:
    return str(uri).startswith("s3://")


def _ensure_parent(uri: str) -> None:
    if not _is_s3(uri):
        os.makedirs(os.path.dirname(os.path.abspath(uri)), exist_ok=True)


def write_table(df: pd.DataFrame, uri: str) -> str:
    _ensure_parent(uri)
    out = df.copy()
    # list-valued columns must survive a CSV round trip
    for col in out.columns:
        if out[col].map(lambda v: isinstance(v, list)).any():
            out[col] = out[col].map(json.dumps)
    out.to_csv(uri, index=False)
    return uri


def read_table(uri: str, list_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    df = pd.read_csv(uri)
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].map(lambda v: json.loads(v) if isinstance(v, str) else [])
    return df


def write_json(obj: dict, uri: str) -> str:
    _ensure_parent(uri)
    if _is_s3(uri):
        import fsspec

        with fsspec.open(uri, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
    else:
        with open(uri, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
    return uri


def read_json(uri: str) -> dict:
    if _is_s3(uri):
        import fsspec

        with fsspec.open(uri, "r") as fh:
            return json.load(fh)
    with open(uri) as fh:
        return json.load(fh)


def write_bytes(data: bytes, uri: str) -> str:
    _ensure_parent(uri)
    if _is_s3(uri):
        import fsspec

        with fsspec.open(uri, "wb") as fh:
            fh.write(data)
    else:
        with open(uri, "wb") as fh:
            fh.write(data)
    return uri


def read_bytes(uri: str) -> bytes:
    if _is_s3(uri):
        import fsspec

        with fsspec.open(uri, "rb") as fh:
            return fh.read()
    with open(uri, "rb") as fh:
        return fh.read()
