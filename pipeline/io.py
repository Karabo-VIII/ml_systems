"""Atomic parquet writes. The file on disk is the previous version or a finished one."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import polars as pl
import pyarrow.parquet as pq

from pipeline.constants import CONTRACT_VERSION, PRODUCER


def params_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def atomic_write_json(path: Path, payload: dict) -> Path:
    data = (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode()
    return atomic_write_bytes(path, data)


def atomic_write_parquet(
    df: pl.DataFrame,
    path: Path | str,
    *,
    schema_name: str,
    params: dict,
    required_cols: Iterable[str] | None = None,
    sort_cols: list[str] | None = None,
) -> Path:
    """Write parquet via temp file, fsync, and rename.

    ``built_at`` is not stored in the parquet. A second write of the same
    frame and params is byte-identical. ``built_at`` belongs in the JSON
    manifest beside the file.
    """
    path = Path(path)
    if sort_cols:
        df = df.sort(sort_cols)
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} missing required columns: {missing}")

    meta_params = params_hash(params)
    table = df.to_arrow()
    meta = dict(table.schema.metadata or {})
    meta[b"contract_version"] = CONTRACT_VERSION.encode()
    meta[b"schema_name"] = schema_name.encode()
    meta[b"producer"] = PRODUCER.encode()
    meta[b"params_hash"] = meta_params.encode()
    table = table.replace_schema_metadata(meta)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        pq.write_table(table, tmp, compression="zstd", compression_level=3)
        with open(tmp, "rb+") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise

    sidecar = path.with_suffix(path.suffix + ".sha256")
    atomic_write_bytes(sidecar, (file_sha256(path) + "\n").encode())
    return path


def read_parquet_metadata(path: Path) -> dict[str, str]:
    schema = pq.read_schema(path)
    raw = schema.metadata or {}
    out: dict[str, str] = {}
    for key in (b"contract_version", b"schema_name", b"producer", b"params_hash"):
        if key in raw:
            out[key.decode()] = raw[key].decode()
    return out
