from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..schema.models import ImportResult, RawSheet
from .excel_reader import build_raw_sheet, find_sheet_names, read_sheet_raw


def _cache_files(cache_dir: Path, sheet_key: str) -> tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    return cache_dir / f"{sheet_key}.parquet", cache_dir / f"{sheet_key}.meta.json"


def _source_fingerprint(path: Path) -> dict:
    stat = Path(path).stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _is_cache_fresh(source_path: Path, meta_file: Path, parquet_file: Path) -> bool:
    if not parquet_file.exists() or not meta_file.exists():
        return False
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    return meta.get("source_fingerprint") == _source_fingerprint(source_path)


def _load_or_build_sheet(
    source_path: Path,
    sheet_name: str,
    sheet_key: str,
    domain_hint: str,
    cache_dir: Path,
    force_refresh: bool,
) -> RawSheet:
    parquet_file, meta_file = _cache_files(cache_dir, sheet_key)

    if not force_refresh and _is_cache_fresh(source_path, meta_file, parquet_file):
        df = pd.read_parquet(parquet_file)
        return build_raw_sheet(sheet_name, df, domain_hint, downcast_measurements=False)

    df = read_sheet_raw(source_path, sheet_name)
    sheet = build_raw_sheet(sheet_name, df, domain_hint)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sheet.dataframe.to_parquet(parquet_file, index=False)
    meta_file.write_text(
        json.dumps({"source_fingerprint": _source_fingerprint(source_path)}),
        encoding="utf-8",
    )
    return sheet


def read_workbook_cached(path: Path, cache_dir: Path, *, force_refresh: bool = False) -> ImportResult:
    path = Path(path)
    harm_name, inter_name = find_sheet_names(path)

    harmonicas = _load_or_build_sheet(
        path, harm_name, "harmonicas", "harmonica", cache_dir, force_refresh
    )
    interharmonicas = _load_or_build_sheet(
        path, inter_name, "interharmonicas", "interharmonica", cache_dir, force_refresh
    )

    unknown = {}
    for sheet in (harmonicas, interharmonicas):
        unknown_cols = [c for c, spec in sheet.columns.items() if spec.kind == "unknown"]
        if unknown_cols:
            unknown[sheet.name] = unknown_cols

    return ImportResult(
        source_path=path,
        imported_at=datetime.now(),
        harmonicas=harmonicas,
        interharmonicas=interharmonicas,
        unknown_columns=unknown,
    )
