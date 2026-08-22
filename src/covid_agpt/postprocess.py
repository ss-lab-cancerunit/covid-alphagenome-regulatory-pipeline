"""Generic COVID AGPT batch postprocess.

This module is branch-agnostic and mode-agnostic.  It merges per-batch outputs
from screen.py/scoring.py, writes QC tables, ranks REL/ACT, and packages results.

Important distinction:
- best_per_snv: one row per input SNV is expected; dedup by variant_id is valid.
- threshold: multiple gene x track events per SNV are expected; do not dedup.
"""

from __future__ import annotations

from pathlib import Path
import tarfile

import pandas as pd


def _read_tsv_allow_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t", low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _merge_files(files: list[Path], source_col: str = "source_file") -> pd.DataFrame:
    dfs = []
    for f in files:
        df = _read_tsv_allow_empty(f)
        if df.empty:
            continue
        df[source_col] = f.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def merge_screening_outputs(
    *,
    branch: str,
    outdir: str | Path,
    expected_n_input: int | None = None,
    expected_n_batches: int | None = None,
    variant_col: str = "variant_id",
    rel_col: str = "rel",
    act_col: str = "act",
    selection_mode: str = "best_per_snv",
) -> dict[str, Path | None]:
    outdir = Path(outdir).expanduser().resolve()
    screening_dir = outdir / "screening"
    failed_dir = outdir / "failed"
    summary_dir = outdir / "summary"
    config_dir = outdir / "config"
    tables_dir = outdir / "tables"
    downloads_dir = outdir / "downloads"
    tables_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    screening_files = sorted(screening_dir.glob("*_screening.tsv"))
    failed_files = sorted(failed_dir.glob("*_failed.tsv"))
    summary_files = sorted(summary_dir.glob("*_summary.tsv"))
    config_files = sorted(config_dir.glob("*_run_config.json"))

    if len(screening_files) == 0:
        raise FileNotFoundError(f"No screening files found in {screening_dir}")

    merged = _merge_files(screening_files)
    if merged.empty:
        raise ValueError(f"Screening files exist but contain no rows: {screening_dir}")

    if rel_col in merged.columns:
        merged[rel_col] = pd.to_numeric(merged[rel_col], errors="coerce")
        merged["rel_rank_desc"] = merged[rel_col].rank(method="min", ascending=False, na_option="bottom").astype("Int64")
    if act_col in merged.columns:
        merged[act_col] = pd.to_numeric(merged[act_col], errors="coerce")
        merged["act_rank_desc"] = merged[act_col].rank(method="min", ascending=False, na_option="bottom").astype("Int64")

    sort_cols = [c for c in [rel_col, act_col] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")

    merged_path = tables_dir / f"{branch}_screening_merged.tsv"
    merged.to_csv(merged_path, sep="\t", index=False)

    dedup_path = None
    if selection_mode == "best_per_snv" and variant_col in merged.columns:
        dedup = merged.drop_duplicates(subset=[variant_col], keep="first").copy()
        dedup_path = tables_dir / f"{branch}_screening_merged_dedup.tsv"
        dedup.to_csv(dedup_path, sep="\t", index=False)

    failed_df = _merge_files(failed_files)
    failed_path = tables_dir / f"{branch}_failed_all.tsv"
    failed_df.to_csv(failed_path, sep="\t", index=False)

    summary_df = _merge_files(summary_files)
    summary_path = tables_dir / f"{branch}_summary_all.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    if "status" in merged.columns:
        status_counts = merged["status"].value_counts(dropna=False).reset_index()
        status_counts.columns = ["status", "n"]
    else:
        status_counts = pd.DataFrame([{"status": "missing_status_column", "n": len(merged)}])
    status_path = tables_dir / f"{branch}_status_counts.tsv"
    status_counts.to_csv(status_path, sep="\t", index=False)

    n_unique_variants = int(merged[variant_col].nunique()) if variant_col in merged.columns else None
    qc = {
        "branch": branch,
        "selection_mode": selection_mode,
        "n_screening_files": len(screening_files),
        "n_summary_files": len(summary_files),
        "n_failed_files": len(failed_files),
        "n_config_files": len(config_files),
        "n_merged_rows": len(merged),
        "n_unique_variants": n_unique_variants,
        "n_failed_rows": len(failed_df),
        "expected_n_input": expected_n_input,
        "expected_n_batches": expected_n_batches,
        "batch_count_match": len(screening_files) == expected_n_batches if expected_n_batches is not None else None,
    }
    if selection_mode == "best_per_snv":
        qc["input_row_count_match"] = len(merged) == expected_n_input if expected_n_input is not None else None
        qc["unique_variant_count_match"] = n_unique_variants == expected_n_input if expected_n_input is not None and n_unique_variants is not None else None
    else:
        qc["input_row_count_match"] = None
        qc["unique_variant_count_match"] = None
        qc["note"] = "threshold mode can output multiple events per SNV; merged row count is not expected to equal input SNV count."

    qc_path = tables_dir / f"{branch}_merge_qc.tsv"
    pd.DataFrame([qc]).to_csv(qc_path, sep="\t", index=False)

    tar_path = downloads_dir / f"{branch}_postprocess_results.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for p in [merged_path, dedup_path, failed_path, summary_path, qc_path, status_path]:
            if p is not None and Path(p).exists():
                tar.add(p, arcname=f"tables/{Path(p).name}")
        for p in screening_files:
            tar.add(p, arcname=f"screening/{p.name}")
        for p in summary_files:
            tar.add(p, arcname=f"summary/{p.name}")
        for p in failed_files:
            tar.add(p, arcname=f"failed/{p.name}")
        for p in config_files:
            tar.add(p, arcname=f"config/{p.name}")

    return {
        "merged_path": merged_path,
        "dedup_path": dedup_path,
        "failed_path": failed_path,
        "summary_path": summary_path,
        "qc_path": qc_path,
        "status_path": status_path,
        "tar_path": tar_path,
    }
