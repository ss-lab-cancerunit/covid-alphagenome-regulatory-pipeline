#!/usr/bin/env python3
"""Learn one-dimensional K-means boundaries for postprocessed AGPT tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from covid_agpt.threshold_learning import fit_1d_kmeans, group_statistics


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/table.tsv")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/table.tsv")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset,
        required=True,
        help="Repeatable NAME=/path/to/postprocessed.tsv argument",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Metric column; repeat for multiple metrics (default: rel)",
    )
    parser.add_argument(
        "--backend",
        choices=("deterministic", "sklearn"),
        required=True,
    )
    parser.add_argument("--k", nargs="+", type=int, default=(3, 4, 5))
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    metrics = args.metric or ["rel"]
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    run_config = {
        "backend": args.backend,
        "k_values": args.k,
        "metrics": metrics,
        "datasets": {name: str(path.resolve()) for name, path in args.dataset},
    }
    (args.outdir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )

    for dataset_name, table_path in args.dataset:
        frame = pd.read_csv(table_path, sep="\t", low_memory=False)
        for metric in metrics:
            if metric not in frame.columns:
                raise KeyError(f"{metric!r} is absent from {table_path}")
            for k in args.k:
                result = fit_1d_kmeans(frame[metric], k, args.backend)
                row = {
                    "dataset": dataset_name,
                    "metric": metric,
                    "backend": args.backend,
                    "k": k,
                    "n": int(frame[metric].notna().sum()),
                    "wcss": result.wcss,
                    "centers": ";".join(f"{x:.17g}" for x in result.centers),
                    "cutoffs": ";".join(f"{x:.17g}" for x in result.cutoffs),
                }
                for index, cutoff in enumerate(result.cutoffs, start=1):
                    row[f"cutoff_{index}"] = float(cutoff)
                summary_rows.append(row)

                stats = group_statistics(frame[metric], result)
                stats.insert(0, "k", k)
                stats.insert(0, "backend", args.backend)
                stats.insert(0, "metric", metric)
                stats.insert(0, "dataset", dataset_name)
                stats.to_csv(
                    args.outdir / f"{dataset_name}_{metric}_k{k}_group_stats.tsv",
                    sep="\t",
                    index=False,
                )

    pd.DataFrame(summary_rows).to_csv(
        args.outdir / "threshold_learning_summary.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()
