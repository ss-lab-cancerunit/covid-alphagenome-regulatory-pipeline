#!/usr/bin/env python3
"""Thin CLI wrapper for covid_agpt.postprocess.merge_screening_outputs."""

from __future__ import annotations

import argparse

from covid_agpt.postprocess import merge_screening_outputs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge COVID AGPT screening batch outputs.")
    p.add_argument("--branch", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--expected-n-input", type=int, default=None)
    p.add_argument("--expected-n-batches", type=int, default=None)
    p.add_argument("--variant-col", default="variant_id")
    p.add_argument("--rel-col", default="rel")
    p.add_argument("--act-col", default="act")
    p.add_argument("--selection-mode", choices=["best_per_snv", "threshold"], default="best_per_snv")
    return p


def main() -> None:
    args = build_parser().parse_args()
    result = merge_screening_outputs(
        branch=args.branch,
        outdir=args.outdir,
        expected_n_input=args.expected_n_input,
        expected_n_batches=args.expected_n_batches,
        variant_col=args.variant_col,
        rel_col=args.rel_col,
        act_col=args.act_col,
        selection_mode=args.selection_mode,
    )
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
