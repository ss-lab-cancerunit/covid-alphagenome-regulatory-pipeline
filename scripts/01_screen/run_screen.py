#!/usr/bin/env python3
"""Thin CLI wrapper for covid_agpt.screen.run_best_snv_screen."""

from __future__ import annotations

import argparse
from pathlib import Path

from covid_agpt.screen import run_best_snv_screen


def _comma_list(x: str | None):
    if x is None:
        return None
    return [v.strip() for v in str(x).split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one COVID AGPT RNA screening batch.")
    p.add_argument("--branch", required=True)
    p.add_argument("--mode", required=True, choices=["cd", "lnc"])
    p.add_argument("--input", "--input-path", dest="input_path", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--fasta-path", required=True)
    p.add_argument("--gtf-path", required=True)
    p.add_argument("--polya-gtf-path", default=None)
    p.add_argument("--track-meta-path", required=True)

    p.add_argument("--batch-id", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--run-label", default=None)
    p.add_argument("--max-snvs", type=int, default=None)
    p.add_argument("--window-size", type=int, default=1_048_576)
    p.add_argument("--rna-resolution", type=int, default=128)
    p.add_argument("--rel-eps", type=float, default=1e-3)

    p.add_argument("--allowed-gene-types", default=None,
                   help="Comma-separated override. Default: cd=protein_coding, lnc=lncRNA.")

    p.add_argument("--selection-mode", choices=["best_per_snv", "threshold"], default="best_per_snv")
    p.add_argument("--track-policy", choices=["auto", "legacy_all_rna", "cd_polya", "lnc_total"], default="auto")
    p.add_argument("--rel-cutoff", type=float, default=None)
    p.add_argument("--act-cutoff", type=float, default=None)
    p.add_argument("--allow-unstranded-tracks", dest="allow_unstranded_tracks",
                   action="store_true", default=True,
                   help="Treat '.'/unstranded RNA tracks as compatible. This is the default.")
    p.add_argument("--no-allow-unstranded-tracks", dest="allow_unstranded_tracks",
                   action="store_false",
                   help="Exclude unstranded tracks; only exact +/+ or -/- strand matches are scored.")

    p.add_argument("--device", default="cuda")
    p.add_argument("--full-float32", action="store_true")
    p.add_argument("--write-all-events", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    allow_unstranded = bool(args.allow_unstranded_tracks)

    run_best_snv_screen(
        branch=args.branch,
        mode=args.mode,
        input_path=args.input_path,
        outdir=args.outdir,
        model_path=args.model_path,
        fasta_path=args.fasta_path,
        gtf_path=args.gtf_path,
        polya_gtf_path=args.polya_gtf_path,
        track_meta_path=args.track_meta_path,
        batch_id=args.batch_id,
        batch_size=args.batch_size,
        run_label=args.run_label,
        max_snvs=args.max_snvs,
        window_size=args.window_size,
        rna_resolution=args.rna_resolution,
        rel_eps=args.rel_eps,
        allowed_gene_types=_comma_list(args.allowed_gene_types),
        selection_mode=args.selection_mode,
        track_policy=args.track_policy,
        rel_cutoff=args.rel_cutoff,
        act_cutoff=args.act_cutoff,
        allow_unstranded_tracks=allow_unstranded,
        device=args.device,
        full_float32=args.full_float32,
        write_all_events=args.write_all_events,
        resume=args.resume,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
