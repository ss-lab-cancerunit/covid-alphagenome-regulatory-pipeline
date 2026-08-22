"""
COVID AGPT RNA screen.

screen.py is the inference/data-extraction layer:
- split SNV input into batches
- build 1Mb REF/ALT sequences
- run REF forward -> move outputs to CPU RAM -> clear GPU
- run ALT forward -> move outputs to CPU RAM -> clear GPU
- extract RNA_SEQ tensors
- pass RNA_SEQ tensors to covid_agpt.scoring

Biological event interpretation is intentionally delegated to scoring.py.
"""

from __future__ import annotations

import contextlib
import gc
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from safetensors.torch import load_file as load_safetensors

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.variant_scoring import GeneAnnotation, Interval, OutputType, Variant
from alphagenome_pytorch.variant_scoring.aggregations import align_alternate

from covid_agpt.scoring import ScoringConfig, load_track_metadata, score_snv_rna_seq_tensors


ORGANISM_INDEX = 0  # human
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


# ---------------------------------------------------------------------
# Basic helpers, copied from validated AGPT screening scripts
# ---------------------------------------------------------------------


def expand_path(x: str | Path | None) -> Path | None:
    if x is None:
        return None
    return Path(x).expanduser().resolve()


def safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def normalize_chrom(chrom: Any) -> str:
    chrom = str(chrom).strip()
    if not chrom.lower().startswith("chr"):
        chrom = "chr" + chrom
    return chrom


def parse_variant_string(v: str) -> tuple[str, int, str, str]:
    v = str(v).strip()
    if ">" in v and ":" in v:
        chrom, pos, alleles = v.split(":")
        ref, alt = alleles.split(">")
        return normalize_chrom(chrom), int(pos), ref.upper(), alt.upper()
    if v.count(":") == 3:
        chrom, pos, ref, alt = v.split(":")
        return normalize_chrom(chrom), int(pos), ref.upper(), alt.upper()
    parts = re.split(r"[_:\s]+", v)
    if len(parts) >= 4:
        chrom, pos, ref, alt = parts[:4]
        return normalize_chrom(chrom), int(pos), ref.upper(), alt.upper()
    raise ValueError(f"Cannot parse variant string: {v}")


def parse_variant_row(row: pd.Series) -> tuple[str, int, str, str, str]:
    cols = set(row.index)
    chr_col = "chr" if "chr" in cols else "chrom" if "chrom" in cols else None
    if chr_col is not None and {"pos", "ref", "alt"}.issubset(cols):
        chrom = normalize_chrom(row[chr_col])
        pos = int(row["pos"])
        ref = str(row["ref"]).upper().strip()
        alt = str(row["alt"]).upper().strip()
        variant_id = str(row["variant_id"]).strip() if "variant_id" in cols else f"{chrom}:{pos}:{ref}:{alt}"
        return chrom, pos, ref, alt, variant_id
    for c in ["variant_id", "variant"]:
        if c in cols:
            chrom, pos, ref, alt = parse_variant_string(row[c])
            return chrom, pos, ref, alt, str(row[c]).strip()
    raise ValueError("Input must contain variant_id/variant or chr|chrom,pos,ref,alt columns")


def onehot(seq: str) -> torch.Tensor:
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq.upper()):
        j = BASE_TO_IDX.get(b)
        if j is not None:
            arr[i, j] = 1.0
    return torch.from_numpy(arr).unsqueeze(0)


def to_cpu_tree(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: to_cpu_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_cpu_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(to_cpu_tree(v) for v in x)
    return x


def tensor_bytes_tree(x: Any) -> int:
    if torch.is_tensor(x):
        return x.numel() * x.element_size()
    if isinstance(x, dict):
        return sum(tensor_bytes_tree(v) for v in x.values())
    if isinstance(x, list):
        return sum(tensor_bytes_tree(v) for v in x)
    if isinstance(x, tuple):
        return sum(tensor_bytes_tree(v) for v in x)
    return 0


def append_tsv(df: pd.DataFrame | None, path: Path) -> None:
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", mode="a", header=not path.exists(), index=False)


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def choose_fasta_chrom(genome: Fasta, chrom: str) -> str:
    if chrom in genome:
        return chrom
    no_chr = chrom[3:] if chrom.startswith("chr") else chrom
    if no_chr in genome:
        return no_chr
    with_chr = normalize_chrom(chrom)
    if with_chr in genome:
        return with_chr
    raise KeyError(f"Chromosome not found in FASTA: {chrom}. Example keys: {list(genome.keys())[:5]}")


# ---------------------------------------------------------------------
# AGPT runtime helpers, copied from validated scripts
# ---------------------------------------------------------------------


def load_model(model_path: Path, device: str, full_float32: bool = False) -> AlphaGenome:
    print("Loading AlphaGenome model...", flush=True)
    dtype_policy = DtypePolicy.full_float32() if full_float32 else DtypePolicy.mixed_precision()
    model = AlphaGenome(num_organisms=2, dtype_policy=dtype_policy)

    if model_path.suffix.lower() == ".safetensors":
        state_dict = load_safetensors(str(model_path))
    else:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded.", flush=True)
    return model


def run_forward_to_ram(model: AlphaGenome, seq: str, device: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Forward one sequence, immediately move outputs to CPU RAM, then clear GPU tensors."""
    x = onehot(seq).to(device)
    organism = torch.tensor([ORGANISM_INDEX], dtype=torch.long, device=device)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        amp_context = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        amp_context = contextlib.nullcontext()

    print(f"Running {label} forward...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad(), amp_context:
        outputs = model(x, organism, return_embeddings=False)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    peak_alloc = None
    peak_reserved = None
    if device == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

    print(f"{label} forward sec: {t1 - t0:.2f}", flush=True)
    if peak_alloc is not None:
        print(f"{label} peak allocated GB: {peak_alloc:.2f}", flush=True)
        print(f"{label} peak reserved GB: {peak_reserved:.2f}", flush=True)

    print(f"Moving {label} outputs to CPU RAM...", flush=True)
    outputs_cpu = to_cpu_tree(outputs)
    cpu_gb = tensor_bytes_tree(outputs_cpu) / 1024**3
    print(f"{label} CPU cache tensor size GB: {cpu_gb:.2f}", flush=True)

    del outputs, x, organism
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return outputs_cpu, {
        f"{label.lower()}_forward_sec": t1 - t0,
        f"{label.lower()}_peak_allocated_GB": peak_alloc,
        f"{label.lower()}_peak_reserved_GB": peak_reserved,
        f"{label.lower()}_cpu_cache_tensor_GB": cpu_gb,
    }


def _get_prediction(outputs: dict[str, Any], output_type: OutputType, resolution: int = 128) -> torch.Tensor:
    key = output_type.value
    if key not in outputs:
        raise KeyError(f"Missing output key '{key}' in model outputs. Available: {list(outputs.keys())}")
    x = outputs[key]
    if isinstance(x, dict):
        if resolution in x:
            return x[resolution]
        if 128 in x:
            return x[128]
        if 1 in x:
            return x[1]
        raise KeyError(f"No usable resolution found for {key}. Available resolutions: {list(x.keys())}")
    return x


def _ensure_bst(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"Expected tensor shape (B,S,T) or (S,T), got {tuple(x.shape)}")
    return x


def extract_rna_seq_tensor(outputs: dict[str, Any], resolution: int = 128) -> torch.Tensor:
    return _ensure_bst(_get_prediction(outputs, OutputType.RNA_SEQ, resolution))


def maybe_align_alt_rna_for_variant(alt_rna: torch.Tensor, variant: Variant, interval: Interval) -> torch.Tensor:
    if getattr(variant, "is_indel", False):
        return align_alternate(
            alt_rna.squeeze(0),
            variant.start,
            len(variant.reference_bases),
            len(variant.alternate_bases),
            interval.start,
        ).unsqueeze(0)
    return alt_rna


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------


def run_best_snv_screen(
    branch: str,
    mode: str,
    input_path: str | Path,
    outdir: str | Path,
    model_path: str | Path,
    fasta_path: str | Path,
    gtf_path: str | Path,
    polya_gtf_path: str | Path | None = None,
    track_meta_path: str | Path | None = None,
    batch_id: int = 0,
    batch_size: int = 50,
    run_label: str | None = None,
    max_snvs: int | None = None,
    window_size: int = 1_048_576,
    rna_resolution: int = 128,
    rel_eps: float = 1e-3,
    allowed_gene_types: Iterable[str] | None = None,
    selection_mode: str = "best_per_snv",
    track_policy: str = "auto",
    rel_cutoff: float | None = None,
    act_cutoff: float | None = None,
    allow_unstranded_tracks: bool = True,
    device: str = "cuda",
    full_float32: bool = False,
    write_all_events: bool = False,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    if mode not in {"cd", "lnc"}:
        raise ValueError("mode must be 'cd' or 'lnc'")
    if selection_mode not in {"best_per_snv", "threshold"}:
        raise ValueError("selection_mode must be best_per_snv or threshold")

    input_path = expand_path(input_path)
    outdir = expand_path(outdir)
    model_path = expand_path(model_path)
    fasta_path = expand_path(fasta_path)
    gtf_path = expand_path(gtf_path)
    polya_gtf_path = expand_path(polya_gtf_path) if polya_gtf_path else None
    track_meta_path = expand_path(track_meta_path) if track_meta_path else None

    assert input_path is not None and outdir is not None
    assert model_path is not None and fasta_path is not None and gtf_path is not None

    allowed_tuple = None
    if allowed_gene_types is not None:
        allowed_tuple = tuple(str(x).strip() for x in allowed_gene_types if str(x).strip())

    scoring_config = ScoringConfig(
        mode=mode,  # type: ignore[arg-type]
        selection_mode=selection_mode,  # type: ignore[arg-type]
        track_policy=track_policy,  # type: ignore[arg-type]
        rel_eps=float(rel_eps),
        rel_cutoff=rel_cutoff,
        act_cutoff=act_cutoff,
        allowed_gene_types=allowed_tuple,
        allow_unstranded_tracks=bool(allow_unstranded_tracks),
        require_strand_compatible=True,
    )

    screening_dir = outdir / "screening"
    raw_dir = outdir / "raw_events"
    summary_dir = outdir / "summary"
    failed_dir = outdir / "failed"
    config_dir = outdir / "config"
    for d in [screening_dir, raw_dir, summary_dir, failed_dir, config_dir]:
        d.mkdir(parents=True, exist_ok=True)

    batch_label = safe_name(run_label or f"batch_{int(batch_id):04d}")
    screening_path = screening_dir / f"{batch_label}_screening.tsv"
    raw_path = raw_dir / f"{batch_label}_all_valid_gene_track_events.tsv"
    summary_path = summary_dir / f"{batch_label}_summary.tsv"
    failed_path = failed_dir / f"{batch_label}_failed.tsv"
    config_path = config_dir / f"{batch_label}_run_config.json"

    if resume and screening_path.exists():
        print(f"Resume mode: screening output exists, skipping: {screening_path}", flush=True)
        return {"screening_path": screening_path, "summary_path": summary_path, "failed_path": failed_path, "raw_path": raw_path, "config_path": config_path}

    if overwrite:
        for p in [screening_path, raw_path, summary_path, failed_path, config_path]:
            if p.exists():
                p.unlink()
    else:
        if screening_path.exists() or summary_path.exists() or failed_path.exists():
            raise FileExistsError(f"Output exists for {batch_label}. Use overwrite=True or resume=True.")

    print("=" * 90, flush=True)
    print("COVID AGPT RNA screen: screen.py -> scoring.py", flush=True)
    print("=" * 90, flush=True)
    print(f"branch: {branch}", flush=True)
    print(f"mode: {mode}", flush=True)
    print(f"selection_mode: {selection_mode}", flush=True)
    print(f"track_policy: {track_policy} -> {scoring_config.resolved_track_policy()}", flush=True)
    print(f"input_path: {input_path}", flush=True)
    print(f"outdir: {outdir}", flush=True)
    print(f"batch_id: {batch_id}", flush=True)
    print(f"batch_size: {batch_size}", flush=True)

    for p in [input_path, model_path, fasta_path, gtf_path, track_meta_path]:
        if p is not None and not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")
    if polya_gtf_path is not None and not polya_gtf_path.exists():
        raise FileNotFoundError(f"Missing polyA GTF: {polya_gtf_path}")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU memory GB: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}", flush=True)

    run_config = {
        "branch": branch,
        "mode": mode,
        "selection_mode": selection_mode,
        "track_policy": track_policy,
        "resolved_track_policy": scoring_config.resolved_track_policy(),
        "input_path": str(input_path),
        "outdir": str(outdir),
        "model_path": str(model_path),
        "fasta_path": str(fasta_path),
        "gtf_path": str(gtf_path),
        "polya_gtf_path": str(polya_gtf_path) if polya_gtf_path else None,
        "track_meta_path": str(track_meta_path) if track_meta_path else None,
        "batch_id": batch_id,
        "batch_size": batch_size,
        "run_label": batch_label,
        "max_snvs": max_snvs,
        "window_size": window_size,
        "rna_resolution": rna_resolution,
        "rel_eps": rel_eps,
        "rel_cutoff": rel_cutoff,
        "act_cutoff": act_cutoff,
        "allowed_gene_types": list(scoring_config.resolved_allowed_gene_types()),
        "allow_unstranded_tracks": allow_unstranded_tracks,
        "device": device,
        "full_float32": full_float32,
        "write_all_events": write_all_events,
        "memory_rule": "REF forward -> CPU RAM -> clear GPU; ALT forward -> CPU RAM -> clear GPU; extract RNA_SEQ; score in scorer",
        "scorer_rule": "mode-specific gene filter + track policy + strand filter happen before metrics",
    }
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    all_df = load_table(input_path)
    start_idx = int(batch_id) * int(batch_size)
    end_idx = start_idx + int(batch_size)
    batch_df = all_df.iloc[start_idx:end_idx].copy()
    if max_snvs is not None:
        batch_df = batch_df.head(int(max_snvs)).copy()
    batch_df = batch_df.reset_index(drop=False).rename(columns={"index": "input_row_index"})

    print(f"Total input SNVs: {len(all_df)}", flush=True)
    print(f"Selected batch rows: start={start_idx} end={end_idx} n={len(batch_df)}", flush=True)
    if batch_df.empty:
        raise ValueError("Selected batch is empty")

    print("Loading FASTA...", flush=True)
    genome = Fasta(str(fasta_path), sequence_always_upper=True)

    print("Loading GENCODE gene annotation...", flush=True)
    gene_annotation = GeneAnnotation(str(gtf_path))

    print("Loading track metadata...", flush=True)
    track_meta = load_track_metadata(track_meta_path)

    model = load_model(model_path, device=device, full_float32=full_float32)
    summary_rows: list[dict[str, Any]] = []

    for local_i, row in batch_df.iterrows():
        snv_t0 = time.perf_counter()
        ref_cache = alt_cache = None

        try:
            chrom, pos, ref, alt, variant_id = parse_variant_row(row)
            print("\n" + "=" * 90, flush=True)
            print(f"SNV {local_i + 1}/{len(batch_df)} | input_row={row['input_row_index']} | {variant_id}", flush=True)

            if len(ref) != 1 or len(alt) != 1:
                raise ValueError(f"SNV-only run expects single-base alleles, got ref={ref}, alt={alt}")

            fasta_chrom = choose_fasta_chrom(genome, chrom)
            fasta_ref = genome[fasta_chrom][pos - 1:pos].seq.upper()
            print(f"FASTA ref={fasta_ref}; input ref={ref}", flush=True)
            if fasta_ref != ref:
                raise ValueError(f"Reference allele mismatch at {variant_id}: input ref={ref}, FASTA ref={fasta_ref}")

            pos0 = int(pos) - 1
            start = pos0 - int(window_size) // 2
            end = start + int(window_size)
            if start < 0:
                raise ValueError(f"Window starts before chromosome start: {variant_id}")

            ref_seq = genome[fasta_chrom][start:end].seq.upper()
            if len(ref_seq) != int(window_size):
                raise ValueError(f"Extracted sequence length {len(ref_seq)} != window_size {window_size}")

            alt_seq_list = list(ref_seq)
            alt_seq_list[pos0 - start] = alt
            alt_seq = "".join(alt_seq_list)

            variant = Variant(chrom, int(pos), ref, alt)
            interval = Interval(chrom, int(start), int(end))

            ref_cache, ref_meta = run_forward_to_ram(model, ref_seq, device, "REF")
            alt_cache, alt_meta = run_forward_to_ram(model, alt_seq, device, "ALT")

            # screen.py extracts only RNA_SEQ tensors; scoring.py interprets them.
            ref_rna = extract_rna_seq_tensor(ref_cache, resolution=int(rna_resolution))
            alt_rna = extract_rna_seq_tensor(alt_cache, resolution=int(rna_resolution))
            alt_rna = maybe_align_alt_rna_for_variant(alt_rna, variant, interval)

            variant_context = {
                "input_row_index": int(row["input_row_index"]),
                "batch_label": batch_label,
                "branch": branch,
                "variant_id": variant_id,
                "chrom": chrom,
                "pos": int(pos),
                "ref": ref,
                "alt": alt,
                "interval_start_0based": int(start),
                "interval_end_0based": int(end),
                "window_bp": int(window_size),
            }

            selected_df, all_valid_df = score_snv_rna_seq_tensors(
                variant_context=variant_context,
                ref_rna=ref_rna,
                alt_rna=alt_rna,
                interval=interval,
                gene_annotation=gene_annotation,
                track_meta=track_meta,
                resolution=int(rna_resolution),
                config=scoring_config,
            )

            append_tsv(selected_df, screening_path)
            if write_all_events:
                append_tsv(all_valid_df, raw_path)

            elapsed = time.perf_counter() - snv_t0
            summary_rows.append({
                **variant_context,
                "mode": mode,
                "selection_mode": selection_mode,
                "track_policy": scoring_config.resolved_track_policy(),
                "status": "ok",
                "n_valid_gene_track_events": int(0 if all_valid_df is None else len(all_valid_df)),
                "n_screening_rows_written": int(0 if selected_df is None else len(selected_df)),
                "elapsed_sec": elapsed,
                **ref_meta,
                **alt_meta,
            })

        except Exception as e:
            err = {
                "input_row_index": int(row.get("input_row_index", -1)),
                "variant_id": str(row.get("variant_id", row.get("variant", "NA"))),
                "batch_label": batch_label,
                "branch": branch,
                "mode": mode,
                "selection_mode": selection_mode,
                "track_policy": track_policy,
                "status": "failed",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }
            append_tsv(pd.DataFrame([err]), failed_path)
            print("ERROR:", err["variant_id"], flush=True)
            print(err["traceback"], flush=True)
            summary_rows.append(err)

        finally:
            del ref_cache, alt_cache
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    append_tsv(pd.DataFrame(summary_rows), summary_path)

    print("\nDone.", flush=True)
    print(f"screening: {screening_path}", flush=True)
    print(f"summary:   {summary_path}", flush=True)
    print(f"failed:    {failed_path}", flush=True)
    if write_all_events:
        print(f"raw:       {raw_path}", flush=True)

    return {
        "screening_path": screening_path,
        "summary_path": summary_path,
        "failed_path": failed_path,
        "raw_path": raw_path,
        "config_path": config_path,
    }
