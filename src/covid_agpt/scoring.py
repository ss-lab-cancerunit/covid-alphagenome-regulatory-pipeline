"""
COVID AGPT RNA event scorer.

This module is intentionally responsible for the biological interpretation of
RNA_SEQ tensors.  screen.py should only run AGPT REF/ALT inference and pass
RNA_SEQ tensors here.

Core design
-----------
1. mode='cd'
   - genes: protein_coding
   - mask: exons
   - tracks: PolyA RNA-seq when track_policy='auto'
   - REL: sum(abs(ALT - REF)) / (sum(abs(REF)) + eps)

2. mode='lnc'
   - genes: lncRNA by default
   - mask: gene body
   - tracks: total RNA-seq when track_policy='auto'
   - REL: mean(abs(ALT - REF)) / (mean(abs(REF)) + eps)

3. Strand validity is applied before metric calculation.
   Opposite-strand or unknown-strand gene x track events do not enter scoring.

4. selection_mode
   - best_per_snv: retain the single highest-REL valid event per SNV
   - threshold: retain all valid events passing rel_cutoff and/or act_cutoff
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import torch

try:
    from alphagenome_pytorch.variant_scoring import Interval, OutputType
    from alphagenome_pytorch.variant_scoring.scorers.gene_mask import GeneMaskMode
except Exception:  # pragma: no cover - allows py_compile outside AGPT env
    Interval = Any  # type: ignore
    OutputType = Any  # type: ignore
    GeneMaskMode = Any  # type: ignore


Mode = Literal["cd", "lnc"]
SelectionMode = Literal["best_per_snv", "threshold"]
TrackPolicy = Literal["auto", "legacy_all_rna", "cd_polya", "lnc_total"]

DEFAULT_ALLOWED_GENE_TYPES = {
    "cd": ("protein_coding",),
    "lnc": ("lncRNA",),
}

# Optional broader lnc set.  Keep default strict lncRNA for reproduction, but
# allow PBS/CLI to pass a broader comma-separated list later if needed.
BROAD_LNC_GENE_TYPES = (
    "lncRNA",
    "lincRNA",
    "antisense",
    "sense_intronic",
    "sense_overlapping",
    "processed_transcript",
    "3prime_overlapping_ncRNA",
    "non_coding",
)


@dataclass(frozen=True)
class ScoringConfig:
    mode: Mode
    selection_mode: SelectionMode = "best_per_snv"
    track_policy: TrackPolicy = "auto"
    rel_eps: float = 1e-3
    rel_cutoff: float | None = None
    act_cutoff: float | None = None
    allowed_gene_types: tuple[str, ...] | None = None
    allow_unstranded_tracks: bool = True
    require_strand_compatible: bool = True

    def resolved_allowed_gene_types(self) -> tuple[str, ...]:
        if self.allowed_gene_types is not None:
            return tuple(str(x).strip() for x in self.allowed_gene_types if str(x).strip())
        return DEFAULT_ALLOWED_GENE_TYPES[self.mode]

    def resolved_track_policy(self) -> TrackPolicy:
        if self.track_policy != "auto":
            return self.track_policy
        return "cd_polya" if self.mode == "cd" else "lnc_total"

    def gene_mask_label(self) -> str:
        return "exons" if self.mode == "cd" else "gene_body"


# ---------------------------------------------------------------------
# Metadata loading and normalization
# ---------------------------------------------------------------------


def load_track_metadata(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Track metadata not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        meta = pd.read_parquet(path)
    elif suffix in {".tsv", ".txt"}:
        meta = pd.read_csv(path, sep="\t", low_memory=False)
    elif suffix == ".csv":
        meta = pd.read_csv(path, low_memory=False)
    elif suffix in {".xlsx", ".xls"}:
        meta = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported track metadata format: {path}")

    meta = normalize_track_metadata(meta)
    print(f"Loaded track metadata: {path} rows={len(meta)} cols={len(meta.columns)}", flush=True)
    return meta


def normalize_track_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()

    if "track_index" not in meta.columns:
        meta = meta.reset_index()
        for c in ["track_index", "index", "track_idx", "idx"]:
            if c in meta.columns:
                meta = meta.rename(columns={c: "track_index"})
                break

    rename_map = {}
    if "name" in meta.columns and "track_name" not in meta.columns:
        rename_map["name"] = "track_name"
    if "Assay title" in meta.columns and "assay_title" not in meta.columns:
        rename_map["Assay title"] = "assay_title"
    if "assay" in meta.columns and "assay_title" not in meta.columns:
        rename_map["assay"] = "assay_title"
    if "strand" in meta.columns and "track_strand" not in meta.columns:
        rename_map["strand"] = "track_strand"
    if "output" in meta.columns and "output_type" not in meta.columns:
        rename_map["output"] = "output_type"
    if "head" in meta.columns and "output_type" not in meta.columns:
        rename_map["head"] = "output_type"
    if "head_name" in meta.columns and "output_type" not in meta.columns:
        rename_map["head_name"] = "output_type"
    if "output_name" in meta.columns and "output_type" not in meta.columns:
        rename_map["output_name"] = "output_type"
    if "modality" in meta.columns and "output_type" not in meta.columns:
        rename_map["modality"] = "output_type"

    if rename_map:
        meta = meta.rename(columns=rename_map)

    if "track_index" not in meta.columns:
        raise ValueError("track_metadata must contain or yield a track_index column")

    meta["track_index"] = pd.to_numeric(meta["track_index"], errors="coerce").astype("Int64")
    meta = meta[meta["track_index"].notna()].copy()
    meta["track_index"] = meta["track_index"].astype(int)

    if "output_type" in meta.columns:
        meta["output_type"] = meta["output_type"].astype(str)
    else:
        meta["output_type"] = "unknown"

    if "track_strand" not in meta.columns:
        meta["track_strand"] = "unknown"

    return meta


def _norm_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip().lower()


def _metadata_text(track_row: pd.Series) -> str:
    cols = [
        "track_name",
        "assay_title",
        "assay",
        "description",
        "biosample_name",
        "biosample_type",
        "data_source",
        "library_strategy",
        "rna_subtype",
        "sample_type",
        "experiment_type",
        "output_type",
    ]
    parts = []
    for c in cols:
        if c in track_row.index:
            v = _norm_text(track_row[c])
            if v:
                parts.append(v)
    return " | ".join(parts)


def _output_type_is_rna_seq(x: Any) -> bool:
    s = _norm_text(x).replace("-", "_").replace(" ", "_")
    return s in {"rna_seq", "rnaseq", "rna"} or "rna_seq" in s or "rnaseq" in s


def prepare_rna_track_metadata(
    track_meta: pd.DataFrame | None,
    *,
    n_tracks: int,
    output_type_value: str = "rna_seq",
) -> pd.DataFrame:
    """Return one metadata row per RNA_SEQ tensor track index.

    AGPT track metadata may either contain all output heads with output_type +
    per-head track_index, or may already be specific to one head.  This helper
    first tries output_type-aware filtering and then falls back to track_index.
    """
    if track_meta is None or track_meta.empty:
        raise ValueError("track_metadata is required because strand and RNA subtype filters are mandatory")

    meta = normalize_track_metadata(track_meta)
    output_type_value = _norm_text(output_type_value)

    if "output_type" in meta.columns:
        rna_meta = meta[meta["output_type"].map(_output_type_is_rna_seq)].copy()
    else:
        rna_meta = meta.copy()

    # Fallback: if output_type naming failed, use track_index range.
    if rna_meta.empty:
        rna_meta = meta.copy()

    rna_meta = rna_meta[rna_meta["track_index"].between(0, int(n_tracks) - 1)].copy()
    rna_meta = rna_meta.drop_duplicates(["track_index"], keep="first").copy()

    if rna_meta.empty:
        raise ValueError(f"No RNA_SEQ track metadata rows found for n_tracks={n_tracks}")

    return rna_meta


# ---------------------------------------------------------------------
# Track policy and strand logic
# ---------------------------------------------------------------------


def track_is_rna_seq(track_row: pd.Series) -> bool:
    if "output_type" in track_row.index and _output_type_is_rna_seq(track_row["output_type"]):
        return True
    text = _metadata_text(track_row)
    return "rna-seq" in text or "rna_seq" in text or "rnaseq" in text or " rna " in f" {text} "


def track_is_polya(track_row: pd.Series) -> bool:
    text = _metadata_text(track_row)
    keys = [
        "polya",
        "poly-a",
        "poly a",
        "poly(a)",
        "polyadenylated",
        "polyadenylation",
        "poly a selected",
        "poly-a selected",
    ]
    return any(k in text for k in keys)


def track_is_total_rna(track_row: pd.Series) -> bool:
    text = _metadata_text(track_row)
    keys = [
        "total rna",
        "total-rna",
        "total_rna",
        "total-rna-seq",
        "total rna-seq",
        "whole transcriptome",
        "ribodepleted",
        "ribo-depleted",
        "ribo depleted",
        "ribo zero",
        "ribozero",
    ]
    return any(k in text for k in keys)


def is_track_allowed_by_policy(track_row: pd.Series, config: ScoringConfig) -> bool:
    policy = config.resolved_track_policy()

    # We are already inside RNA_SEQ tensor metadata, but keep this defensive.
    if not track_is_rna_seq(track_row):
        return False

    if policy == "legacy_all_rna":
        return True
    if policy == "cd_polya":
        return track_is_polya(track_row)
    if policy == "lnc_total":
        return track_is_total_rna(track_row)
    raise ValueError(f"Unsupported track_policy: {policy}")


def normalize_strand_value(x: Any) -> str:
    if x is None:
        return "unknown"
    try:
        if pd.isna(x):
            return "unknown"
    except Exception:
        pass

    s = str(x).strip().lower()
    if s in {"+", "plus", "forward", "sense", "1", "+1"}:
        return "+"
    if s in {"-", "minus", "reverse", "antisense", "-1"}:
        return "-"
    if s in {".", "*", "none", "na", "nan", "unstranded", "not stranded", "not_stranded"}:
        return "unstranded"
    return "unknown"


def classify_strand_match(
    gene_strand: Any,
    track_strand: Any,
    *,
    allow_unstranded_tracks: bool = True,
) -> str:
    gs = normalize_strand_value(gene_strand)
    ts = normalize_strand_value(track_strand)

    if gs in {"+", "-"} and ts in {"+", "-"}:
        return "compatible_same_strand" if gs == ts else "incompatible_opposite_strand"
    if ts == "unstranded":
        return "compatible_unstranded" if allow_unstranded_tracks else "excluded_unstranded"
    if gs == "unknown":
        return "unknown_gene_strand"
    if ts == "unknown":
        return "unknown_track_strand"
    return "unknown_or_invalid"


def is_valid_strand_status(status: str) -> bool:
    return status in {"compatible_same_strand", "compatible_unstranded"}


def allowed_tracks_for_gene(
    rna_track_meta: pd.DataFrame,
    *,
    gene_strand: Any,
    config: ScoringConfig,
) -> pd.DataFrame:
    """Apply RNA subtype and strand filters before metric calculation."""
    rows = []
    for _, track in rna_track_meta.iterrows():
        if not is_track_allowed_by_policy(track, config):
            continue
        strand_status = classify_strand_match(
            gene_strand=gene_strand,
            track_strand=track.get("track_strand", "unknown"),
            allow_unstranded_tracks=config.allow_unstranded_tracks,
        )
        if config.require_strand_compatible and not is_valid_strand_status(strand_status):
            continue
        out = track.copy()
        out["track_strand_norm"] = normalize_strand_value(track.get("track_strand", "unknown"))
        out["strand_match_status"] = strand_status
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=list(rna_track_meta.columns) + ["track_strand_norm", "strand_match_status"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Gene/mask and metric logic
# ---------------------------------------------------------------------


def gene_is_allowed(gene_info: dict[str, Any] | None, config: ScoringConfig) -> bool:
    if not gene_info:
        return False
    gene_type = str(gene_info.get("gene_type", ""))
    return gene_type in set(config.resolved_allowed_gene_types())


def build_gene_mask(
    *,
    gene_annotation: Any,
    gene_id: str,
    interval: Any,
    resolution: int,
    seq_length: int,
    device: torch.device | str,
    config: ScoringConfig,
) -> tuple[torch.Tensor, str]:
    if config.mode == "cd":
        mask = gene_annotation.get_exon_mask(
            gene_id=gene_id,
            interval=interval,
            resolution=resolution,
            seq_length=seq_length,
            device=device,
        )
        return mask.bool(), "exons"

    if config.mode == "lnc":
        mask = gene_annotation.get_gene_mask(
            gene_id=gene_id,
            interval=interval,
            resolution=resolution,
            seq_length=seq_length,
            device=device,
        )
        return mask.bool(), "gene_body"

    raise ValueError(f"Unsupported mode: {config.mode}")


def compute_masked_metrics(
    ref_values: torch.Tensor,
    alt_values: torch.Tensor,
    *,
    config: ScoringConfig,
) -> dict[str, float]:
    """Calculate REL/ACT/signed change for one valid gene x track event."""
    if ref_values.numel() == 0 or alt_values.numel() == 0:
        raise ValueError("Empty masked values")

    ref_values = ref_values.detach().float()
    alt_values = alt_values.detach().float()
    signed_change = alt_values - ref_values
    abs_change = torch.abs(signed_change)

    ref_mean = ref_values.mean()
    alt_mean = alt_values.mean()

    if config.mode == "cd":
        # Validated CD211 whole-exon sum-ratio axis.
        denom = torch.abs(ref_values).sum() + float(config.rel_eps)
        rel = abs_change.sum() / denom
    elif config.mode == "lnc":
        # Validated lnc gene-body mean-ratio axis.
        denom = torch.abs(ref_values).mean() + float(config.rel_eps)
        rel = abs_change.mean() / denom
    else:
        raise ValueError(f"Unsupported mode: {config.mode}")

    signed_mean_change = (alt_mean - ref_mean) / (ref_mean + float(config.rel_eps))
    act = torch.maximum(ref_mean, alt_mean)

    return {
        "rel": float(rel.detach().cpu()),
        "ref_mean": float(ref_mean.detach().cpu()),
        "alt_mean": float(alt_mean.detach().cpu()),
        "act": float(act.detach().cpu()),
        "signed_mean_change": float(signed_mean_change.detach().cpu()),
        "abs_mean_change": float(abs_change.mean().detach().cpu()),
    }


def _safe_get(row: pd.Series, col: str, default: Any = None) -> Any:
    return row[col] if col in row.index else default


def _track_output_columns(track: pd.Series) -> dict[str, Any]:
    return {
        "track_index": int(track.get("track_index")),
        "track_name": _safe_get(track, "track_name"),
        "track_strand": _safe_get(track, "track_strand"),
        "track_strand_norm": _safe_get(track, "track_strand_norm"),
        "strand_match_status": _safe_get(track, "strand_match_status"),
        "biosample_name": _safe_get(track, "biosample_name"),
        "biosample_type": _safe_get(track, "biosample_type"),
        "gtex_tissue": _safe_get(track, "gtex_tissue"),
        "assay_title": _safe_get(track, "assay_title"),
        "data_source": _safe_get(track, "data_source"),
    }


def _passes_threshold(event: dict[str, Any], config: ScoringConfig) -> bool:
    if config.rel_cutoff is not None and float(event["rel"]) < float(config.rel_cutoff):
        return False
    if config.act_cutoff is not None and float(event["act"]) < float(config.act_cutoff):
        return False
    return True


def _no_event_row(
    *,
    variant_context: dict[str, Any],
    config: ScoringConfig,
    reason: str,
) -> pd.DataFrame:
    row = {
        **variant_context,
        "mode": config.mode,
        "selection_mode": config.selection_mode,
        "track_policy": config.resolved_track_policy(),
        "metric": "rel",
        "primary_axis": "REL",
        "best_score": np.nan,
        "rel": np.nan,
        "ref_mean": np.nan,
        "alt_mean": np.nan,
        "act": np.nan,
        "signed_mean_change": np.nan,
        "status": reason,
    }
    return pd.DataFrame([row])


def select_events(
    valid_events: list[dict[str, Any]],
    *,
    variant_context: dict[str, Any],
    config: ScoringConfig,
) -> pd.DataFrame:
    if not valid_events:
        return _no_event_row(variant_context=variant_context, config=config, reason="no_valid_event")

    if config.selection_mode == "best_per_snv":
        best = max(valid_events, key=lambda e: float(e["rel"]))
        row = {
            **variant_context,
            **best,
            "selection_mode": "best_per_snv",
            "metric": "rel",
            "primary_axis": "REL",
            "best_score": best["rel"],
            "best_gene_id": best.get("gene_id"),
            "best_gene_name": best.get("gene_name"),
            "best_gene_type": best.get("gene_type"),
            "best_gene_strand": best.get("gene_strand"),
            "status": "ok",
        }
        return pd.DataFrame([row])

    if config.selection_mode == "threshold":
        selected = [e for e in valid_events if _passes_threshold(e, config)]
        if not selected:
            return _no_event_row(variant_context=variant_context, config=config, reason="no_threshold_event")
        rows = []
        for e in selected:
            rows.append({
                **variant_context,
                **e,
                "selection_mode": "threshold",
                "metric": "rel",
                "primary_axis": "REL",
                "best_score": e["rel"],
                "status": "ok",
            })
        return pd.DataFrame(rows)

    raise ValueError(f"Unsupported selection_mode: {config.selection_mode}")


def score_snv_rna_seq_tensors(
    *,
    variant_context: dict[str, Any],
    ref_rna: torch.Tensor,
    alt_rna: torch.Tensor,
    interval: Any,
    gene_annotation: Any,
    track_meta: pd.DataFrame | None,
    resolution: int,
    config: ScoringConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one SNV from RNA_SEQ tensors.

    screen.py must pass CPU/RAM tensors shaped B x S x T or S x T.
    This scorer filters gene type, RNA track subtype and strand before metrics.
    """
    if ref_rna.dim() == 2:
        ref_rna = ref_rna.unsqueeze(0)
    if alt_rna.dim() == 2:
        alt_rna = alt_rna.unsqueeze(0)
    if ref_rna.dim() != 3 or alt_rna.dim() != 3:
        raise ValueError(f"Expected RNA tensors BxSxT or SxT, got {tuple(ref_rna.shape)} and {tuple(alt_rna.shape)}")
    if ref_rna.shape != alt_rna.shape:
        raise ValueError(f"RNA ref/alt shape mismatch: {tuple(ref_rna.shape)} vs {tuple(alt_rna.shape)}")
    if ref_rna.shape[0] != 1:
        raise ValueError(f"This scorer expects one SNV at a time, got B={ref_rna.shape[0]}")

    _, seq_len, n_tracks = ref_rna.shape
    ref_st = ref_rna[0]
    alt_st = alt_rna[0]
    device = ref_st.device

    rna_track_meta = prepare_rna_track_metadata(track_meta, n_tracks=int(n_tracks), output_type_value="rna_seq")
    gene_ids = gene_annotation.get_genes_in_interval(interval)

    valid_events: list[dict[str, Any]] = []
    n_genes_considered = 0
    n_gene_track_pairs_prefilter = 0
    n_gene_track_pairs_after_track_policy = 0
    n_gene_track_pairs_after_strand = 0

    for gene_id in gene_ids:
        gene_info = gene_annotation.get_gene_info(gene_id)
        if not gene_is_allowed(gene_info, config):
            continue

        gene_strand = gene_info.get("strand") if gene_info else None
        gene_strand_norm = normalize_strand_value(gene_strand)

        gene_mask, mask_label = build_gene_mask(
            gene_annotation=gene_annotation,
            gene_id=gene_id,
            interval=interval,
            resolution=resolution,
            seq_length=int(seq_len),
            device=device,
            config=config,
        )
        n_gene_bins = int(gene_mask.sum().item())
        if n_gene_bins == 0:
            continue

        n_genes_considered += 1
        n_gene_track_pairs_prefilter += int(n_tracks)

        # Track subtype + strand compatibility are resolved before metrics.
        allowed_tracks = allowed_tracks_for_gene(rna_track_meta, gene_strand=gene_strand, config=config)
        n_gene_track_pairs_after_track_policy += int(len(allowed_tracks))
        if allowed_tracks.empty:
            continue

        ref_selected = ref_st[gene_mask, :]
        alt_selected = alt_st[gene_mask, :]

        for _, track in allowed_tracks.iterrows():
            track_index = int(track["track_index"])
            if track_index < 0 or track_index >= int(n_tracks):
                continue

            # At this point strand is valid; only now calculate metrics.
            metrics = compute_masked_metrics(
                ref_selected[:, track_index],
                alt_selected[:, track_index],
                config=config,
            )
            n_gene_track_pairs_after_strand += 1

            event = {
                "mode": config.mode,
                "selection_mode": config.selection_mode,
                "track_policy": config.resolved_track_policy(),
                "output_type": "rna_seq",
                "resolution": int(resolution),
                "gene_mask_mode": mask_label,
                "allowed_gene_types": ";".join(config.resolved_allowed_gene_types()),
                "gene_id": gene_id,
                "gene_name": gene_info.get("gene_name") if gene_info else None,
                "gene_type": gene_info.get("gene_type") if gene_info else None,
                "gene_strand": gene_strand,
                "gene_strand_norm": gene_strand_norm,
                "n_gene_bins": n_gene_bins,
                **_track_output_columns(track),
                **metrics,
                "n_genes_considered": n_genes_considered,
                "n_gene_track_pairs_considered_prefilter": n_gene_track_pairs_prefilter,
                "n_gene_track_pairs_after_track_policy": n_gene_track_pairs_after_track_policy,
                "n_gene_track_pairs_considered": n_gene_track_pairs_after_strand,
                "status": "ok",
            }

            # Compatibility aliases for old/new downstream scripts.
            if config.mode == "cd":
                event["rna_rel_whole"] = event["rel"]
                event["ref_exon_mean"] = event["ref_mean"]
                event["alt_exon_mean"] = event["alt_mean"]
                event["rna_signed_mean_rel_change"] = event["signed_mean_change"]
                event["agpt_act_like"] = event["act"]
            else:
                event["lnc_rel_gene_body"] = event["rel"]
                event["ref_gene_body_mean"] = event["ref_mean"]
                event["alt_gene_body_mean"] = event["alt_mean"]
                event["lnc_signed_mean_rel_change"] = event["signed_mean_change"]
                event["lnc_act_like"] = event["act"]

            valid_events.append(event)

    all_valid_df = pd.DataFrame(valid_events)
    selected_df = select_events(valid_events, variant_context=variant_context, config=config)
    return selected_df, all_valid_df
