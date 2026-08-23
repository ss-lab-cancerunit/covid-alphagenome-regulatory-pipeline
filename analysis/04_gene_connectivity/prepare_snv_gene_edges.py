#!/usr/bin/env python3
"""Derive the unique coding SNV-gene pairs used by the connectivity analysis."""

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE.parent
    / "01_threshold_and_event_expansion"
    / "data"
    / "input_coding_gwas_expanded_events.tsv.gz"
)
OUTPUT = HERE / "data" / "edges_snv_gene.csv"


def main() -> None:
    events = pd.read_csv(
        SOURCE,
        sep="\t",
        usecols=["variant_id", "gene_id"],
    )
    pairs = (
        events.rename(columns={"variant_id": "snv_id"})
        .drop_duplicates()
        .sort_values(["snv_id", "gene_id"])
        .reset_index(drop=True)
    )

    observed = {
        "pairs": len(pairs),
        "snvs": pairs["snv_id"].nunique(),
        "genes": pairs["gene_id"].nunique(),
    }
    expected = {"pairs": 248, "snvs": 187, "genes": 79}
    if observed != expected:
        raise RuntimeError(f"Unexpected SNV-gene counts: {observed}; expected {expected}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(pairs)} unique pairs to {OUTPUT}")


if __name__ == "__main__":
    main()
