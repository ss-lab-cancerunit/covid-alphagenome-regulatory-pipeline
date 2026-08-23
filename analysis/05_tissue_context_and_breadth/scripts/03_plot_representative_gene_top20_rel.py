from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent
ANALYSIS_DIR = MODULE_DIR.parent
EXPANDED_FILE = (
    ANALYSIS_DIR
    / "01_threshold_and_event_expansion"
    / "data"
    / "input_coding_gwas_expanded_events.tsv.gz"
)
METADATA_FILE = MODULE_DIR / "data" / "track_metadata_human.csv.gz"
OUTDIR = MODULE_DIR / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

cd = pd.read_csv(EXPANDED_FILE, sep="\t")
meta = pd.read_csv(METADATA_FILE)

genes = ["CCR1", "ACE2", "PSORS1C1", "MAPT"]

meta_small = meta[["track_index", "track_name", "biosample_name", "strand"]].copy()
merged = cd.merge(
    meta_small,
    on=["track_index", "track_name"],
    how="left",
    suffixes=("_event", "_meta"),
)
merged["display_label"] = merged["biosample_name_meta"].fillna(merged["biosample_name_event"])
merged["display_strand"] = (
    merged["strand"]
    .fillna(merged["track_strand"])
    .astype(str)
    .replace({"0": ".", "nan": "."})
)

all_top20 = []

for gene in genes:
    sub = merged.loc[merged["gene_name"].eq(gene)].copy()

    # One exact gene × track entry; retain max REL across SNVs.
    idx = sub.groupby(["track_index", "track_name"])["rel"].idxmax()
    track_df = sub.loc[idx].copy()
    track_df["REL_percent"] = track_df["rel"] * 100.0

    top20 = (
        track_df.sort_values(
            ["REL_percent", "display_label", "track_index"],
            ascending=[False, True, True],
        )
        .head(20)
        .reset_index(drop=True)
    )

    # Strand suffixes distinguish exact RNA tracks sharing a biosample name.
    top20["plot_label"] = (
        top20["display_label"].astype(str)
        + " ("
        + top20["display_strand"].astype(str)
        + ")"
    )
    top20["gene"] = gene

    all_top20.append(
        top20[
            [
                "gene", "plot_label", "display_label", "display_strand", "REL_percent",
                "variant_id", "track_index", "track_name", "act"
            ]
        ]
    )

top20_all = pd.concat(all_top20, ignore_index=True)

shared_ymax = np.ceil(top20_all["REL_percent"].max() / 10.0) * 10.0

fig, axes = plt.subplots(2, 2, figsize=(20, 13), sharey=True)
axes = axes.ravel()

for ax, gene in zip(axes, genes):
    d = top20_all.loc[top20_all["gene"].eq(gene)].copy()
    x = np.arange(len(d))

    ax.bar(x, d["REL_percent"].to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels(
        d["plot_label"].tolist(),
        rotation=60,
        ha="right",
        fontsize=7.5,
    )
    ax.set_ylim(0, shared_ymax)
    ax.set_title(gene, fontsize=13)
    ax.set_xlabel("RNA-track biosample", fontsize=10)
    ax.set_ylabel("Maximum REL (%)", fontsize=10)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Top 20 responsive RNA tracks for representative coding genes",
    fontsize=16,
    y=0.995,
)

fig.tight_layout(rect=[0, 0, 1, 0.97])

fig.savefig(
    OUTDIR / "CCR1_ACE2_PSORS1C1_MAPT_top20_REL_4panel.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    OUTDIR / "CCR1_ACE2_PSORS1C1_MAPT_top20_REL_4panel.pdf",
    bbox_inches="tight",
)
fig.savefig(
    OUTDIR / "CCR1_ACE2_PSORS1C1_MAPT_top20_REL_4panel.svg",
    bbox_inches="tight",
)
plt.close(fig)

top20_all.to_csv(
    OUTDIR / "CCR1_ACE2_PSORS1C1_MAPT_top20_REL.tsv",
    sep="\t",
    index=False,
)
