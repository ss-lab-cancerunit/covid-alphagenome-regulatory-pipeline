from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D

script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
data_dir = project_dir / "data"
outdir = project_dir / "outputs"
outdir.mkdir(parents=True, exist_ok=True)

cd_eqtl = pd.read_csv(data_dir / "input_coding_eqtl_211_best_of_snv.tsv.gz", sep="\t")
cd_gwas = pd.read_csv(data_dir / "input_coding_gwas_6268_best_of_snv.tsv.gz", sep="\t")
lnc_eqtl = pd.read_csv(data_dir / "input_lnc_eqtl_211_best_of_snv.tsv.gz", sep="\t")
lnc_gwas = pd.read_csv(data_dir / "input_lnc_gwas_6268_best_of_snv.tsv.gz", sep="\t")

cmap = LinearSegmentedColormap.from_list(
    "signed_change",
    ["#2166AC", "#D9D9D9", "#D73027"],
    N=256,
)

# Unified axis limits across all four panels for direct visual comparability.
# X uses the common ACT domain on log scale; Y uses the common REL range.
X_LIM = (1e-4, 3e3)
Y_LIM = (0, 105)

fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.2))

plot_order = [
    (axes[0, 0], "A", "Protein-coding eQTL-supported SNVs", cd_eqtl, "rna_signed_mean_rel_change"),
    (axes[0, 1], "B", "lncRNA eQTL-supported SNVs", lnc_eqtl, "lnc_signed_mean_rel_change"),
    (axes[1, 0], "C", "Protein-coding GWAS SNVs", cd_gwas, "rna_signed_mean_rel_change"),
    (axes[1, 1], "D", "lncRNA GWAS SNVs", lnc_gwas, "lnc_signed_mean_rel_change"),
]

summary_rows = []

for ax, panel, title, df, signed_col in plot_order:
    x = df["act"].to_numpy(dtype=float)
    y = df["rel"].to_numpy(dtype=float) * 100
    signed = df[signed_col].to_numpy(dtype=float) * 100

    p99 = float(np.nanpercentile(np.abs(signed), 99))
    norm = TwoSlopeNorm(vmin=-p99, vcenter=0, vmax=p99)

    order = np.argsort(np.abs(signed))
    point_size = 15 if len(df) < 1000 else 9
    alpha = 0.90 if len(df) < 1000 else 0.82

    ax.scatter(
        x[order], y[order],
        c=signed[order],
        cmap=cmap,
        norm=norm,
        s=point_size,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )

    ax.set_xscale("log")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=12, pad=8)

    ax.text(
        -0.12, 1.04, panel,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    n_up = int(np.sum(signed > 0))
    n_down = int(np.sum(signed < 0))
    summary_rows.append(
        {"panel": panel, "dataset": title, "n_total": len(df), "n_alt_up": n_up, "n_alt_down": n_down}
    )

    handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=6,
               markerfacecolor="#D73027", markeredgecolor='none',
               label=f"ALT up-regulated: n={n_up}"),
        Line2D([0], [0], marker='o', linestyle='None', markersize=6,
               markerfacecolor="#2166AC", markeredgecolor='none',
               label=f"ALT down-regulated: n={n_down}"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=10,
        frameon=True,
        borderpad=0.4,
        handletextpad=0.6,
        labelspacing=0.45,
    )
    leg.get_frame().set_alpha(0.92)

    cax = inset_axes(
        ax, width="3.0%", height="43%", loc="upper right", borderpad=1.65
    )
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax
    )
    cb.set_ticks([-p99, 0, p99])
    cb.set_ticklabels([f"−{p99:.1f}", "0", f"+{p99:.1f}"])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.outline.set_linewidth(0.5)

for ax in axes.ravel():
    ax.set_xlabel("RNA expression activity (ACT)", fontsize=11)
    ax.set_ylabel("Relative RNA perturbation (%)", fontsize=11)

fig.subplots_adjust(left=0.09, right=0.985, bottom=0.08, top=0.96, wspace=0.24, hspace=0.28)

png = outdir / "Figure_Results_best_of_SNV_ACT_REL_2x2_v5_all_axes.png"
svg = outdir / "Figure_Results_best_of_SNV_ACT_REL_2x2_v5_all_axes.svg"
tsv = outdir / "Figure_Results_best_of_SNV_ACT_REL_2x2_v5_all_axes_ALT_up_down_counts.tsv"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
plt.close(fig)

pd.DataFrame(summary_rows).to_csv(tsv, sep="\t", index=False)
print("Saved:", png)
print("Saved:", svg)
print("Saved:", tsv)
