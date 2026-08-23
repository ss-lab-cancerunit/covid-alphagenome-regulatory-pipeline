
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "K": [3, 4, 5],
    "CD_eQTL": [22.553, 12.410, 9.138],
    "CD_GWAS": [8.873, 7.261, 2.962],
    "lnc_eQTL": [7.490, 6.327, 6.327],
    "lnc_GWAS": [7.385, 2.575, 2.423],
})

script_dir = Path(__file__).resolve().parent
outdir = script_dir.parent / "outputs"
outdir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)

ax = axes[0]
ax.plot(df["K"], df["CD_eQTL"], marker="o", linewidth=2, label="eQTL-supported SNVs")
ax.plot(df["K"], df["CD_GWAS"], marker="o", linewidth=2, label="GWAS SNVs")
for x, y in zip(df["K"], df["CD_eQTL"]):
    ax.text(x, y + 0.5, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
for x, y in zip(df["K"], df["CD_GWAS"]):
    ax.text(x, y + 0.5, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
ax.set_title("Protein-coding mode", fontsize=13)
ax.set_xlabel("Number of clusters (K)", fontsize=11)
ax.set_ylabel("First REL cutoff (%)", fontsize=11)
ax.set_xticks([3, 4, 5])
ax.grid(True, alpha=0.3)
ax.legend(frameon=True, fontsize=9)

ax = axes[1]
ax.plot(df["K"], df["lnc_eQTL"], marker="o", linewidth=2, label="eQTL-supported SNVs")
ax.plot(df["K"], df["lnc_GWAS"], marker="o", linewidth=2, label="GWAS SNVs")
for x, y in zip(df["K"], df["lnc_eQTL"]):
    ax.text(x, y + 0.5, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
for x, y in zip(df["K"], df["lnc_GWAS"]):
    if x == 3:
        ax.text(x, y - 0.55, f"{y:.1f}", ha="center", va="top", fontsize=8)
    else:
        ax.text(x, y + 0.5, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
ax.set_title("lncRNA mode", fontsize=13)
ax.set_xlabel("Number of clusters (K)", fontsize=11)
ax.set_xticks([3, 4, 5])
ax.grid(True, alpha=0.3)
ax.legend(frameon=True, fontsize=9)

fig.suptitle("Learned first REL cutoff across K values", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(outdir / "Results_3_2_first_REL_cutoff_across_K_two_panels_v2.png", dpi=300, bbox_inches="tight")
plt.savefig(outdir / "Results_3_2_first_REL_cutoff_across_K_two_panels_v2.svg", bbox_inches="tight")
plt.close(fig)
