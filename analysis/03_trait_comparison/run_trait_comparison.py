
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import binomtest, chi2

HERE = Path(__file__).resolve().parent
outdir = HERE / "outputs"
outdir.mkdir(parents=True, exist_ok=True)

exact_groups = {
    "A": ["FBRSL1","IFNA10","S1PR5","MUC1","SFTPD","CCR9","SLC6A20"],
    "B": ["CIB4","CSF2","ELF5","HLA-B","ICAM3","PIR","VEGFD","CCHCR1","CRHR1",
          "IFNAR2-IL10RB","MAPT","OAS2","PSORS1C1","PSORS1C2","SPPL2C","STH",
          "TCF19","CCR1","IFNAR2","POU5F1","XCR1"],
    "C": ["ACE2","ADAMTS13","BMX","CLTRN","ENSG00000285602","FGF21","FUT1","MAMSTR",
          "PLEKHA4","PPP1R15A","SLC2A6","THBS3","TULP2","MUC1","OBP2B","ABO",
          "IZUMO1","CCR9","SLC6A20"],
    "AB": ["ARHGAP27","C6orf15","CAT","CCRL2","CDSN","CRIPTO","ENSG00000283877","ICAM4",
           "ICAM5","IFNA14","KCNC3","LINC02210-CRHR1","LRRC2","LTF","MICB","MUC5B",
           "NAPSA","PDE4A","RTP3","SLC35B1","SLC5A3","TNFAIP8L1","ZGLP1","CCHCR1",
           "CCR3","CCR5","CRHR1","CXCR6","MAPT","PSORS1C1","PSORS1C2","SPPL2C","STH",
           "TCF19","CCR1","IFNAR2","POU5F1","XCR1","CCR9"],
    "AC": ["RASIP1","IZUMO1"],
    "BC": ["CEP97","ENSG00000293268","OAS2","OAS3","ABO","SFTPD","SLC6A20"],
    "ABC": ["CCR2","FUT2","IL10RB","LZTFL1","OAS1","CCR3","CCR5","CXCR6","IFNAR2-IL10RB",
            "OAS3","OBP2B","ABO","CCR1","IFNAR2","IZUMO1","POU5F1","SFTPD","XCR1",
            "CCR9","SLC6A20"]
}
trait_order = ["A", "B", "C", "AB", "AC", "BC", "ABC"]

all_genes = sorted(set().union(*[set(v) for v in exact_groups.values()]))
membership = pd.DataFrame(
    {g: [int(gene in set(exact_groups[g])) for gene in all_genes] for g in trait_order},
    index=all_genes
)
membership["n_groups"] = membership.sum(axis=1)
membership_sorted = (
    membership.assign(gene=membership.index)
    .sort_values(["n_groups"] + trait_order + ["gene"], ascending=[False] * (1 + len(trait_order)) + [True])
)
ordered_genes = membership_sorted["gene"].tolist()
gene_counts_exact = {k: len(v) for k, v in exact_groups.items()}
snv_counts_exact = {
    "A": 7,
    "B": 20,
    "C": 40,
    "AB": 68,
    "AC": 2,
    "BC": 9,
    "ABC": 41,
}

A = set(exact_groups["A"]) | set(exact_groups["AB"]) | set(exact_groups["AC"]) | set(exact_groups["ABC"])
B = set(exact_groups["B"]) | set(exact_groups["AB"]) | set(exact_groups["BC"]) | set(exact_groups["ABC"])
C = set(exact_groups["C"]) | set(exact_groups["AC"]) | set(exact_groups["BC"]) | set(exact_groups["ABC"])

abc_matrix = pd.DataFrame({
    "A": [int(g in A) for g in all_genes],
    "B": [int(g in B) for g in all_genes],
    "C": [int(g in C) for g in all_genes],
}, index=all_genes)

def cochrans_q(binary_matrix):
    """Cochran Q test for matched binary observations."""
    matrix = np.asarray(binary_matrix, dtype=float)
    n_groups = matrix.shape[1]
    column_totals = matrix.sum(axis=0)
    row_totals = matrix.sum(axis=1)
    grand_total = column_totals.sum()
    numerator = (n_groups - 1) * (
        n_groups * np.square(column_totals).sum() - grand_total**2
    )
    denominator = n_groups * grand_total - np.square(row_totals).sum()
    statistic = float(numerator / denominator)
    pvalue = float(chi2.sf(statistic, n_groups - 1))
    return SimpleNamespace(
        statistic=statistic,
        df=n_groups - 1,
        pvalue=pvalue,
    )

cq = cochrans_q(abc_matrix.values)

def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adj_ranked = ranked * len(p) / np.arange(1, len(p) + 1)
    adj_ranked = np.minimum.accumulate(adj_ranked[::-1])[::-1]
    out = np.empty_like(adj_ranked)
    out[order] = np.minimum(adj_ranked, 1.0)
    return out

rows = []
for xlab, ylab, X, Y in [("A","B",A,B), ("A","C",A,C), ("B","C",B,C)]:
    x_only = len(X - Y)
    y_only = len(Y - X)
    discordant = x_only + y_only
    p = binomtest(min(x_only, y_only), n=discordant, p=0.5, alternative="two-sided").pvalue if discordant else 1.0
    rows.append({"comparison": f"{xlab} vs {ylab}", "mcnemar_p": p})

pair_df = pd.DataFrame(rows)
pair_df["mcnemar_bh_q"] = bh_adjust(pair_df["mcnemar_p"].values)

def q_to_label(q):
    if q < 0.001:
        return "***"
    elif q < 0.01:
        return "**"
    elif q < 0.05:
        return "*"
    return "ns"

meaning_text = (
    "A = very severe respiratory confirmed COVID\n"
    "B = hospitalised COVID\n"
    "C = reported SARS-CoV-2 infection"
)

# Figure A: exact trait-group SNV and target-gene counts
x_exact = np.arange(len(trait_order))
width = 0.34
fig, ax = plt.subplots(figsize=(10.8, 6.8))
snv_bars = ax.bar(
    x_exact - width / 2,
    [snv_counts_exact[group] for group in trait_order],
    width,
    label="Unique SNVs",
)
gene_bars = ax.bar(
    x_exact + width / 2,
    [gene_counts_exact[group] for group in trait_order],
    width,
    label="Protein-coding target genes",
)
ax.set_xticks(x_exact, trait_order)
ax.set_xlabel("Exact COVID-19 trait group")
ax.set_ylabel("Count")
ax.set_title("SNVs and coding target genes across exact COVID-19 trait groups")
ax.bar_label(snv_bars, padding=2, fontsize=9)
ax.bar_label(gene_bars, padding=2, fontsize=9)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
for ext in ["png", "pdf", "svg"]:
    fig.savefig(
        outdir / f"Figure_3_5_2A_exact_trait_group_SNV_gene_counts.{ext}",
        dpi=300,
        bbox_inches="tight",
    )
plt.close(fig)

# Figure B: blue-only matrix
xs, ys = [], []
for yi, gene in enumerate(ordered_genes):
    for xi, grp in enumerate(trait_order):
        if membership.loc[gene, grp] == 1:
            xs.append(xi)
            ys.append(yi)

fig, ax = plt.subplots(figsize=(10.2, 20.8))
ax.scatter(xs, ys, s=44, marker="s")
for yi, gene in enumerate(ordered_genes):
    ax.text(7.45, yi, str(int(membership.loc[gene, "n_groups"])),
            va="center", ha="center", fontsize=7.3)
ax.set_xlim(-0.55, 8.05)
ax.set_ylim(len(ordered_genes)-0.5, -0.5)
ax.set_xticks(range(len(trait_order)))
ax.set_xticklabels([f"{g}\n(n={gene_counts_exact[g]})" for g in trait_order])
ax.set_yticks(range(len(ordered_genes)))
ax.set_yticklabels(ordered_genes, fontsize=7.0)
ax.set_xlabel("Exact COVID trait group")
ax.set_title("Exact trait-group membership of 79 AGPT-prioritised coding genes", pad=12)
ax.text(7.45, -1.1, "# groups", ha="center", va="bottom", fontsize=9)
ax.grid(axis="x", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
for ext in ["png", "pdf", "svg"]:
    fig.savefig(outdir / f"Figure_3_5_2B_gene_exact_group_membership_matrix_blue_clean.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure C: clean collapsed A/B/C bar plot with significance only
collapsed_counts = {"A": len(A), "B": len(B), "C": len(C)}
labels = ["A", "B", "C"]
vals = [collapsed_counts[k] for k in labels]
x = np.arange(len(labels))

pair_q = {
    ("A", "B"): float(pair_df.loc[pair_df["comparison"] == "A vs B", "mcnemar_bh_q"].iloc[0]),
    ("A", "C"): float(pair_df.loc[pair_df["comparison"] == "A vs C", "mcnemar_bh_q"].iloc[0]),
    ("B", "C"): float(pair_df.loc[pair_df["comparison"] == "B vs C", "mcnemar_bh_q"].iloc[0]),
}

fig, ax = plt.subplots(figsize=(10.8, 7.2))
bars = ax.bar(x, vals, width=0.32)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Number of coding genes")
ax.set_xlabel("Collapsed original COVID-19 trait label")
ax.set_title("Coding-gene membership after collapsing exact groups to A/B/C", pad=20)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.bar_label(bars, padding=2, fontsize=11)
ax.text(
    0.02, 0.94, meaning_text,
    transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
    bbox=dict(boxstyle="round,pad=0.30", facecolor="white", alpha=0.95, edgecolor="0.7")
)

def add_sig(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5)
    ax.text((x1+x2)/2, y+h+0.8, text, ha="center", va="bottom", fontsize=12)

add_sig(ax, 0, 1, 64.0, 1.4, q_to_label(pair_q[("A", "B")]))
add_sig(ax, 0, 2, 69.5, 1.4, q_to_label(pair_q[("A", "C")]))
add_sig(ax, 1, 2, 75.0, 1.4, q_to_label(pair_q[("B", "C")]))
ax.set_ylim(0, 100)

fig.tight_layout()
for ext in ["png", "pdf", "svg"]:
    fig.savefig(outdir / f"Figure_3_5_2C_collapsed_ABC_barplot_with_significance_clean100.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure D: merged A/B versus C gene-set overlap
merged_ab = A | B
ab_only = sorted(merged_ab - C)
shared = sorted(merged_ab & C)
c_only = sorted(C - merged_ab)

if (len(ab_only), len(shared), len(c_only)) != (41, 25, 13):
    raise RuntimeError("Unexpected merged A/B versus C gene-set counts")

def column_text(values, n_columns):
    rows = int(np.ceil(len(values) / n_columns))
    columns = [values[index * rows:(index + 1) * rows] for index in range(n_columns)]
    height = max(map(len, columns))
    columns = [column + [""] * (height - len(column)) for column in columns]
    return "\n".join("   ".join(items) for items in zip(*columns))

fig, ax = plt.subplots(figsize=(18, 11))
ax.add_patch(Ellipse((-1.6, 0), 6.4, 8.3, facecolor="#7EA6D8", alpha=0.35, edgecolor="#315F9B", lw=2))
ax.add_patch(Ellipse((1.6, 0), 6.4, 8.3, facecolor="#E9A46A", alpha=0.35, edgecolor="#A65E24", lw=2))
ax.text(-3.25, 4.45, "Merged A/B", ha="center", va="bottom", fontsize=15, fontweight="bold")
ax.text(3.25, 4.45, "C susceptibility", ha="center", va="bottom", fontsize=15, fontweight="bold")
ax.text(-3.05, 3.75, f"A/B only (n={len(ab_only)})\n\n{column_text(ab_only, 2)}", ha="center", va="top", fontsize=8.2, linespacing=1.25)
ax.text(0, 3.75, f"Shared (n={len(shared)})\n\n{column_text(shared, 2)}", ha="center", va="top", fontsize=8.2, linespacing=1.25)
ax.text(3.05, 3.75, f"C only (n={len(c_only)})\n\n{column_text(c_only, 1)}", ha="center", va="top", fontsize=8.5, linespacing=1.3)
ax.set_xlim(-6.2, 6.2)
ax.set_ylim(-5.0, 5.2)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Coding-gene overlap between merged severe/hospitalised COVID-19 and susceptibility", pad=14, fontsize=16)
fig.tight_layout()
for ext in ["png", "pdf", "svg"]:
    fig.savefig(
        outdir / f"Figure_3_5_2D_merged_AB_vs_C_gene_overlap.{ext}",
        dpi=300,
        bbox_inches="tight",
    )
plt.close(fig)

# Save stats tables for caption/reporting
pair_df.to_csv(outdir / "Table_3_5_2_pairwise_mcnemar_statistics.tsv", sep="\t", index=False)
abc_matrix.to_csv(outdir / "Table_3_5_2_collapsed_ABC_membership_matrix.tsv", sep="\t")
print("Cochran's Q:", cq)
print(pair_df)
