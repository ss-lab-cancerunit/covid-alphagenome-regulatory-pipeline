#!/usr/bin/env python3
"""Recluster informative coding targets across GTEx tissues and SMTS groups.

This repository version retains the Ward-clustering analysis used for the final
73-gene results while removing notebook-only workbook packaging. It reads the
curated source matrices included in ``data`` and writes all results to
``outputs``.
"""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent
DATA = HERE / "data"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DETAILED_INPUT = DATA / "gtex_79_gene_68_tissue_zscore.tsv.gz"
SMTS_INPUT = DATA / "gtex_73_gene_30_smts_zscore.tsv.gz"
K_VALUES = list(range(2, 13))
DETAILED_K = 8
SMTS_K = 10


def load_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    detailed = pd.read_csv(DETAILED_INPUT, sep="\t").set_index("gene_name")
    detailed = detailed.drop(columns=["gene_id", "string_group"], errors="ignore")

    smts = pd.read_csv(SMTS_INPUT, sep="\t").set_index("gene_name")
    common = sorted(set(detailed.index) & set(smts.index))
    detailed = detailed.loc[common].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    smts = smts.loc[common].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    informative = [
        gene
        for gene in common
        if np.nanstd(detailed.loc[gene].to_numpy(float)) > 0
        and np.nanstd(smts.loc[gene].to_numpy(float)) > 0
    ]
    if len(informative) != 73:
        raise RuntimeError(f"Expected 73 informative genes; observed {len(informative)}")
    return detailed.loc[informative], smts.loc[informative]


def relabel_by_first_leaf(labels: np.ndarray, leaf_order: np.ndarray) -> np.ndarray:
    mapping: dict[int, int] = {}
    next_label = 1
    for index in leaf_order:
        old = int(labels[index])
        if old not in mapping:
            mapping[old] = next_label
            next_label += 1
    return np.array([mapping[int(label)] for label in labels], dtype=int)


def evaluate_k(matrix: np.ndarray) -> pd.DataFrame:
    tree = linkage(matrix, method="ward", metric="euclidean")
    rows = []
    for k in K_VALUES:
        labels = fcluster(tree, t=k, criterion="maxclust")
        sizes = sorted(Counter(labels).values())
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(matrix, labels),
                "calinski_harabasz": calinski_harabasz_score(matrix, labels),
                "davies_bouldin": davies_bouldin_score(matrix, labels),
                "minimum_cluster_size": min(sizes),
                "maximum_cluster_size": max(sizes),
                "cluster_sizes": ",".join(map(str, sizes)),
            }
        )
    return pd.DataFrame(rows)


def cluster_frame(frame: pd.DataFrame, selected_k: int) -> dict:
    matrix = frame.to_numpy(float)
    tree = linkage(matrix, method="ward", metric="euclidean")
    global_order = leaves_list(tree)
    labels = relabel_by_first_leaf(
        fcluster(tree, t=selected_k, criterion="maxclust"),
        global_order,
    )

    ordered_indices: list[int] = []
    blocks: list[tuple[int, int, int]] = []
    start = 0
    for cluster in range(1, selected_k + 1):
        indices = np.where(labels == cluster)[0]
        if len(indices) > 1:
            indices = indices[leaves_list(linkage(matrix[indices], method="ward"))]
        ordered_indices.extend(indices.tolist())
        end = start + len(indices)
        blocks.append((cluster, start, end))
        start = end

    ordered_indices_array = np.asarray(ordered_indices)
    ordered_matrix = matrix[ordered_indices_array]
    ordered_genes = frame.index[ordered_indices_array].tolist()
    ordered_labels = labels[ordered_indices_array]

    centroids = np.vstack(
        [matrix[labels == cluster].mean(axis=0) for cluster in range(1, selected_k + 1)]
    )
    feature_order = leaves_list(linkage(centroids.T, method="ward"))

    return {
        "tree": tree,
        "labels": labels,
        "ordered_genes": ordered_genes,
        "ordered_labels": ordered_labels,
        "ordered_matrix": ordered_matrix,
        "blocks": blocks,
        "feature_order": feature_order,
    }


def write_results(prefix: str, frame: pd.DataFrame, result: dict, k_table: pd.DataFrame) -> None:
    k_table.to_csv(OUT / f"{prefix}_k_evaluation.tsv", sep="\t", index=False)

    assignments = frame.copy()
    assignments.insert(0, "cluster", result["labels"])
    assignments.index.name = "gene_name"
    assignments.to_csv(OUT / f"{prefix}_gene_assignments.tsv", sep="\t")

    reordered = pd.DataFrame(
        result["ordered_matrix"][:, result["feature_order"]],
        index=result["ordered_genes"],
        columns=frame.columns[result["feature_order"]],
    )
    reordered.insert(0, "cluster", result["ordered_labels"])
    reordered.index.name = "gene_name"
    reordered.to_csv(OUT / f"{prefix}_reordered_matrix.tsv", sep="\t")

    summary_rows = []
    for cluster in sorted(set(result["labels"])):
        members = frame.index[result["labels"] == cluster].tolist()
        centroid = frame.loc[members].mean(axis=0).sort_values(ascending=False)
        summary_rows.append(
            {
                "cluster": cluster,
                "n_genes": len(members),
                "genes": "; ".join(members),
                "top_5_positive_tissues": "; ".join(
                    f"{name} ({value:.2f})" for name, value in centroid.head(5).items()
                ),
                "bottom_3_tissues": "; ".join(
                    f"{name} ({value:.2f})" for name, value in centroid.tail(3).items()
                ),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        OUT / f"{prefix}_cluster_summary.tsv", sep="\t", index=False
    )


def plot_k(table: pd.DataFrame, selected_k: int, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["k"], table["silhouette"], marker="o")
    selected = table.loc[table["k"] == selected_k, "silhouette"].iloc[0]
    ax.scatter([selected_k], [selected], s=90, zorder=3)
    ax.axvline(selected_k, linestyle="--", linewidth=1)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title(title)
    ax.set_xticks(K_VALUES)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dendrogram(tree: np.ndarray, genes: list[str], title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    dendrogram(tree, labels=genes, leaf_rotation=90, leaf_font_size=7, ax=ax)
    ax.set_xlabel("Genes")
    ax.set_ylabel("Ward linkage distance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    frame: pd.DataFrame,
    result: dict,
    title: str,
    output: Path,
) -> None:
    matrix = result["ordered_matrix"][:, result["feature_order"]]
    features = frame.columns[result["feature_order"]]
    maximum = float(np.nanmax(np.abs(matrix)))
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
    fig, ax = plt.subplots(
        figsize=(max(15, len(features) * 0.24), max(15, len(frame) * 0.19))
    )
    image = ax.imshow(matrix, aspect="auto", cmap="bwr", norm=norm)
    ax.set_xticks(range(len(features)), features, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(frame)), result["ordered_genes"], fontsize=8)
    for _, _, end in result["blocks"][:-1]:
        ax.axhline(end - 0.5, linewidth=2)
    ax.set_xlabel("GTEx tissue or SMTS category")
    ax.set_ylabel("GTEx-informative coding target genes")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Gene-wise tissue-expression z-score")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compare_partitions(detailed: pd.DataFrame, detailed_result: dict, smts_result: dict) -> None:
    detailed_labels = pd.Series(detailed_result["labels"], index=detailed.index)
    smts_labels = pd.Series(smts_result["labels"], index=detailed.index)
    contingency = pd.crosstab(detailed_labels, smts_labels)
    contingency.index.name = "detailed_cluster"
    contingency.columns = [f"SMTS_cluster_{column}" for column in contingency.columns]
    contingency.to_csv(OUT / "detailed_vs_smts_contingency.tsv", sep="\t")

    metrics = pd.DataFrame(
        [
            {
                "comparison": "73-gene detailed-tissue k=8 vs 73-gene SMTS k=10",
                "n_genes": len(detailed),
                "adjusted_rand_index": adjusted_rand_score(
                    detailed_result["labels"], smts_result["labels"]
                ),
                "normalized_mutual_information": normalized_mutual_info_score(
                    detailed_result["labels"], smts_result["labels"]
                ),
            }
        ]
    )
    metrics.to_csv(OUT / "partition_similarity_summary.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(contingency.to_numpy(), aspect="auto")
    ax.set_xticks(range(contingency.shape[1]), contingency.columns, rotation=45, ha="right")
    ax.set_yticks(range(contingency.shape[0]), contingency.index)
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            ax.text(j, i, int(contingency.iloc[i, j]), ha="center", va="center")
    ax.set_xlabel("73 × 30 official-SMTS clustering")
    ax.set_ylabel("73 × 68 detailed-tissue clustering")
    ax.set_title("Detailed-tissue versus SMTS gene partitions")
    fig.colorbar(image, ax=ax, label="Gene count")
    fig.tight_layout()
    fig.savefig(OUT / "detailed_vs_smts_contingency.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    detailed, smts = load_matrices()
    detailed_k = evaluate_k(detailed.to_numpy(float))
    smts_k = evaluate_k(smts.to_numpy(float))
    detailed_result = cluster_frame(detailed, DETAILED_K)
    smts_result = cluster_frame(smts, SMTS_K)

    write_results("detailed_73x68", detailed, detailed_result, detailed_k)
    write_results("smts_73x30", smts, smts_result, smts_k)
    plot_k(detailed_k, DETAILED_K, "K evaluation: 73 genes × 68 GTEx tissues", OUT / "detailed_k_selection.png")
    plot_k(smts_k, SMTS_K, "K evaluation: 73 genes × 30 GTEx SMTS", OUT / "smts_k_selection.png")
    plot_dendrogram(detailed_result["tree"], detailed.index.tolist(), "Ward dendrogram: detailed GTEx tissues", OUT / "detailed_dendrogram.png")
    plot_dendrogram(smts_result["tree"], smts.index.tolist(), "Ward dendrogram: GTEx SMTS", OUT / "smts_dendrogram.png")
    plot_heatmap(detailed, detailed_result, "73 informative genes × 68 GTEx tissues (k=8)", OUT / "detailed_reordered_heatmap.png")
    plot_heatmap(smts, smts_result, "73 informative genes × 30 GTEx SMTS (k=10)", OUT / "smts_reordered_heatmap.png")
    compare_partitions(detailed, detailed_result, smts_result)
    print(f"Completed GTEx reclustering; outputs written to {OUT}")


if __name__ == "__main__":
    main()
