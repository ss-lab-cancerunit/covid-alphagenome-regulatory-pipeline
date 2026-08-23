from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent
ANALYSIS_DIR = MODULE_DIR.parent
EXPANDED_FILE = (
    ANALYSIS_DIR
    / '01_threshold_and_event_expansion'
    / 'data'
    / 'input_coding_gwas_expanded_events.tsv.gz'
)
OUTDIR = MODULE_DIR / 'outputs'
OUTDIR.mkdir(parents=True, exist_ok=True)

expanded_df = pd.read_csv(
    EXPANDED_FILE,
    sep='\t',
    usecols=['gene_name', 'variant_id', 'track_name', 'rel'],
)

# One exact gene x RNA-track entry, retaining the largest REL across SNVs.
pair_df = (
    expanded_df.dropna(subset=['gene_name', 'track_name', 'rel'])
    .groupby(['gene_name', 'track_name'], as_index=False)['rel']
    .max()
    .rename(columns={'rel': 'max_rel'})
)

snv_counts = (
    expanded_df.dropna(subset=['gene_name', 'variant_id'])
    .drop_duplicates()
    .groupby('gene_name')['variant_id']
    .nunique()
    .rename('n_snvs')
    .reset_index()
)

gene_order_df = (
    pair_df.groupby('gene_name')
    .agg(
        n_tracks=('track_name', 'nunique'),
        strongest_rel=('max_rel', 'max'),
        mean_rel=('max_rel', 'mean'),
    )
    .reset_index()
    .merge(snv_counts, on='gene_name', how='left')
)
gene_order_df['n_snvs'] = gene_order_df['n_snvs'].fillna(0).astype(int)

gene_order_df = gene_order_df.sort_values(
    ['n_tracks', 'strongest_rel', 'mean_rel', 'gene_name'],
    ascending=[False, False, False, True],
).reset_index(drop=True)

gene_order = gene_order_df['gene_name'].tolist()
panel1_genes = gene_order[:40]
panel2_genes = gene_order[40:]

gene_to_count = dict(zip(gene_order_df['gene_name'], gene_order_df['n_snvs']))
gene_to_label = {g: f'{g} ({gene_to_count.get(g,0)})' for g in gene_order}

track_lists = {}
for gene in gene_order:
    track_lists[gene] = (
        pair_df.loc[pair_df['gene_name'].eq(gene), ['track_name', 'max_rel']]
        .sort_values(['max_rel', 'track_name'], ascending=[False, True])
        .reset_index(drop=True)
    )

panel1_max = max(len(track_lists[g]) for g in panel1_genes)
panel2_display_max = 20

def draw_panel(ax, genes, ymax):
    x = np.arange(len(genes))
    col_width = 0.82

    for i, gene in enumerate(genes):
        vals = track_lists[gene].iloc[:ymax].copy()
        for j, row in vals.iterrows():
            rel = float(row['max_rel'])
            width = col_width * rel
            left = i - width / 2.0
            y = j + 1
            color = 'tab:blue' if j % 2 == 0 else 'tab:orange'
            ax.barh(y, width, left=left, height=0.75, color=color, edgecolor='none')

    ax.set_xlim(-0.6, len(genes) - 0.4)
    ax.set_ylim(0.5, ymax + 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([gene_to_label[g] for g in genes], rotation=90, ha='center', fontsize=7.5)
    ax.set_ylabel('Number of responsive tracks', fontsize=11)
    ax.set_xlabel('Protein-coding target gene (number of linked SNVs)', fontsize=11)
    ax.grid(axis='y', alpha=0.16)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

upper_range = panel1_max
lower_range = panel2_display_max
unit_height = 0.08
fig_height = unit_height * (upper_range + lower_range) + 2.7
fig = plt.figure(figsize=(26, fig_height))
gs = fig.add_gridspec(2, 1, height_ratios=[upper_range, lower_range], hspace=0.18)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

draw_panel(ax1, panel1_genes, panel1_max)
draw_panel(ax2, panel2_genes, panel2_display_max)

fig.suptitle('RNA-track response profiles across 79 coding genes', fontsize=16, y=0.995)
fig.subplots_adjust(left=0.055, right=0.995, top=0.975, bottom=0.09)

out_png = OUTDIR / 'CD_79genes_gene_track_REL_two_panel_with_snvcounts.png'
out_pdf = OUTDIR / 'CD_79genes_gene_track_REL_two_panel_with_snvcounts.pdf'
out_svg = OUTDIR / 'CD_79genes_gene_track_REL_two_panel_with_snvcounts.svg'
out_tsv = OUTDIR / 'gene_order_with_snvcounts.tsv'
fig.savefig(out_png, dpi=300, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_svg, bbox_inches='tight')
plt.close(fig)

gene_order_df[['gene_name','n_snvs','n_tracks','strongest_rel','mean_rel']].to_csv(out_tsv, sep='\t', index=False)
pair_df.to_csv(
    OUTDIR / 'CD_gene_x_track_maxREL_unique_pairs.tsv.gz',
    sep='\t',
    index=False,
    compression='gzip',
)

print(out_png)
print(out_tsv)
