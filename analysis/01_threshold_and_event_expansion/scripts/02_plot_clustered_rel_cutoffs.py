from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
base = project_dir / 'data'
outdir = project_dir / 'outputs'
outdir.mkdir(parents=True, exist_ok=True)
files = {
    'A': base / 'input_coding_eqtl_211_best_of_snv.tsv.gz',
    'B': base / 'input_lnc_eqtl_211_best_of_snv.tsv.gz',
    'C': base / 'input_coding_gwas_6268_best_of_snv.tsv.gz',
    'D': base / 'input_lnc_gwas_6268_best_of_snv.tsv.gz',
}

panels = [
    {'key': 'A', 'title': 'Protein-coding eQTL-supported SNVs', 'k': 4, 'cutoffs': [12.410, 34.104, 62.815]},
    {'key': 'B', 'title': 'lncRNA eQTL-supported SNVs', 'k': 3, 'cutoffs': [7.490, 37.209]},
    {'key': 'C', 'title': 'Protein-coding GWAS SNVs', 'k': 4, 'cutoffs': [7.261, 24.742, 63.684]},
    {'key': 'D', 'title': 'lncRNA GWAS SNVs', 'k': 3, 'cutoffs': [7.385, 33.999]},
]

dfs = {}
for p in panels:
    df = pd.read_csv(files[p['key']], sep='\t').copy()
    df['REL_percent'] = df['rel'] * 100.0
    df['ACT'] = df['act']
    df['above_first_cutoff'] = df['REL_percent'] >= p['cutoffs'][0]
    dfs[p['key']] = df

xmin, xmax = 1e-4, 3e3
ymin, ymax = 0, 105
grey = '#c8c8c8'
blue = '#2b7bba'

fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.4))
axes = axes.flatten()

for ax, p in zip(axes, panels):
    df = dfs[p['key']]
    bg = df.loc[~df['above_first_cutoff']]
    fg = df.loc[df['above_first_cutoff']]

    ax.scatter(bg['ACT'], bg['REL_percent'], s=10, alpha=0.9, edgecolors='none', color=grey)
    ax.scatter(fg['ACT'], fg['REL_percent'], s=10, alpha=0.95, edgecolors='none', color=blue)

    for c in p['cutoffs']:
        ax.axhline(c, linestyle='--', linewidth=1.2, color='black', alpha=0.8)

    ax.set_xscale('log')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(p['title'], fontsize=12, pad=8)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('RNA expression activity (ACT)', fontsize=11)
    ax.set_ylabel('Relative RNA perturbation (%)', fontsize=11)

    cutoff_text = f"{p['cutoffs'][0]:.2f}%"
    n_bg = len(bg)
    n_fg = len(fg)
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=grey, markersize=6,
               label=f"Below first K={p['k']} cutoff (< {cutoff_text}): n={n_bg}"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=blue, markersize=6,
               label=f"At or above first K={p['k']} cutoff (≥ {cutoff_text}): n={n_fg}"),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=8.5, frameon=True)

for letter, ax in zip(['A', 'B', 'C', 'D'], axes):
    ax.text(-0.12, 1.02, letter, transform=ax.transAxes, fontsize=14, fontweight='bold',
            va='bottom', ha='left')

plt.tight_layout()
plt.savefig(outdir / 'Results_3_2_clustered_REL_cutoff_2x2_v3_all_axes.png', dpi=300, bbox_inches='tight')
plt.savefig(outdir / 'Results_3_2_clustered_REL_cutoff_2x2_v3_all_axes.svg', bbox_inches='tight')
plt.close(fig)
