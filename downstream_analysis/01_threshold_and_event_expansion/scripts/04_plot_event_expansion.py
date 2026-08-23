from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTDIR = PROJECT_DIR / 'outputs'
OUTDIR.mkdir(parents=True, exist_ok=True)

CD_SEED_PATH = DATA_DIR / 'input_coding_gwas_6268_best_of_snv.tsv.gz'
LNC_SEED_PATH = DATA_DIR / 'input_lnc_gwas_6268_best_of_snv.tsv.gz'
CD_EXP_PATH = DATA_DIR / 'input_coding_gwas_expanded_events.tsv.gz'
LNC_EXP_PATH = DATA_DIR / 'input_lnc_gwas_expanded_events.tsv.gz'
CD_RISK_PATH = DATA_DIR / 'input_coding_gwas_expanded_events_risk_oriented.tsv.gz'
LNC_RISK_PATH = DATA_DIR / 'input_lnc_gwas_expanded_events_risk_oriented.tsv.gz'

CD_CUTOFF = 7.261310375538144
LNC_CUTOFF = 7.38546642438381


def norm_track_index(s):
    return pd.to_numeric(s, errors='coerce').astype('Int64')


def prep_seed_vs_new(seed_df, exp_df, cutoff_pct, mode='coding'):
    seed = seed_df.copy()
    exp = exp_df.copy()
    seed['rel_pct'] = seed['rel'] * 100.0
    exp['rel_pct'] = exp['rel'] * 100.0
    seed_above = seed[seed['rel_pct'] >= cutoff_pct].copy()
    seed_below = seed[seed['rel_pct'] < cutoff_pct].copy()
    exp_above = exp[exp['rel_pct'] >= cutoff_pct].copy()

    if mode == 'coding':
        seed_above['track_index_norm'] = norm_track_index(seed_above['track_index'])
        exp_above['track_index_norm'] = norm_track_index(exp_above['track_index'])
        key_cols = ['variant_id', 'gene_id', 'track_index_norm']
        seed_keyset = set(map(tuple, seed_above[key_cols].astype(str).values.tolist()))
        is_seed_event = exp_above[key_cols].astype(str).apply(tuple, axis=1).isin(seed_keyset)
    else:
        key_cols = ['variant_id', 'gene_id', 'track_name', 'biosample_name']
        seed_keyset = set(map(tuple, seed_above[key_cols].astype(str).values.tolist()))
        is_seed_event = exp_above[key_cols].astype(str).apply(tuple, axis=1).isin(seed_keyset)

    new_expanded = exp_above.loc[~is_seed_event].copy()
    return seed_below, seed_above, new_expanded


def prep_alt_direction(exp_df, cutoff_pct):
    df = exp_df.copy()
    df['rel_pct'] = df['rel'] * 100.0
    df['signed_pct'] = df['signed_mean_change'] * 100.0
    above = df[df['rel_pct'] >= cutoff_pct].copy()
    up = above[above['signed_pct'] > 0].copy()
    down = above[above['signed_pct'] < 0].copy()
    return up, down


def prep_risk_direction(risk_df, cutoff_pct):
    df = risk_df.copy()
    df['rel_pct'] = df['rel'] * 100.0
    df['risk_signed_pct'] = df['risk_oriented_signed_rna_relative_expression_change_percent']
    above = df[df['rel_pct'] >= cutoff_pct].copy()
    return above


def add_inset_cbar(fig, ax, mappable, ticks, ticklabels):
    cax = inset_axes(ax, width='3.2%', height='43%', loc='upper right', borderpad=1.1)
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklabels)
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.tick_params(labelsize=9, pad=2)
    return cb


cd_seed = pd.read_csv(CD_SEED_PATH, sep='\t')
lnc_seed = pd.read_csv(LNC_SEED_PATH, sep='\t')

cd_exp = pd.read_csv(CD_EXP_PATH, sep='\t')
lnc_exp = pd.read_csv(LNC_EXP_PATH, sep='\t')
cd_risk = pd.read_csv(CD_RISK_PATH, sep='\t', compression='gzip')
lnc_risk = pd.read_csv(LNC_RISK_PATH, sep='\t', compression='gzip')

cd_seed_below, cd_seed_above, cd_new = prep_seed_vs_new(cd_seed, cd_exp, CD_CUTOFF, mode='coding')
lnc_seed_below, lnc_seed_above, lnc_new = prep_seed_vs_new(lnc_seed, lnc_exp, LNC_CUTOFF, mode='lnc')

cd_alt_up, cd_alt_down = prep_alt_direction(cd_exp, CD_CUTOFF)
lnc_alt_up, lnc_alt_down = prep_alt_direction(lnc_exp, LNC_CUTOFF)

cd_risk_above = prep_risk_direction(cd_risk, CD_CUTOFF)
lnc_risk_above = prep_risk_direction(lnc_risk, LNC_CUTOFF)

fig = plt.figure(figsize=(18, 21), constrained_layout=False)
gs = fig.add_gridspec(3, 2, hspace=0.30, wspace=0.20)
axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(3)])

BG = '#c9c9c9'
SEED = '#2b6cb0'
NEW = '#e53e3e'
GRID_ALPHA = 0.22
LABEL_X = -0.12
LABEL_Y = 1.03
coding_xlim = (8e-5, 2e3)
lnc_xlim = (4e-4, 2e2)
ylim = (0, 105)

# A
ax = axes[0, 0]
ax.scatter(cd_seed_below['act'], cd_seed_below['rel_pct'], s=10, c=BG, edgecolors='none', alpha=0.8)
ax.scatter(cd_seed_above['act'], cd_seed_above['rel_pct'], s=16, c=SEED, edgecolors='none', alpha=0.9)
ax.scatter(cd_new['act'], cd_new['rel_pct'], s=10, c=NEW, edgecolors='none', alpha=0.72)
for y in [CD_CUTOFF, 24.742, 63.684]:
    ax.axhline(y, color='black', linestyle='--', linewidth=1, alpha=0.8)
ax.set_xscale('log')
ax.set_xlim(coding_xlim); ax.set_ylim(ylim)
ax.grid(True, alpha=GRID_ALPHA)
ax.set_title('Protein-coding GWAS SNVs: seed vs expanded events', fontsize=14, pad=8)
ax.legend([
    Line2D([0], [0], marker='o', color='w', markerfacecolor=BG, markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=SEED, markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=NEW, markersize=8),
], [
    f'Below first K=4 cutoff (< {CD_CUTOFF:.2f}%), n={len(cd_seed_below):,}',
    f'Previously selected seed SNV events, n={len(cd_seed_above):,}',
    f'Newly added expanded events, n={len(cd_new):,}'
], loc='upper left', frameon=True, fontsize=10)
ax.text(LABEL_X, LABEL_Y, 'A', transform=ax.transAxes, fontsize=18, fontweight='bold')

# B
ax = axes[0, 1]
ax.scatter(lnc_seed_below['act'], lnc_seed_below['rel_pct'], s=10, c=BG, edgecolors='none', alpha=0.8)
ax.scatter(lnc_seed_above['act'], lnc_seed_above['rel_pct'], s=16, c=SEED, edgecolors='none', alpha=0.9)
ax.scatter(lnc_new['act'], lnc_new['rel_pct'], s=10, c=NEW, edgecolors='none', alpha=0.72)
for y in [LNC_CUTOFF, 34.0]:
    ax.axhline(y, color='black', linestyle='--', linewidth=1, alpha=0.8)
ax.set_xscale('log')
ax.set_xlim(lnc_xlim); ax.set_ylim(ylim)
ax.grid(True, alpha=GRID_ALPHA)
ax.set_title('lncRNA GWAS SNVs: seed vs expanded events', fontsize=14, pad=8)
ax.legend([
    Line2D([0], [0], marker='o', color='w', markerfacecolor=BG, markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=SEED, markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=NEW, markersize=8),
], [
    f'Below first K=3 cutoff (< {LNC_CUTOFF:.2f}%), n={len(lnc_seed_below):,}',
    f'Seed SNV representative events, n={len(lnc_seed_above):,}',
    f'New expanded events above cutoff, n={len(lnc_new):,}'
], loc='upper left', frameon=True, fontsize=10)
ax.text(LABEL_X, LABEL_Y, 'B', transform=ax.transAxes, fontsize=18, fontweight='bold')

# C
ax = axes[1, 0]
cd_norm = TwoSlopeNorm(vmin=-53.9, vcenter=0, vmax=53.9)
sc = ax.scatter(cd_alt_down['act'], cd_alt_down['rel_pct'], s=12, c=cd_alt_down['signed_pct'], cmap='coolwarm',
                norm=cd_norm, edgecolors='none', alpha=0.92)
ax.scatter(cd_alt_up['act'], cd_alt_up['rel_pct'], s=12, c=cd_alt_up['signed_pct'], cmap='coolwarm',
           norm=cd_norm, edgecolors='none', alpha=0.92)
for y in [CD_CUTOFF, 24.742, 63.684]:
    ax.axhline(y, color='black', linestyle='--', linewidth=1, alpha=0.8)
ax.set_xscale('log'); ax.set_xlim(coding_xlim); ax.set_ylim(ylim)
ax.grid(True, alpha=GRID_ALPHA)
ax.set_title('Protein-coding GWAS SNVs: ALT-oriented direction of expanded events', fontsize=14, pad=8)
ax.legend([
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62828', markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2155d6', markersize=8),
], [
    f'ALT up-regulated: n={len(cd_alt_up):,}',
    f'ALT down-regulated: n={len(cd_alt_down):,}'
], loc='upper left', frameon=True, fontsize=10)
add_inset_cbar(fig, ax, sc, [-53.9, 0, 53.9], ['−53.9', '0', '+53.9'])
ax.text(-0.16, LABEL_Y, 'C', transform=ax.transAxes, fontsize=18, fontweight='bold')

# D
ax = axes[1, 1]
lnc_norm = TwoSlopeNorm(vmin=-58.0, vcenter=0, vmax=58.0)
sc = ax.scatter(lnc_alt_down['act'], lnc_alt_down['rel_pct'], s=12, c=lnc_alt_down['signed_pct'], cmap='coolwarm',
                norm=lnc_norm, edgecolors='none', alpha=0.92)
ax.scatter(lnc_alt_up['act'], lnc_alt_up['rel_pct'], s=12, c=lnc_alt_up['signed_pct'], cmap='coolwarm',
           norm=lnc_norm, edgecolors='none', alpha=0.92)
for y in [LNC_CUTOFF, 34.0]:
    ax.axhline(y, color='black', linestyle='--', linewidth=1, alpha=0.8)
ax.set_xscale('log'); ax.set_xlim(lnc_xlim); ax.set_ylim(ylim)
ax.grid(True, alpha=GRID_ALPHA)
ax.set_title('lncRNA GWAS SNVs: ALT-oriented direction of expanded events', fontsize=14, pad=8)
ax.legend([
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62828', markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2155d6', markersize=8),
], [
    f'ALT up-regulated: n={len(lnc_alt_up):,}',
    f'ALT down-regulated: n={len(lnc_alt_down):,}'
], loc='upper left', frameon=True, fontsize=10)
add_inset_cbar(fig, ax, sc, [-58.0, 0, 58.0], ['−58.0', '0', '+58.0'])
ax.text(LABEL_X, LABEL_Y, 'D', transform=ax.transAxes, fontsize=18, fontweight='bold')

# E
ax = axes[2, 0]
cd_alt_risk = cd_risk_above[cd_risk_above['risk_allele_is'].astype(str).str.upper() == 'ALT']
cd_ref_risk = cd_risk_above[cd_risk_above['risk_allele_is'].astype(str).str.upper() == 'REF']
cd_risk_norm = TwoSlopeNorm(vmin=-63.2, vcenter=0, vmax=63.2)
sc = ax.scatter(cd_alt_risk['act'], cd_alt_risk['rel_pct'], s=20, c=cd_alt_risk['risk_signed_pct'], cmap='coolwarm',
                norm=cd_risk_norm, marker='s', edgecolors='none', alpha=0.9)
ax.scatter(cd_ref_risk['act'], cd_ref_risk['rel_pct'], s=20, c=cd_ref_risk['risk_signed_pct'], cmap='coolwarm',
           norm=cd_risk_norm, marker='o', edgecolors='none', alpha=0.82)
ax.axhline(CD_CUTOFF, color='black', linestyle='--', linewidth=1, alpha=0.8)
ax.set_xscale('log'); ax.set_xlim(coding_xlim); ax.set_ylim(ylim)
ax.grid(True, alpha=GRID_ALPHA)
ax.set_title('Protein-coding GWAS SNVs: risk-allele-oriented direction', fontsize=14, pad=8)
ax.legend([
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62828', markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2155d6', markersize=8),
], [
    f'Risk allele = ALT, n={len(cd_alt_risk):,}',
    f'Risk allele = REF, n={len(cd_ref_risk):,}',
    f'Risk increases gene expression, n={(cd_risk_above["risk_signed_pct"] > 0).sum():,}',
    f'Risk decreases gene expression, n={(cd_risk_above["risk_signed_pct"] < 0).sum():,}'
], loc='upper left', frameon=True, fontsize=10)
add_inset_cbar(fig, ax, sc, [-63.2, 0, 63.2], ['−63.2', '0', '+63.2'])
ax.text(LABEL_X, LABEL_Y, 'E', transform=ax.transAxes, fontsize=18, fontweight='bold')

# F
ax = axes[2, 1]
lnc_alt_risk = lnc_risk_above[lnc_risk_above['risk_allele_is'].astype(str).str.upper() == 'ALT']
lnc_ref_risk = lnc_risk_above[lnc_risk_above['risk_allele_is'].astype(str).str.upper() == 'REF']
lnc_risk_norm = TwoSlopeNorm(vmin=-91.1, vcenter=0, vmax=91.1)
sc = ax.scatter(lnc_alt_risk['act'], lnc_alt_risk['rel_pct'], s=20, c=lnc_alt_risk['risk_signed_pct'], cmap='coolwarm',
                norm=lnc_risk_norm, marker='s', edgecolors='none', alpha=0.9)
ax.scatter(lnc_ref_risk['act'], lnc_ref_risk['rel_pct'], s=20, c=lnc_ref_risk['risk_signed_pct'], cmap='coolwarm',
           norm=lnc_risk_norm, marker='o', edgecolors='none', alpha=0.82)
ax.axhline(LNC_CUTOFF, color='black', linestyle='--', linewidth=1, alpha=0.8)
ax.set_xscale('log'); ax.set_xlim(lnc_xlim); ax.set_ylim(ylim)
ax.grid(True, alpha=GRID_ALPHA)
ax.set_title('lncRNA GWAS SNVs: risk-allele-oriented direction', fontsize=14, pad=8)
ax.legend([
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62828', markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2155d6', markersize=8),
], [
    f'Risk allele = ALT, n={len(lnc_alt_risk):,}',
    f'Risk allele = REF, n={len(lnc_ref_risk):,}',
    f'Risk increases gene expression, n={(lnc_risk_above["risk_signed_pct"] > 0).sum():,}',
    f'Risk decreases gene expression, n={(lnc_risk_above["risk_signed_pct"] < 0).sum():,}'
], loc='upper left', frameon=True, fontsize=10)
add_inset_cbar(fig, ax, sc, [-91.1, 0, 91.1], ['−91.1', '0', '+91.1'])
ax.text(LABEL_X, LABEL_Y, 'F', transform=ax.transAxes, fontsize=18, fontweight='bold')

for ax in axes.ravel():
    ax.set_xlabel('RNA expression activity (ACT)', fontsize=12)
    ax.set_ylabel('Relative RNA perturbation (%)', fontsize=12)

png_path = OUTDIR / 'Results_3_3_event_expansion_six_panel_v4_all_axes.png'
svg_path = OUTDIR / 'Results_3_3_event_expansion_six_panel_v4_all_axes.svg'
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(svg_path, bbox_inches='tight')
plt.close(fig)

# source data and script copy
source_tsv = OUTDIR / 'Results_3_3_event_expansion_six_panel_v4_all_axes_source_data.tsv'
source_frames = []
for name, df in [
    ('cd_seed_below', cd_seed_below), ('cd_seed_above', cd_seed_above), ('cd_new_expanded', cd_new),
    ('lnc_seed_below', lnc_seed_below), ('lnc_seed_above', lnc_seed_above), ('lnc_new_expanded', lnc_new),
    ('cd_alt_up', cd_alt_up), ('cd_alt_down', cd_alt_down), ('lnc_alt_up', lnc_alt_up), ('lnc_alt_down', lnc_alt_down),
    ('cd_risk_above', cd_risk_above), ('lnc_risk_above', lnc_risk_above),
]:
    tmp = df.copy()
    tmp.insert(0, 'panel_data_group', name)
    source_frames.append(tmp)
pd.concat(source_frames, ignore_index=True, sort=False).to_csv(source_tsv, sep='\t', index=False)

print(png_path)
print(svg_path)
print(source_tsv)
