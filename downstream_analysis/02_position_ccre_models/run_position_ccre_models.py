#!/usr/bin/env python3
"""
3.4 SNV-position × SCREEN-cCRE crossed mixed-effects analysis.

Input:
    3_4_pair_level_374_model_input.tsv

Primary observation:
    one unique SNV × target-gene pair (n = 374)

Models:
    REL:
        log(mean_rel) ~ position + cCRE + analysis_mode
        crossed random intercepts for SNV and gene
        interaction tested by ML likelihood-ratio test
        additive model selected if interaction P >= 0.05

    Track breadth:
        log(retained_track_count + 1) ~ position * cCRE + analysis_mode
        crossed random intercepts for SNV and gene
        interaction tested by ML likelihood-ratio test

Sensitivity:
        negative-binomial regression of raw retained_track_count,
        additive vs position × cCRE interaction.

Adjusted 4 × 4 cells:
    predictions are standardized over the observed composition of
    protein-coding and lncRNA analysis modes.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import statsmodels.formula.api as smf
from patsy import build_design_matrices

HERE = Path(__file__).resolve().parent
INPUT = HERE / "data" / "pair_level_374_model_input.tsv"
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CELLS = OUTDIR / "3_4_adjusted_16_cells.tsv"
OUT_TESTS = OUTDIR / "3_4_model_tests.tsv"
OUT_COEF = OUTDIR / "3_4_model_coefficients.tsv"
OUT_PNG = OUTDIR / "Figure_3_4_adjusted_REL_and_track_breadth_2panel.png"
OUT_SVG = OUTDIR / "Figure_3_4_adjusted_REL_and_track_breadth_2panel.svg"

df = pd.read_csv(INPUT, sep="\t")

position_order_model = ["downstream", "exon", "intron", "upstream"]
ccre_order_model = ["none", "PLS", "pELS", "dELS"]
mode_order = ["protein_coding", "lncRNA"]

df["position"] = pd.Categorical(df["position"], categories=position_order_model)
df["ccre"] = pd.Categorical(df["ccre"], categories=ccre_order_model)
df["analysis_mode"] = pd.Categorical(df["analysis_mode"], categories=mode_order)

df["log_rel"] = np.log(df["mean_rel"])
df["log_track_plus_1"] = np.log1p(df["retained_track_count"])
df["all_group"] = 1

vc = {"gene": "0 + C(gene_id)", "variant": "0 + C(variant_id)"}

def fit_pair(outcome):
    add_formula = f"{outcome} ~ C(position) + C(ccre) + C(analysis_mode)"
    int_formula = f"{outcome} ~ C(position) * C(ccre) + C(analysis_mode)"

    add_ml = smf.mixedlm(
        add_formula, df, groups=df["all_group"],
        vc_formula=vc, re_formula="0"
    ).fit(reml=False, method="lbfgs", maxiter=1000, disp=False)

    int_ml = smf.mixedlm(
        int_formula, df, groups=df["all_group"],
        vc_formula=vc, re_formula="0"
    ).fit(reml=False, method="lbfgs", maxiter=1000, disp=False)

    lr = 2 * (int_ml.llf - add_ml.llf)
    lr_df = len(int_ml.fe_params) - len(add_ml.fe_params)
    lr_p = stats.chi2.sf(lr, lr_df)

    selected_formula = int_formula if lr_p < 0.05 else add_formula
    selected_name = "interaction" if lr_p < 0.05 else "additive"

    final = smf.mixedlm(
        selected_formula, df, groups=df["all_group"],
        vc_formula=vc, re_formula="0"
    ).fit(reml=True, method="lbfgs", maxiter=1000, disp=False)

    return final, {
        "outcome": outcome,
        "additive_logLik_ML": add_ml.llf,
        "interaction_logLik_ML": int_ml.llf,
        "LR_statistic": lr,
        "LR_df": lr_df,
        "interaction_LRT_p": lr_p,
        "selected_model": selected_name,
    }

rel_model, rel_test = fit_pair("log_rel")
track_model, track_test = fit_pair("log_track_plus_1")

nb_add = smf.negativebinomial(
    "retained_track_count ~ C(position) + C(ccre) + C(analysis_mode)", df
).fit(disp=False)
nb_int = smf.negativebinomial(
    "retained_track_count ~ C(position) * C(ccre) + C(analysis_mode)", df
).fit(disp=False)

nb_lr = 2 * (nb_int.llf - nb_add.llf)
nb_df = len(nb_int.params) - len(nb_add.params)
nb_p = stats.chi2.sf(nb_lr, nb_df)

pd.DataFrame([
    rel_test,
    track_test,
    {
        "outcome": "raw_retained_track_count_NB_sensitivity",
        "additive_logLik_ML": nb_add.llf,
        "interaction_logLik_ML": nb_int.llf,
        "LR_statistic": nb_lr,
        "LR_df": nb_df,
        "interaction_LRT_p": nb_p,
        "selected_model": "interaction" if nb_p < 0.05 else "additive",
    }
]).to_csv(OUT_TESTS, sep="\t", index=False)

coef_rows = []
for model_name, model in [("REL", rel_model), ("track_breadth", track_model)]:
    ci = model.conf_int()
    for term, beta in model.fe_params.items():
        coef_rows.append({
            "model": model_name,
            "term": term,
            "coefficient_log_scale": beta,
            "standard_error": model.bse_fe[term],
            "p_value": model.pvalues[term],
            "ci_low_log_scale": ci.loc[term, 0],
            "ci_high_log_scale": ci.loc[term, 1],
            "multiplicative_ratio": np.exp(beta),
            "ratio_ci_low": np.exp(ci.loc[term, 0]),
            "ratio_ci_high": np.exp(ci.loc[term, 1]),
        })
pd.DataFrame(coef_rows).to_csv(OUT_COEF, sep="\t", index=False)

mode_counts = df["analysis_mode"].value_counts().reindex(mode_order)
mode_weights = (mode_counts / mode_counts.sum()).to_dict()

def fixed_design_row(model, new_df):
    design_info = model.model.data.design_info
    return np.asarray(build_design_matrices([design_info], new_df)[0])

def adjusted_prediction(model, position, ccre, transform):
    new = pd.DataFrame({
        "position": [position, position],
        "ccre": [ccre, ccre],
        "analysis_mode": mode_order,
    })
    new["position"] = pd.Categorical(new["position"], categories=position_order_model)
    new["ccre"] = pd.Categorical(new["ccre"], categories=ccre_order_model)
    new["analysis_mode"] = pd.Categorical(new["analysis_mode"], categories=mode_order)

    X = fixed_design_row(model, new)
    w = np.array([mode_weights[m] for m in mode_order])
    xbar = np.average(X, axis=0, weights=w)

    beta = model.fe_params.to_numpy()
    cov = model.cov_params().loc[
        model.fe_params.index, model.fe_params.index
    ].to_numpy()

    eta = float(xbar @ beta)
    se = float(np.sqrt(xbar @ cov @ xbar))
    lo, hi = eta - 1.96 * se, eta + 1.96 * se

    if transform == "rel":
        return tuple(np.exp([eta, lo, hi]))
    return tuple(np.exp([eta, lo, hi]) - 1)

display_positions = ["exon", "intron", "upstream", "downstream"]
display_ccre = ["PLS", "pELS", "dELS", "none"]

rows = []
for pos in display_positions:
    for c in display_ccre:
        rel_est, rel_lo, rel_hi = adjusted_prediction(rel_model, pos, c, "rel")
        trk_est, trk_lo, trk_hi = adjusted_prediction(track_model, pos, c, "track")
        sub = df[(df["position"] == pos) & (df["ccre"] == c)]
        rows.append({
            "position": pos,
            "ccre": c,
            "pair_count": len(sub),
            "unique_snv_count": sub["variant_id"].nunique(),
            "unique_gene_count": sub["gene_id"].nunique(),
            "adjusted_geometric_mean_REL": rel_est,
            "REL_ci_low": rel_lo,
            "REL_ci_high": rel_hi,
            "adjusted_typical_track_count": trk_est,
            "track_ci_low": trk_lo,
            "track_ci_high": trk_hi,
        })

adj = pd.DataFrame(rows)
adj.to_csv(OUT_CELLS, sep="\t", index=False)

rel_mat = adj.pivot(
    index="position", columns="ccre",
    values="adjusted_geometric_mean_REL"
).loc[display_positions, display_ccre]

track_mat = adj.pivot(
    index="position", columns="ccre",
    values="adjusted_typical_track_count"
).loc[display_positions, display_ccre]

grey_blue = LinearSegmentedColormap.from_list(
    "grey_blue",
    ["#f2f2f2", "#d9e2ec", "#9fbad0", "#5f8fb7", "#2f5f8f"]
)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))

for letter, ax, mat, title, cbar_label, fmt in [
    ("A", axes[0], rel_mat, "Adjusted REL",
     "Adjusted geometric mean REL", ".3f"),
    ("B", axes[1], track_mat, "Adjusted track breadth",
     "Adjusted typical track count", ".1f"),
]:
    im = ax.imshow(mat.values, cmap=grey_blue, aspect="equal")
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["PLS", "pELS", "dELS", "None"], fontsize=11)
    ax.set_yticks(range(4))

    if ax is axes[0]:
        ax.set_yticklabels(["Exon", "Intron", "Upstream", "Downstream"], fontsize=11)
        ax.set_ylabel("Position relative to target gene", fontsize=11)
    else:
        ax.set_yticklabels([])

    ax.set_xlabel("SCREEN cCRE class", fontsize=11)

    vals = mat.values
    threshold = (vals.min() + vals.max()) / 2
    for i in range(4):
        for j in range(4):
            val = vals[i, j]
            ax.text(
                j, i, format(val, fmt),
                ha="center", va="center", fontsize=11,
                color="white" if val > threshold else "black"
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.text(
        -0.14, 1.04, letter,
        transform=ax.transAxes,
        fontsize=14, fontweight="bold",
        va="bottom", ha="left"
    )

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_SVG, bbox_inches="tight")
plt.close(fig)

print("Done.")
print(pd.read_csv(OUT_TESTS, sep="\t").to_string(index=False))
