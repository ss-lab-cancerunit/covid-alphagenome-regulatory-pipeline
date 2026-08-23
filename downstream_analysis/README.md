# Downstream analysis and figure generation

This directory contains the post-prediction analyses used after AlphaGenome-pytorch screening, post-processing and one-dimensional K-means threshold learning. The numbered folders follow the order of the Results chapter.

## Contents

1. `01_threshold_and_event_expansion`: best-of-SNV plots, learned REL cut-offs, K-sensitivity and event expansion.
2. `02_position_ccre_models`: position and candidate cis-regulatory element models for 374 retained SNV-gene pairs.
3. `03_trait_comparison`: exact and collapsed COVID-19 trait-group comparisons (Results 3.5.2).
4. `04_gene_connectivity`: direct Pathway Commons and shared-SNV coding-gene connectivity (Results 3.5.5).
5. `05_tissue_context_and_breadth`: GTEx baseline-expression clustering and AGPT RNA-track breadth/context figures (Results 3.6).

Each script resolves paths relative to its own folder, so it can be run from any working directory. Generated files are written to the corresponding `outputs/` directory and are ignored by Git.

## Environment

Install the main repository environment, then add the downstream-analysis dependencies if needed:

```bash
python -m pip install -r downstream_analysis/requirements-analysis.txt
```

## Recommended execution order

```bash
python downstream_analysis/01_threshold_and_event_expansion/scripts/01_plot_best_of_snv_act_rel.py
python downstream_analysis/01_threshold_and_event_expansion/scripts/02_plot_clustered_rel_cutoffs.py
python downstream_analysis/01_threshold_and_event_expansion/scripts/03_plot_first_rel_cutoff_across_k.py
python downstream_analysis/01_threshold_and_event_expansion/scripts/04_plot_event_expansion.py
python downstream_analysis/02_position_ccre_models/run_position_ccre_models.py
python downstream_analysis/03_trait_comparison/run_trait_comparison.py
python downstream_analysis/04_gene_connectivity/prepare_snv_gene_edges.py
python downstream_analysis/04_gene_connectivity/run_gene_connectivity.py
python downstream_analysis/05_tissue_context_and_breadth/scripts/01_recluster_gtex_73_informative_genes.py
python downstream_analysis/05_tissue_context_and_breadth/scripts/02_plot_gene_track_rel_skyline.py
python downstream_analysis/05_tissue_context_and_breadth/scripts/03_plot_representative_gene_top20_rel.py
```

## Scope and provenance

The included tables are compact analysis inputs rather than raw AlphaGenome prediction tracks. Input provenance is documented in each subfolder README. The upstream GTEx retrieval and annotation workflow is not tracked in this repository. Folder 05 starts from the curated GTEx expression matrices used in the reported analyses.

See `RESULTS_SCRIPT_COVERAGE.md` for the mapping between Results subsections, figures and executable source code.
