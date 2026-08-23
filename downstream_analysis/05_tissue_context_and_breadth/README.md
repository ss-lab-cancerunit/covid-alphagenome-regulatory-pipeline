# Tissue context and regulatory breadth

This folder follows Results 3.6. Script 01 reproduces the final clustering of 73 informative coding targets across 68 detailed GTEx tissues and 30 official SMTS groups. It uses Euclidean distance, Ward hierarchical clustering, a K scan from 2 to 12, k=8 for detailed tissues and k=10 for SMTS.

Script 02 derives the maximum REL for every unique coding gene x RNA-track combination directly from the 6,625 expanded coding events and plots the response breadth of all 79 coding targets. Script 03 plots the 20 highest-REL exact RNA tracks for CCR1, ACE2, PSORS1C1 and MAPT. It uses strand suffixes `(+)`, `(-)` and `(.)` rather than ambiguous numerical suffixes.

Inputs:

- `gtex_79_gene_68_tissue_zscore.tsv.gz`: gene-wise GTEx V11 tissue-expression z-scores from the final coding tissue-annotation package;
- `gtex_73_gene_30_smts_zscore.tsv.gz`: the 30-SMTS matrix used for the 73-gene analysis, with the previous cluster label removed;
- `track_metadata_human.csv.gz`: AlphaGenome-pytorch human RNA-track metadata used for biosample labels.

The AGPT response scripts read the coding expanded-event table from `../01_threshold_and_event_expansion/data/`, avoiding a duplicate copy of the same 6,625-event dataset.

Script 01 writes standard TSV and PNG outputs. The upstream GTEx retrieval and annotation workflow is not tracked in this repository. This analysis starts from the curated expression matrices listed above.
