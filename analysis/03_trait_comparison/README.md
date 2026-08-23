# COVID-19 trait comparison

`run_trait_comparison.py` reproduces all locally generated figures used for Results 3.5.2: exact-group SNV/gene counts, the collapsed A/B/C coding-gene comparison, the merged A/B-versus-C overlap figure and the supplementary exact-group membership matrix. It also performs the Cochran Q and pairwise exact McNemar tests with Benjamini-Hochberg adjustment.

The exact gene sets are embedded in the script because they are the final curated results being compared, not an external database query.
