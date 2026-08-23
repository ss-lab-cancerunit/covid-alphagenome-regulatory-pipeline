# Coding-gene connectivity

`prepare_snv_gene_edges.py` derives 248 unique SNV-gene pairs from the 6,625 coding expansion events stored in folder 01. It checks the expected 187 SNVs and 79 genes before writing `data/edges_snv_gene.csv`.

`run_gene_connectivity.py` combines shared-SNV relationships with the 34 curated Pathway Commons coding-gene edges used in the analysis, ranks genes by unique direct and indirect neighbours and generates the top-20 tables and figures.

The compact edge table is included for convenience, but it can be regenerated at any time with the preparation script.
