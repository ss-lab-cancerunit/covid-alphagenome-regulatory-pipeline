# Results-to-code coverage

This inventory maps the Results subsections to source code tracked in this repository.

| Results section | Local source status | Repository location |
|---|---|---|
| 3.1 best-of-SNV screening figure | Covered | `01_threshold_and_event_expansion/scripts/01_plot_best_of_snv_act_rel.py` |
| 3.2 K-sensitivity and selected-cutoff figures | Covered | `01_threshold_and_event_expansion/scripts/02_*` and `03_*` |
| 3.3 event-expansion six-panel figure | Covered | `01_threshold_and_event_expansion/scripts/04_plot_event_expansion.py` |
| 3.4.1 dual position/cCRE knowledge-graph view | Not tracked | Neo4j/Cypher source is not tracked in this repository |
| 3.4.2 position x cCRE models and figure | Covered | `02_position_ccre_models/run_position_ccre_models.py` |
| 3.5.1 lncRNA annotation coverage | External analysis, not tracked | PANTHER, LncSEA and LncTarD queries are not tracked in this repository |
| 3.5.2 trait-aware figures and tests | Covered | `03_trait_comparison/run_trait_comparison.py` |
| 3.5.3 functional enrichment | External analysis, not tracked | Enrichr web analysis is not tracked in this repository |
| 3.5.4 STRING modules | External analysis, not tracked | STRING network analysis is not tracked in this repository |
| 3.5.5 integrated Pathway Commons/shared-SNV connectivity | Covered | `04_gene_connectivity/` |
| 3.6.1 GTEx baseline-expression clustering | Covered | `05_tissue_context_and_breadth/scripts/01_*` |
| 3.6.2 79-gene response breadth and representative top-20 tracks | Covered | `05_tissue_context_and_breadth/scripts/02_*` and `03_*` |

The Neo4j/Cypher source for Results 3.4.1 and the external PANTHER, LncSEA, LncTarD, Enrichr and STRING analysis sessions are not tracked in this repository.
