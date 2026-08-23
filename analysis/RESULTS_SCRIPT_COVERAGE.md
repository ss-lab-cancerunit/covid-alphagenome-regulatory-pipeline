# Results-to-code coverage

This inventory was checked against `Result-COVID(4).docx` and all Python, R, R Markdown, notebook and Cypher files present in the uploaded migration and result packages.

| Results section | Local source status | Repository location |
|---|---|---|
| 3.1 best-of-SNV screening figure | Covered | `01_threshold_and_event_expansion/scripts/01_plot_best_of_snv_act_rel.py` |
| 3.2 K-sensitivity and selected-cutoff figures | Covered | `01_threshold_and_event_expansion/scripts/02_*` and `03_*` |
| 3.3 event-expansion six-panel figure | Covered | `01_threshold_and_event_expansion/scripts/04_plot_event_expansion.py` |
| 3.4.1 dual position/cCRE knowledge-graph view | No local Python/R/Cypher source was present in the uploaded packages | Neo4j/Cypher visualisation source should be added later if retained as a reproducibility target |
| 3.4.2 position x cCRE models and figure | Covered | `02_position_ccre_models/run_position_ccre_models.py` |
| 3.5.1 lncRNA annotation coverage | External database analysis | PANTHER, LncSEA and LncTarD results; no local plotting script required |
| 3.5.2 trait-aware figures and tests | Covered, including two figures absent from the first bundle | `03_trait_comparison/run_trait_comparison.py` |
| 3.5.3 functional enrichment | External web analysis | Enrichr output; no local enrichment-calculation script was used |
| 3.5.4 STRING modules | External STRING/network visualisation | STRING result bundle is preserved in the migration package; no local Python source was present |
| 3.5.5 integrated Pathway Commons/shared-SNV connectivity | Covered | `04_gene_connectivity/` |
| 3.6.1 GTEx baseline-expression clustering | Covered | `05_tissue_context_and_breadth/scripts/01_*` |
| 3.6.2 79-gene response breadth and representative top-20 tracks | Covered after adding the two newly supplied scripts | `05_tissue_context_and_breadth/scripts/02_*` and `03_*` |

The only genuine code-provenance gap identified in the uploaded materials is the Neo4j/Cypher source used for the 3.4.1 regulatory-context graph. Enrichr and STRING are externally generated analyses rather than missing Python scripts.
