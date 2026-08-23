# Threshold learning and event expansion

The four scripts generate the best-of-SNV ACT-REL figure, the K-means-derived cut-off figure, the K-sensitivity figure and the six-panel event-expansion figure. Every panel has its own x-axis and y-axis title.

The eight compressed input tables contain the formal coding and lncRNA screening and expansion results used in the figures. Compressed copies are stored in `data/`.

Input provenance:

- coding best-of-SNV data: formal coding eQTL and GWAS screening results;
- lncRNA best-of-SNV data: formal lncRNA eQTL and GWAS screening results;
- coding and lncRNA expanded events: threshold-filtered event tables with ALT-oriented and risk-oriented directions.

The numerical K=3, K=4 and K=5 first-cut-off summary plotted by script 03 is the compact summary used in the Methods and Results; it is embedded directly in that script.
