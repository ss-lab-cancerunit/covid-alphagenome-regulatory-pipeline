# COVID-19 AlphaGenome-pytorch regulatory pipeline

This repository contains the reproducible code used to screen genome-wide
significant COVID-19 and Long COVID SNVs with AlphaGenome-pytorch, quantify
allele-specific RNA perturbations, learn one-dimensional K-means thresholds,
and expand selected SNVs into SNV-gene-RNA-track regulatory events.

## Pipeline

1. `scripts/01_screen/run_screen.py`: REF/ALT sequence construction, sequential
   AlphaGenome-pytorch inference, RNA tensor extraction, and mode-specific
   event scoring.
2. `scripts/02_postprocess/run_postprocess.py`: batch merging, deduplication,
   QC summaries, and result packaging.
3. `scripts/03_threshold_learning/run_threshold_learning.py`: one-dimensional
   K-means solutions for K=3, K=4, and K=5.
4. The same screening/scoring code is reused with `selection_mode=threshold`
   to retain all gene-by-track events above the selected REL threshold.

Protein-coding and lncRNA analyses use different gene masks and RNA track
policies. They are kept separate until downstream event integration.

## Formal threshold implementations

The protein-coding analysis used deterministic multi-start one-dimensional
K-means. The lncRNA analysis used scikit-learn KMeans with `random_state=42`
and `n_init=10`. Both minimise within-cluster squared error; thresholds are
the midpoints between adjacent sorted cluster centres.

Fisher-Jenks code from an unused development branch is not included in
this clean repository and remains preserved in the original HPC working directory.

## External and large files

Model weights, GENCODE v46 reference files, input tables, results, logs, and
the AlphaGenome-pytorch clone are intentionally excluded from Git. See the
README files in those directories and `configs/analysis_reference.json`.

The validated AlphaGenome-pytorch commit was:

```text
bfa6b9bb03a48bb4f0fc5416ed631156660db234
```

Apply `patches/alphagenome_pytorch_noviz.patch` to a clean checkout of that
commit when the optional visualization dependencies are unavailable.

## HPC use

Submit PBS jobs from the repository root so that `PBS_O_WORKDIR` resolves to
the project directory. Alternatively pass the root explicitly:

```bash
qsub -v PROJECT_ROOT="$PWD" jobs/imperial_hpc/<analysis>/<job>.pbs
```

Before committing, run:

```bash
export PYTHONPATH="$PWD/src"
python -m compileall -q src scripts
python -m unittest discover -s tests -v
python scripts/audit_repository.py
git status --short
```
