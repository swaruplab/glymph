# 20261116-2010_glymph-datasets
Request: "Keep running all the tiers ... Download all the datasets as needed, and let me know if you cannot get access to a dataset. Only once you have all the datasets ready, then you can start parallel benchmarks." This run gathers and prepares every dataset; it scores nothing.
Routing: HPC3 for storage, alignment and ingestion (data lives there; STAR needs about 40 GB). The office Studio only drives the downloads from tmux so they survive the laptop sleeping; no data passes through it or through the MacBook.
Resources: downloads through hpc3-xfer (wget); STAR index 32 cores, 110 GB, standard; STARsolo 24 cores, 80 GB; ingestion array 4 cores, 64 GB per library. Account vswarup_lab.   Job/tmux: see jobs.tsv; Studio tmux sessions glymph_dl, glymph_dl2, glymph_dl3.
Inputs: config/datasets_manifest.tsv (70 files, about 45 GB), config/datasets.tsv (43 libraries).          Env: scanpy env on HPC3 for ingestion, module star/2.7.10a for alignment.
Started: 2026-11-16 20:10 EST  Finished: 2026-11-16 22:19 EST  Status: DONE
Actual:
Outputs: standard inputs on HPC3 under /dfs7/swaruplab/vswarup/compute/glymph/data/std (one .h5ad, cell list, summary and rank plot per library); summaries and rank plots copied here when done. Large data is listed in data/DATA_LOCATIONS.md.
Notes: see decisions.tsv.
