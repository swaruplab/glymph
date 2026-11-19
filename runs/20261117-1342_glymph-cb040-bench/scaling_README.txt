scaling.tsv: one 20,000-cell library (hgmm_20k) copied 1, 2, 4 and 8 times, one A30 (24 GB). Time and memory only, never accuracy.
peak_gpu_gb: the largest amount of GPU memory the tool's tensors occupied (PyTorch's own figure, from the Glymph run report; CellBender does not report one).
gpu_held_gb: the largest amount the process held on the card as nvidia-smi saw it, which includes blocks PyTorch keeps for reuse.
