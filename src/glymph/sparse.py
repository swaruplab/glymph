"""A whole library held as CSR on the device, with batches densified there.

Keeping the matrix on the GPU removes the CPU dataloader from the training loop,
which is the usual bottleneck for VAE-sized models.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


class DeviceCSR:
    def __init__(self, X: sp.csr_matrix, device):
        X = sp.csr_matrix(X)
        X.sum_duplicates()
        X.sort_indices()
        self.shape = X.shape
        self.device = device
        self.indptr = torch.from_numpy(X.indptr.astype(np.int64)).to(device)
        # Column indices stay 32-bit on the device (half the memory of the largest array of a big
        # library); take() hands them out as 64-bit, which indexing needs.
        self.indices = torch.from_numpy(X.indices.astype(np.int32)).to(device)
        self.data = torch.from_numpy(X.data.astype(np.float32)).to(device)
        self.col_nnz = np.bincount(X.indices, minlength=X.shape[1])  # entries per gene, on the host

    def take(self, rows: torch.Tensor):
        """Nonzero entries of the given rows as (local_row, column, value, position)."""
        starts = self.indptr[rows]
        lens = self.indptr[rows + 1] - starts
        total = int(lens.sum())
        ar = torch.arange(rows.numel(), device=self.device)
        r = torch.repeat_interleave(ar, lens)
        first = torch.cumsum(lens, 0) - lens
        pos = torch.arange(total, device=self.device) - first[r] + starts[r]
        return r, self.indices[pos].long(), self.data[pos], pos

    def row_chunks(self, rows: torch.Tensor, max_entries: int = 30_000_000):
        """Cut rows into consecutive pieces of at most max_entries nonzero entries: (start, stop) pairs.

        Gathering every entry of a large library at once does not fit on a GPU (80,000 cells are
        about 400 million entries and three arrays of that length), so whole-library sums go
        through these pieces.
        """
        lens = (self.indptr[rows + 1] - self.indptr[rows]).cpu().numpy()
        out, start, acc = [], 0, 0
        for i, n in enumerate(lens):
            if acc and acc + n > max_entries:
                out.append((start, i))
                start, acc = i, 0
            acc += int(n)
        if start < len(lens):
            out.append((start, len(lens)))
        return out

    def group_sums(self, rows: torch.Tensor, group: torch.Tensor, n_groups: int, max_entries: int = 30_000_000):
        """Counts per (group of the row, gene) as float64 [n_groups, n_genes]; group has one label per row."""
        G = self.shape[1]
        X = torch.zeros(n_groups * G, dtype=torch.float64, device=self.device)
        for a, b in self.row_chunks(rows, max_entries):
            r, c, v, _ = self.take(rows[a:b])
            X.index_add_(0, group[a:b][r] * G + c, v.to(torch.float64))
        return X.view(n_groups, G)

    def masked_row_sums(self, rows: torch.Tensor, group: torch.Tensor, mask: torch.Tensor, max_entries: int = 30_000_000):
        """For every row, its counts on the genes that mask[group of the row] selects (float64)."""
        y = torch.zeros(rows.numel(), dtype=torch.float64, device=self.device)
        for a, b in self.row_chunks(rows, max_entries):
            r, c, v, _ = self.take(rows[a:b])
            y[a:b].index_add_(0, r, v.to(torch.float64) * mask[group[a:b][r], c].to(torch.float64))
        return y

    def dense(self, rows: torch.Tensor):
        """Dense [len(rows), n_genes] float32 batch plus its nonzero entries."""
        r, c, v, pos = self.take(rows)
        x = torch.zeros(rows.numel(), self.shape[1], device=self.device)
        x[r, c] = v
        return x, (r, c, v, pos)
