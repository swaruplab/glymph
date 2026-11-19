#!/usr/bin/env python
"""Spliced, unspliced and ambiguous molecule counts per barcode and gene, from a Cell Ranger alignment file.

Why not the file's own RE tag: libraries aligned to a pre-mRNA reference (the kidney nuclei of Janssen et al. 2023)
mark every read inside a gene as exonic. Reads are classified here by position against a real exon annotation:
    unspliced   at least --min-intronic aligned bases lie outside every annotated exon
    spliced     the alignment crosses a splice junction (N in the CIGAR) and is not unspliced
    ambiguous   inside exons, no junction (a mature or a nascent transcript could give it)
A molecule (barcode, UMI, gene as assigned by Cell Ranger: CB, UB, GX) is unspliced if any of its reads is, else
spliced if any read is, else ambiguous. One process per chromosome reads `samtools view` (the file must be indexed).

    python 51_bam_splice_counts.py --bam nuc2_possorted_genome_bam.bam --gtf genes.gtf --h5ad data/std/kidney_nuc2.h5ad \
        --out data/splice/kidney_nuc2 --threads 16

Writes <out>.{unspliced,spliced,ambiguous}.npz (barcodes of the h5ad by its genes, same order) and <out>.qc.tsv.
"""
import argparse
import gzip
import os
import re
import subprocess
import sys
from array import array
from multiprocessing import Pool

import h5py
import numpy as np
import scipy.sparse as sp

CIG = re.compile(r"(\d+)([MIDNSHP=X])")
UMI = str.maketrans("ACGT", "0123")


def h5_strings(f, group, key):
    v = f[group][key]
    if isinstance(v, h5py.Group):  # categorical
        cats = [c.decode() if isinstance(c, bytes) else c for c in v["categories"][:]]
        return [cats[i] for i in v["codes"][:]]
    return [x.decode() if isinstance(x, bytes) else x for x in v[:]]


def load_exons(gtf):
    """Exon intervals per chromosome (names without 'chr', M as MT), 0-based half-open."""
    ex = {}
    op = gzip.open if gtf.endswith(".gz") else open
    with op(gtf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 5)
            if p[2] != "exon":
                continue
            c = p[0][3:] if p[0].startswith("chr") else p[0]
            c = "MT" if c == "M" else c
            ex.setdefault(c, []).append((int(p[3]) - 1, int(p[4])))
    return ex


def one_chrom(args):
    bam, chrom, length, exons, h5ad, min_intronic = args
    nonexon = np.ones(length + 1, dtype=np.int32)
    for a, b in exons:
        nonexon[a + 1:b + 1] = 0
    nonexon[0] = 0
    cs = np.cumsum(nonexon, dtype=np.int32)  # cs[p] = bases outside exons in [0, p)
    del nonexon
    with h5py.File(h5ad, "r") as f:
        bc = {b: i for i, b in enumerate(h5_strings(f, "obs", f["obs"].attrs["_index"]))}
        gi = {g: i for i, g in enumerate(h5_strings(f, "var", "gene_id"))}
    keys, flags = array("q"), array("B")
    n_read = n_used = 0
    cmd = ["samtools", "view", "-F", "0x904", "-e", "[CB] && [UB] && [GX]", bam, chrom]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    for line in p.stdout:
        n_read += 1
        f = line.split("\t")
        cb = ub = gx = None
        for t in f[11:]:
            k = t[:5]
            if k == "CB:Z:":
                cb = t[5:].rstrip("\n")
            elif k == "UB:Z:":
                ub = t[5:].rstrip("\n")
            elif k == "GX:Z:":
                gx = t[5:].rstrip("\n")
        i = bc.get(cb)
        g = gi.get(gx)
        if i is None or g is None or "N" in ub or len(ub) > 12:
            continue
        cigar = f[5]
        pos = int(f[3]) - 1
        if cigar[:-1].isdigit() and cigar[-1] == "M":
            n = int(cigar[:-1])
            out, junction = int(cs[pos + n] - cs[pos]), False
        else:
            out, junction = 0, False
            for n, o in CIG.findall(cigar):
                n = int(n)
                if o in "M=X":
                    e = min(pos + n, length)
                    out += int(cs[e] - cs[min(pos, length)])
                    pos += n
                elif o == "D":
                    pos += n
                elif o == "N":
                    junction = True
                    pos += n
        fl = 1 if out >= min_intronic else (2 if junction else 0)
        keys.append((i << 40) | (g << 24) | int(ub.translate(UMI), 4))
        flags.append(fl)
        n_used += 1
    p.wait()
    k = np.frombuffer(keys, dtype=np.int64)
    fl = np.frombuffer(flags, dtype=np.uint8)
    if k.size == 0:
        return chrom, n_read, 0, 0, np.zeros((0, 2), np.int64), np.zeros((0, 3), np.int32)
    order = np.argsort(k, kind="stable")
    k, fl = k[order], fl[order]
    start = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    u = np.bitwise_or.reduceat(fl & 1, start)            # any unspliced read
    j = np.bitwise_or.reduceat((fl >> 1) & 1, start)     # any junction read
    mol = k[start] >> 24                                # barcode and gene of each molecule
    cls = np.where(u > 0, 0, np.where(j > 0, 1, 2))      # 0 unspliced, 1 spliced, 2 ambiguous
    pair, inv = np.unique(mol, return_inverse=True)
    counts = np.zeros((pair.size, 3), dtype=np.int32)
    np.add.at(counts, (inv, cls), 1)
    return chrom, n_read, n_used, int(start.size), np.stack([pair >> 16, pair & 0xFFFF], axis=1), counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bam", required=True)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--min-intronic", type=int, default=10)
    a = ap.parse_args()
    exons = load_exons(a.gtf)
    hdr = subprocess.run(["samtools", "view", "-H", a.bam], capture_output=True, text=True, check=True).stdout
    chroms = []
    for line in hdr.splitlines():
        if line.startswith("@SQ"):
            d = dict(x.split(":", 1) for x in line.split("\t")[1:])
            if d["SN"] in exons:
                chroms.append((d["SN"], int(d["LN"])))
    print(f"{len(chroms)} chromosomes with annotated exons: {' '.join(c for c, _ in chroms)}", flush=True)
    chroms.sort(key=lambda x: -x[1])
    jobs = [(a.bam, c, n, exons[c], a.h5ad, a.min_intronic) for c, n in chroms]
    with h5py.File(a.h5ad, "r") as f:
        n_bc, n_g = len(f["obs"][f["obs"].attrs["_index"]]), len(f["var"]["gene_id"])
    rows, cols, cnt, qc = [], [], [], []
    with Pool(a.threads) as pool:
        for chrom, n_read, n_used, n_mol, pair, counts in pool.imap_unordered(one_chrom, jobs):
            rows.append(pair[:, 0]); cols.append(pair[:, 1]); cnt.append(counts)
            qc.append((chrom, n_read, n_used, n_mol, *counts.sum(0).tolist()))
            print(f"{chrom}: {n_read:,} reads with barcode, UMI and gene, {n_used:,} on listed barcodes and genes, {n_mol:,} molecules "
                  f"(unspliced {counts[:, 0].sum():,}, spliced {counts[:, 1].sum():,}, ambiguous {counts[:, 2].sum():,})", flush=True)
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(cnt)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for j, name in enumerate(("unspliced", "spliced", "ambiguous")):
        keep = v[:, j] > 0
        sp.save_npz(f"{a.out}.{name}.npz", sp.csr_matrix((v[keep, j], (r[keep], c[keep])), shape=(n_bc, n_g), dtype=np.int32))
    with open(a.out + ".qc.tsv", "w") as f:
        f.write("chrom\treads\treads_used\tmolecules\tunspliced\tspliced\tambiguous\n")
        for q in sorted(qc):
            f.write("\t".join(str(x) for x in q) + "\n")
    tot = v.sum(0)
    print(f"all: {int(tot.sum()):,} molecules; unspliced {100 * tot[0] / tot.sum():.1f}%, spliced {100 * tot[1] / tot.sum():.1f}%, ambiguous {100 * tot[2] / tot.sum():.1f}%")


if __name__ == "__main__":
    sys.exit(main())
