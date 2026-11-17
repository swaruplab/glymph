#!/usr/bin/env Rscript
# SoupX and DecontX on one library, at their documented defaults.
#
#   Rscript run_r_baselines.R --tool soupx|decontx|decontx_bg \
#       --input out/prep/<name>.raw_v3.h5 --cells out/prep/<name>.cells.txt --output out/<tool>/<name>.h5
#
# Input: the raw matrix in the Cell Ranger v3 HDF5 layout and the cell barcodes, both written by
# prep_inputs.py. Output: the corrected cells x genes counts in the same HDF5 layout, values kept
# exactly as the tool returns them (SoupX and DecontX return fractional counts by default).
#
#   soupx       SoupChannel(all droplets, cells) + Seurat clusters + autoEstCont + adjustCounts.
#               If autoEstCont finds no marker genes (common in cell lines), it is retried with
#               the looser settings the SoupX documentation suggests; the log says which ran.
#   decontx     decontX(cells), its own clustering, no background.
#   decontx_bg  decontX(cells, background = all other droplets).

suppressPackageStartupMessages({
  library(Matrix)
  library(rhdf5)
  library(optparse)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option("--tool", type = "character"),
  make_option("--input", type = "character"),
  make_option("--cells", type = "character"),
  make_option("--output", type = "character"),
  make_option("--seed", type = "integer", default = 1)
)))
set.seed(opt$seed)

read_10x <- function(path) {
  rd <- function(name) h5read(path, name, bit64conversion = "double")
  shp <- as.integer(rd("matrix/shape"))
  feat <- list(
    id = as.character(rd("matrix/features/id")),
    name = as.character(rd("matrix/features/name")),
    genome = as.character(rd("matrix/features/genome"))
  )
  m <- sparseMatrix(
    i = as.integer(rd("matrix/indices")) + 1L, p = as.numeric(rd("matrix/indptr")),
    x = as.numeric(rd("matrix/data")), dims = shp,
    dimnames = list(feat$id, as.character(rd("matrix/barcodes")))
  )
  list(m = m, feat = feat)
}

write_10x <- function(path, m, feat) {
  stopifnot(identical(rownames(m), feat$id))
  m <- as(as(m, "CsparseMatrix"), "generalMatrix")
  m <- drop0(m)
  if (file.exists(path)) file.remove(path)
  h5createFile(path)
  h5createGroup(path, "matrix")
  h5createGroup(path, "matrix/features")
  h5write(as.numeric(m@x), path, "matrix/data")
  h5write(as.integer(m@i), path, "matrix/indices")
  h5write(as.integer(m@p), path, "matrix/indptr")
  h5write(as.integer(dim(m)), path, "matrix/shape")
  h5write(colnames(m), path, "matrix/barcodes")
  h5write(feat$id, path, "matrix/features/id")
  h5write(feat$name, path, "matrix/features/name")
  h5write(feat$genome, path, "matrix/features/genome")
  h5write(rep("Gene Expression", length(feat$id)), path, "matrix/features/feature_type")
  H5close()
}

seurat_clusters <- function(toc) {
  suppressPackageStartupMessages(library(Seurat))
  so <- CreateSeuratObject(toc)
  so <- NormalizeData(so, verbose = FALSE)
  so <- FindVariableFeatures(so, nfeatures = 2000, verbose = FALSE)
  so <- ScaleData(so, verbose = FALSE)
  so <- RunPCA(so, npcs = 30, verbose = FALSE)
  so <- FindNeighbors(so, dims = 1:30, verbose = FALSE)
  so <- FindClusters(so, resolution = 1.0, verbose = FALSE)
  setNames(as.character(so$seurat_clusters), colnames(toc))
}

t0 <- Sys.time()
inp <- read_10x(opt$input)
tod <- inp$m
cells <- readLines(opt$cells)
cells <- cells[nzchar(cells)]
stopifnot(all(cells %in% colnames(tod)))
toc <- tod[, cells]
message(sprintf("[%s] %d droplets, %d cells, %d genes", opt$tool, ncol(tod), ncol(toc), nrow(tod)))

if (opt$tool == "soupx") {
  suppressPackageStartupMessages(library(SoupX))
  cl <- seurat_clusters(toc)
  message(sprintf("[soupx] %d Seurat clusters", length(unique(cl))))
  sc <- SoupChannel(tod, toc)
  sc <- setClusters(sc, cl)
  est <- tryCatch(autoEstCont(sc, doPlot = FALSE), error = function(e) {
    message("[soupx] autoEstCont at defaults failed: ", conditionMessage(e))
    NULL
  })
  mode <- "autoEstCont defaults"
  if (is.null(est)) {
    est <- autoEstCont(sc, doPlot = FALSE, tfidfMin = 0.5, soupQuantile = 0.5, forceAccept = TRUE)
    mode <- "autoEstCont with tfidfMin 0.5, soupQuantile 0.5"
  }
  message(sprintf("[soupx] %s; contamination fraction rho = %.4f", mode, est$metaData$rho[1]))
  out <- adjustCounts(est)
} else if (opt$tool %in% c("decontx", "decontx_bg")) {
  suppressPackageStartupMessages(library(decontX))
  if (opt$tool == "decontx_bg") {
    bg <- tod[, setdiff(colnames(tod), cells)]
    res <- decontX(x = toc, background = bg, seed = 12345)
  } else {
    res <- decontX(x = toc, seed = 12345)
  }
  message(sprintf("[%s] median estimated contamination = %.4f", opt$tool, median(res$contamination)))
  out <- res$decontXcounts
  dimnames(out) <- dimnames(toc)
} else {
  stop("unknown --tool ", opt$tool)
}

dir.create(dirname(opt$output), showWarnings = FALSE, recursive = TRUE)
write_10x(opt$output, out, inp$feat)
message(sprintf("[%s] wrote %s; removed %.3f%% of counts; %.1f s", opt$tool, opt$output,
                100 * (1 - sum(out) / sum(toc)), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
