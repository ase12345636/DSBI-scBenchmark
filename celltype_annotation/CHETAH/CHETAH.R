#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(argparse)
  library(CHETAH)
  library(Matrix)
  library(S4Vectors)
  library(SingleCellExperiment)
})

parser <- ArgumentParser(description = "CHETAH annotation benchmark")
parser$add_argument("--task", required = TRUE, choices = c("train_reference", "predict"))
parser$add_argument("--input_dir", required = TRUE)
parser$add_argument("--reference_rds")
parser$add_argument("--save_path", required = TRUE)
parser$add_argument("--label_col", required = TRUE)
parser$add_argument("--seed", type = "integer", default = 42)
args <- parser$parse_args()

set.seed(args$seed)
dir.create(args$save_path, recursive = TRUE, showWarnings = FALSE)

load_exchange <- function(input_dir) {
  matrix <- as(readMM(file.path(input_dir, "matrix.mtx")), "CsparseMatrix")
  barcodes <- read.csv(file.path(input_dir, "barcodes.csv"), check.names = FALSE)$barcode
  features <- read.csv(file.path(input_dir, "features.csv"), check.names = FALSE)$feature
  metadata <- read.csv(
    file.path(input_dir, "metadata.csv"), row.names = 1, check.names = FALSE
  )
  rownames(matrix) <- features
  colnames(matrix) <- barcodes
  list(matrix = matrix, metadata = metadata, barcodes = barcodes)
}

log_normalize <- function(matrix) {
  if (length(matrix@x) > 0 && any(abs(matrix@x - round(matrix@x)) > 1e-6)) return(matrix)
  library_sizes <- Matrix::colSums(matrix)
  library_sizes[library_sizes == 0] <- 1
  normalized <- t(t(matrix) / library_sizes * 10000)
  normalized@x <- log1p(normalized@x)
  normalized
}

make_query_sce <- function(data) {
  SingleCellExperiment(
    assays = list(logcounts = log_normalize(data$matrix)),
    colData = DataFrame(data$metadata)
  )
}

MAX_REFERENCE_CELLS_PER_TYPE <- 200L

make_reference_cells <- function(matrix, labels) {
  labels <- as.character(labels)
  groups <- split(seq_along(labels), labels)
  selected <- unlist(lapply(groups, function(indices) {
    if (length(indices) <= MAX_REFERENCE_CELLS_PER_TYPE) return(indices)
    indices[sample.int(length(indices), MAX_REFERENCE_CELLS_PER_TYPE)]
  }), use.names = FALSE)
  selected <- sort(selected)
  normalized <- log_normalize(matrix[, selected, drop = FALSE])

  SingleCellExperiment(
    assays = list(logcounts = normalized),
    colData = DataFrame(
      celltypes = labels[selected], row.names = colnames(normalized)
    )
  )
}

if (args$task == "train_reference") {
  data <- load_exchange(args$input_dir)
  if (!(args$label_col %in% colnames(data$metadata))) {
    stop(sprintf("Label column '%s' not found in training metadata", args$label_col))
  }
  original_cells <- ncol(data$matrix)
  reference <- make_reference_cells(
    data$matrix, data$metadata[[args$label_col]]
  )
  saveRDS(reference, file.path(args$save_path, "reference.rds"))
  cat(sprintf(
    "[CHETAH] stratified reference sampling: %d -> %d cells (%d genes)\n",
    original_cells, ncol(reference), nrow(reference)
  ))
} else {
  if (is.null(args$reference_rds)) stop("--reference_rds is required for prediction")
  reference <- readRDS(args$reference_rds)

  query_data <- load_exchange(args$input_dir)
  query <- make_query_sce(query_data)
  common_genes <- intersect(rownames(reference), rownames(query))
  if (length(common_genes) < 2) stop("Fewer than two genes overlap reference and query")

  chunk_size <- 10000L
  starts <- seq.int(1L, ncol(query), by = chunk_size)
  if (length(starts) > 1L && ncol(query) - tail(starts, 1L) + 1L < 100L) {
    starts <- head(starts, -1L)
  }
  if (length(starts) > 1L) {
    cat(sprintf(
      "[CHETAH] predicting %d query cells in %d chunks\n",
      ncol(query), length(starts)
    ))
  }

  label_parts <- vector("list", length(starts))
  for (i in seq_along(starts)) {
    first <- starts[[i]]
    last <- if (i < length(starts)) starts[[i + 1L]] - 1L else ncol(query)
    annotated <- CHETAHclassifier(
      input = query[common_genes, first:last, drop = FALSE],
      ref_cells = reference[common_genes, ],
      input_c = "logcounts",
      ref_c = "logcounts"
    )
    label_parts[[i]] <- as.character(annotated$celltype_CHETAH)
    rm(annotated)
    gc(verbose = FALSE)
  }
  labels <- unlist(label_parts, use.names = FALSE)
  labels[is.na(labels) | labels == ""] <- "Unknown"
  write.csv(
    data.frame(
      barcode = query_data$barcodes,
      prediction = labels,
      check.names = FALSE
    ),
    file.path(args$save_path, "predictions.csv"),
    row.names = FALSE,
    quote = TRUE
  )
}
