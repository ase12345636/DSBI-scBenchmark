#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(argparse)
  library(Matrix)
  library(S4Vectors)
  library(SingleCellExperiment)
  library(SingleR)
})

parser <- ArgumentParser(description = "SingleR annotation benchmark")
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
  matrix <- as(readMM(file.path(input_dir, "matrix.mtx")), "dgCMatrix")
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

make_sce <- function(data, label_col = NULL) {
  sce <- SingleCellExperiment(
    assays = list(logcounts = log_normalize(data$matrix)),
    colData = DataFrame(data$metadata)
  )
  if (!is.null(label_col) && !(label_col %in% colnames(colData(sce)))) {
    stop(sprintf("Label column '%s' not found in training metadata", label_col))
  }
  sce
}

if (args$task == "train_reference") {
  data <- load_exchange(args$input_dir)
  reference <- make_sce(data, args$label_col)
  saveRDS(
    list(reference = reference, label_col = args$label_col),
    file.path(args$save_path, "reference.rds")
  )
  cat(sprintf("[SingleR] reference: %d cells x %d genes\n", ncol(reference), nrow(reference)))
} else {
  if (is.null(args$reference_rds)) stop("--reference_rds is required for prediction")
  reference_object <- readRDS(args$reference_rds)
  reference <- reference_object$reference
  label_col <- reference_object$label_col
  query_data <- load_exchange(args$input_dir)
  query <- make_sce(query_data)

  common_genes <- intersect(rownames(reference), rownames(query))
  if (length(common_genes) < 2) stop("Fewer than two genes overlap reference and query")
  de_method <- if (requireNamespace("scran", quietly = TRUE)) "wilcox" else "classic"
  prediction <- SingleR(
    test = query[common_genes, ],
    ref = reference[common_genes, ],
    labels = as.character(colData(reference)[[label_col]]),
    assay.type.test = "logcounts",
    assay.type.ref = "logcounts",
    de.method = de_method
  )

  labels <- as.character(prediction$pruned.labels)
  fallback <- is.na(labels) | labels == ""
  labels[fallback] <- as.character(prediction$labels[fallback])
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

  scores <- as.data.frame(as.matrix(prediction$scores), check.names = FALSE)
  scores <- cbind(barcode = query_data$barcodes, scores)
  write.csv(
    scores,
    file.path(args$save_path, "scores.csv"),
    row.names = FALSE,
    quote = TRUE
  )
}
