#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(argparse)
  library(Matrix)
  library(Seurat)
  library(SeuratObject)
})

parser <- ArgumentParser(description = "Seurat label-transfer benchmark")
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

if (args$task == "train_reference") {
  data <- load_exchange(args$input_dir)
  if (!(args$label_col %in% colnames(data$metadata))) {
    stop(sprintf("Label column '%s' not found in training metadata", args$label_col))
  }
  reference <- CreateSeuratObject(counts = data$matrix, meta.data = data$metadata)
  reference <- NormalizeData(reference, verbose = FALSE)
  reference <- FindVariableFeatures(
    reference, nfeatures = min(2000, nrow(reference)), verbose = FALSE
  )
  reference <- ScaleData(reference, features = VariableFeatures(reference), verbose = FALSE)
  n_pcs <- min(30, ncol(reference) - 1, length(VariableFeatures(reference)) - 1)
  if (n_pcs < 2) stop("Reference data is too small for Seurat PCA")
  reference <- RunPCA(
    reference, features = VariableFeatures(reference), npcs = n_pcs,
    seed.use = args$seed, verbose = FALSE
  )
  reference@misc$annotation_n_pcs <- n_pcs
  reference@misc$annotation_levels <- sort(unique(as.character(data$metadata[[args$label_col]])))
  saveRDS(reference, file.path(args$save_path, "reference.rds"))
  cat(sprintf("[Seurat] reference: %d cells x %d genes\n", ncol(reference), nrow(reference)))
} else {
  if (is.null(args$reference_rds)) stop("--reference_rds is required for prediction")
  reference <- readRDS(args$reference_rds)
  query_data <- load_exchange(args$input_dir)
  query <- CreateSeuratObject(counts = query_data$matrix, meta.data = query_data$metadata)
  query <- NormalizeData(query, verbose = FALSE)

  common_features <- intersect(VariableFeatures(reference), rownames(query))
  if (length(common_features) < 2) stop("Fewer than two variable genes overlap reference and query")
  n_pcs <- min(reference@misc$annotation_n_pcs, length(common_features) - 1)
  anchors <- FindTransferAnchors(
    reference = reference,
    query = query,
    normalization.method = "LogNormalize",
    reference.reduction = "pca",
    features = common_features,
    dims = seq_len(n_pcs),
    verbose = FALSE
  )
  prediction <- TransferData(
    anchorset = anchors,
    refdata = as.character(reference@meta.data[[args$label_col]]),
    dims = seq_len(n_pcs),
    verbose = FALSE
  )

  labels <- as.character(prediction$predicted.id)
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

  score_columns <- grep("^prediction\\.score\\.", colnames(prediction), value = TRUE)
  score_columns <- setdiff(score_columns, "prediction.score.max")
  if (length(score_columns) > 0) {
    scores <- as.data.frame(prediction[, score_columns, drop = FALSE], check.names = FALSE)
    raw_names <- sub("^prediction\\.score\\.", "", score_columns)
    reference_levels <- reference@misc$annotation_levels
    matched <- match(raw_names, make.names(reference_levels))
    score_names <- raw_names
    score_names[!is.na(matched)] <- reference_levels[matched[!is.na(matched)]]
    colnames(scores) <- score_names
    scores <- cbind(barcode = query_data$barcodes, scores)
    write.csv(
      scores,
      file.path(args$save_path, "scores.csv"),
      row.names = FALSE,
      quote = TRUE
    )
  }
}
