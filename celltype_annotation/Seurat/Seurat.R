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


MAX_REFERENCE_CELLS <- 50000L
QUERY_CHUNK_SIZE <- 20000L

stratified_reference_indices <- function(labels, max_cells) {
  labels <- as.character(labels)
  if (length(labels) <= max_cells) return(seq_along(labels))

  groups <- split(seq_along(labels), labels)
  quota <- max(1L, floor(max_cells / length(groups)))
  selected <- unlist(lapply(groups, function(indices) {
    if (length(indices) <= quota) return(indices)
    indices[sample.int(length(indices), quota)]
  }), use.names = FALSE)

  slots <- max_cells - length(selected)
  if (slots > 0L) {
    remaining <- setdiff(seq_along(labels), selected)
    take <- min(slots, length(remaining))
    if (take > 0L) {
      selected <- c(
        selected, remaining[sample.int(length(remaining), take)]
      )
    }
  }
  sort(selected)
}

if (args$task == "train_reference") {
  data <- load_exchange(args$input_dir)
  if (!(args$label_col %in% colnames(data$metadata))) {
    stop(sprintf("Label column '%s' not found in training metadata", args$label_col))
  }
  original_cells <- ncol(data$matrix)
  keep <- stratified_reference_indices(
    data$metadata[[args$label_col]], MAX_REFERENCE_CELLS
  )
  if (length(keep) < original_cells) {
    cat(sprintf(
      "[Seurat] stratified reference sampling: %d -> %d cells\n",
      original_cells, length(keep)
    ))
    data$matrix <- data$matrix[, keep, drop = FALSE]
    data$metadata <- data$metadata[keep, , drop = FALSE]
    data$barcodes <- data$barcodes[keep]
    gc(verbose = FALSE)
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
  starts <- seq.int(1L, ncol(query_data$matrix), by = QUERY_CHUNK_SIZE)
  if (
    length(starts) > 1L &&
    ncol(query_data$matrix) - tail(starts, 1L) + 1L < 100L
  ) {
    starts <- head(starts, -1L)
  }
  if (length(starts) > 1L) {
    cat(sprintf(
      "[Seurat] predicting %d query cells in %d chunks\n",
      ncol(query_data$matrix), length(starts)
    ))
  }

  label_parts <- vector("list", length(starts))
  score_parts <- vector("list", length(starts))
  for (i in seq_along(starts)) {
    first <- starts[[i]]
    last <- if (i < length(starts)) {
      starts[[i + 1L]] - 1L
    } else {
      ncol(query_data$matrix)
    }
    indices <- first:last
    query <- CreateSeuratObject(
      counts = query_data$matrix[, indices, drop = FALSE],
      meta.data = query_data$metadata[indices, , drop = FALSE]
    )
    query <- NormalizeData(query, verbose = FALSE)
    common_features <- intersect(
      VariableFeatures(reference), rownames(query)
    )
    if (length(common_features) < 2L) {
      stop("Fewer than two variable genes overlap reference and query")
    }
    n_pcs <- min(
      reference@misc$annotation_n_pcs,
      length(common_features) - 1L,
      ncol(query) - 1L
    )
    if (n_pcs < 2L) stop("Query chunk is too small for Seurat transfer")
    neighbor_limit <- min(ncol(query), ncol(reference)) - 1L
    anchors <- FindTransferAnchors(
      reference = reference,
      query = query,
      reference.reduction = "pca",
      features = common_features,
      dims = seq_len(n_pcs),
      k.anchor = min(5L, neighbor_limit),
      k.score = min(30L, neighbor_limit),
      verbose = FALSE
    )
    prediction <- TransferData(
      anchorset = anchors,
      refdata = as.character(reference@meta.data[[args$label_col]]),
      dims = seq_len(n_pcs),
      k.weight = min(50L, neighbor_limit),
      verbose = FALSE
    )

    chunk_labels <- as.character(prediction$predicted.id)
    chunk_labels[is.na(chunk_labels) | chunk_labels == ""] <- "Unknown"
    label_parts[[i]] <- chunk_labels
    score_columns <- grep(
      "^prediction\\.score\\.", colnames(prediction), value = TRUE
    )
    score_columns <- setdiff(score_columns, "prediction.score.max")
    score_parts[[i]] <- as.data.frame(
      prediction[, score_columns, drop = FALSE], check.names = FALSE
    )
    rm(query, anchors, prediction)
    gc(verbose = FALSE)
  }
  labels <- unlist(label_parts, use.names = FALSE)
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

  if (length(score_parts) > 0L && ncol(score_parts[[1L]]) > 0L) {
    scores <- do.call(rbind, score_parts)
    raw_names <- sub("^prediction\\.score\\.", "", colnames(scores))
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
