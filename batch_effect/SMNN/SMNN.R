#!/usr/bin/env Rscript
# SMNN 1.2.1 supervised batch correction for the scIB pipeline.
#
# Patched implementation:
#   - SMNN correction still uses all genes and the original input batch order;
#   - core SMNN 1.2.1 correction equations are retained, while adaptive k and
#     step-local shared labels can change pair selection for rare cell types;
#   - mnnpy's parallel adjust_shift_variance() replaces SMNN's serial R loop;
#   - accumulated reference labels now grow together with reference cells;
#   - 2,000 unsupervised integration features are used only for the final PCA.
#
# Called by SMNN.py. Reads matrix.mtx + CSV files and writes a 50-D PCA embedding.

suppressPackageStartupMessages({
  library(SMNN)
  library(Seurat)
  library(SeuratObject)
  library(argparse)
  library(Matrix)
  library(reticulate)
})

# ============================================================
# 1. Constants and helpers
# ============================================================

SMNN_K <- 20L
SMNN_SIGMA <- 1
SMNN_NPCS <- 50L

cores_detected <- parallel::detectCores(logical = TRUE)
if (is.na(cores_detected)) cores_detected <- 1L
N_WORKERS <- max(1L, min(8L, as.integer(cores_detected)))

# The official package helper is not exported, but using it keeps correction-vector
# computation identical to SMNN 1.2.1.
smnn_compute_correction <- getFromNamespace("compute_correction", "SMNN")

if (as.character(utils::packageVersion("SMNN")) != "1.2.1") {
  warning(sprintf(
    "This script was validated against SMNN 1.2.1; installed version is %s",
    as.character(utils::packageVersion("SMNN"))
  ))
}

# Patched supervised SMNN 1.2.1 loop.
# Input matrices retain the official orientation: genes x cells.
SMNNcorrect_patched <- function(
    batches,
    batch.cluster.labels,
    matched.clusters,
    correct.others = FALSE,
    k = 20L,
    sigma = 1,
    cos.norm.in = TRUE,
    cos.norm.out = TRUE,
    var.adj = TRUE,
    order = NULL,
    n.jobs = NULL) {

  nbatches <- length(batches)
  if (nbatches < 2L) {
    stop("SMNN requires at least two batches")
  }

  if (length(batch.cluster.labels) != nbatches) {
    stop("The number of cluster-label vectors must equal the number of batches")
  }

  if (is.null(n.jobs)) {
    n.jobs <- parallel::detectCores(logical = TRUE)
  }
  if (is.na(n.jobs)) n.jobs <- 1L
  n.jobs <- max(1L, as.integer(n.jobs))

  if (length(k) != 1L || is.na(k) || k < 2L) {
    stop("k must be an integer >= 2")
  }
  k <- as.integer(k)

  # Validate matrices and labels before entering Python.
  reference_genes <- rownames(batches[[1]])
  if (is.null(reference_genes)) {
    stop("Every batch matrix must have gene row names")
  }

  for (i in seq_len(nbatches)) {
    if (!identical(rownames(batches[[i]]), reference_genes)) {
      stop("All batch matrices must contain the same genes in the same order")
    }
    if (length(batch.cluster.labels[[i]]) != ncol(batches[[i]])) {
      stop(sprintf("Cluster-label length does not match cell count in batch %d", i))
    }
  }

  # mnnpy expects cells x genes and converts inputs to float32.
  # IMPORTANT: remove list names before crossing the reticulate boundary.
  # A named R list is converted to a Python dict, and iterating that dict in
  # transform_input_data() yields batch-name strings instead of matrices.
  batches.t <- unname(lapply(batches, t))
  mnnpy <- import("mnnpy")
  # Keep NumPy results as Python ndarrays. With automatic conversion enabled,
  # np$array() is immediately converted back to an R matrix, which has no
  # Python setflags() method and can reintroduce read-only buffer problems.
  np <- import("numpy", convert = FALSE)

  cat("Data preparation and cosine normalization...\n")

  # Correction intentionally uses the full common gene set, so no var_subset
  # is needed. Use zero-based feature identifiers, matching the convention
  # used by the upstream SMNN wrapper.
  feature_index <- as.character(seq_len(ncol(batches.t[[1]])) - 1L)

  prep.out <- mnnpy$utils$transform_input_data(
    datas = batches.t,
    cos_norm_in = cos.norm.in,
    cos_norm_out = cos.norm.out,
    var_index = feature_index,
    var_subset = NULL,
    n_jobs = n.jobs
  )

  in.batches <- prep.out[[1]]
  out.batches <- prep.out[[2]]
  same.set <- isTRUE(prep.out[[4]])

  rm(batches.t)
  invisible(gc(verbose = FALSE))

  if (is.null(order)) {
    order <- seq_len(nbatches)
  } else {
    order <- as.integer(order)
    if (!identical(seq_len(nbatches), sort(order))) {
      stop(sprintf("'order' must contain each value in 1:%d exactly once", nbatches))
    }
  }

  ref <- order[[1]]
  ref.batch.in <- in.batches[[ref]]
  ref.cluster.labels <- batch.cluster.labels[[ref]]

  if (!same.set) {
    ref.batch.out <- out.batches[[ref]]
  }

  output <- vector("list", nbatches)
  output[[ref]] <- t(out.batches[[ref]])

  mnn.list <- vector("list", nbatches)
  mnn.list[[ref]] <- S4Vectors::DataFrame(
    current.cell = integer(0),
    other.cell = integer(0),
    other.batch = integer(0)
  )

  original.batch <- rep(ref, nrow(ref.batch.in))

  cat(sprintf(
    "Starting sequential SMNN; reference batch: %s (%d cells)\n",
    names(batches)[[ref]],
    nrow(ref.batch.in)
  ))

  for (step in 2:nbatches) {
    target <- order[[step]]
    other.batch.in <- in.batches[[target]]
    target.labels <- batch.cluster.labels[[target]]

    if (!same.set) {
      other.batch.out <- out.batches[[target]]
    }

    cat(sprintf(
      "Step %d/%d: target batch %s (%d cells), accumulated reference %d cells\n",
      step - 1L,
      nbatches - 1L,
      names(batches)[[target]],
      nrow(other.batch.in),
      nrow(ref.batch.in)
    ))

    s1 <- integer(0)
    s2 <- integer(0)

    # Determine shared cell types for the current accumulated-reference/target
    # pair instead of requiring a cell type to occur in every input batch.
    step.clusters <- intersect(
      unique(ref.cluster.labels),
      unique(target.labels)
    )
    step.clusters <- setdiff(step.clusters, 0L)
    if (!is.null(matched.clusters)) {
      step.clusters <- intersect(step.clusters, matched.clusters)
    }

    if (length(step.clusters) == 0L) {
      stop(sprintf(
        "No shared cell type was found for target batch %s",
        names(batches)[[target]]
      ))
    }

    cat(sprintf(
      "  Shared cell types at this step: %d\n",
      length(step.clusters)
    ))

    # Supervised MNN search within each cell type shared at this step.
    for (cluster_id in step.clusters) {
      ref_idx <- which(ref.cluster.labels == cluster_id)
      target_idx <- which(target.labels == cluster_id)

      # Use the official k whenever possible, but do not let one rare cell
      # type reduce k for every other type.
      cluster.k <- min(k, length(ref_idx), length(target_idx))
      if (cluster.k < 2L) {
        cat(sprintf(
          "  Skipping cell type %s: fewer than 2 cells on one side\n",
          as.character(cluster_id)
        ))
        next
      }

      cat(sprintf(
        paste0(
          "  Finding MNNs for cell type %s: reference=%d, ",
          "target=%d, k=%d\n"
        ),
        as.character(cluster_id),
        length(ref_idx),
        length(target_idx),
        cluster.k
      ))

      pair_result <- mnnpy$utils$find_mutual_nn(
        data1 = ref.batch.in[ref_idx, , drop = FALSE],
        data2 = other.batch.in[target_idx, , drop = FALSE],
        k1 = cluster.k,
        k2 = cluster.k,
        n_jobs = n.jobs
      )

      local_ref <- as.integer(unlist(pair_result[[1]], use.names = FALSE))
      local_target <- as.integer(unlist(pair_result[[2]], use.names = FALSE))

      if (length(local_ref) != length(local_target)) {
        stop("mnnpy returned inconsistent MNN index vectors")
      }

      if (length(local_ref) > 0L) {
        # Convert Python's zero-based subset indices to R's one-based full indices.
        s1 <- c(s1, ref_idx[local_ref + 1L])
        s2 <- c(s2, target_idx[local_target + 1L])
      }
    }

    if (isTRUE(correct.others)) {
      ref_idx <- which(ref.cluster.labels == 0L)
      target_idx <- which(target.labels == 0L)

      if (length(ref_idx) >= k && length(target_idx) >= k) {
        pair_result <- mnnpy$utils$find_mutual_nn(
          data1 = ref.batch.in[ref_idx, , drop = FALSE],
          data2 = other.batch.in[target_idx, , drop = FALSE],
          k1 = k,
          k2 = k,
          n_jobs = n.jobs
        )

        local_ref <- as.integer(unlist(pair_result[[1]], use.names = FALSE))
        local_target <- as.integer(unlist(pair_result[[2]], use.names = FALSE))

        if (length(local_ref) > 0L) {
          s1 <- c(s1, ref_idx[local_ref + 1L])
          s2 <- c(s2, target_idx[local_target + 1L])
        }
      }
    }

    if (length(s1) == 0L) {
      stop(sprintf("No supervised MNN pairs were found for target batch %s", names(batches)[[target]]))
    }

    cat(sprintf("  Computing correction vectors from %d MNN pairs...\n", length(s1)))

    correction.in <- smnn_compute_correction(
      ref.batch.in,
      other.batch.in,
      s1,
      s2,
      other.batch.in,
      sigma
    )

    if (!same.set) {
      correction.out <- smnn_compute_correction(
        ref.batch.out,
        other.batch.out,
        s1,
        s2,
        other.batch.in,
        sigma
      )
    }

    if (isTRUE(var.adj)) {
      cat(sprintf("  Adjusting shift variance with %d workers...\n", n.jobs))

      # Arrays returned through reticulate/mnnpy can be non-contiguous or
      # read-only views. Numba's typed kernels require writable, C-contiguous
      # float32 buffers, especially when multiprocessing serializes them.
      ref.var <- np$array(
        ref.batch.in,
        dtype = np$float32,
        order = "C",
        copy = TRUE
      )
      other.var <- np$array(
        other.batch.in,
        dtype = np$float32,
        order = "C",
        copy = TRUE
      )
      correction.var <- np$array(
        correction.in,
        dtype = np$float32,
        order = "C",
        copy = TRUE
      )
      ref.var$setflags(write = TRUE)
      other.var$setflags(write = TRUE)
      correction.var$setflags(write = TRUE)

      correction.in <- mnnpy$utils$adjust_shift_variance(
        ref.var,
        other.var,
        correction.var,
        as.numeric(sigma),
        as.integer(n.jobs)
      )

      if (!same.set) {
        ref.out.var <- np$array(
          ref.batch.out,
          dtype = np$float32,
          order = "C",
          copy = TRUE
        )
        other.out.var <- np$array(
          other.batch.out,
          dtype = np$float32,
          order = "C",
          copy = TRUE
        )
        correction.out.var <- np$array(
          correction.out,
          dtype = np$float32,
          order = "C",
          copy = TRUE
        )
        ref.out.var$setflags(write = TRUE)
        other.out.var$setflags(write = TRUE)
        correction.out.var$setflags(write = TRUE)

        correction.out <- mnnpy$utils$adjust_shift_variance(
          ref.out.var,
          other.out.var,
          correction.out.var,
          as.numeric(sigma),
          as.integer(n.jobs),
          prep.out[[3]]
        )
      }

      rm(ref.var, other.var, correction.var)
      if (exists("ref.out.var", inherits = FALSE)) rm(ref.out.var)
      if (exists("other.out.var", inherits = FALSE)) rm(other.out.var)
      if (exists("correction.out.var", inherits = FALSE)) rm(correction.out.var)
    }

    other.batch.in <- other.batch.in + correction.in
    ref.batch.in <- rbind(ref.batch.in, other.batch.in)

    # Critical multi-batch fix: labels must grow with the accumulated reference.
    ref.cluster.labels <- c(ref.cluster.labels, target.labels)

    if (same.set) {
      output[[target]] <- t(other.batch.in)
    } else {
      other.batch.out <- other.batch.out + correction.out
      ref.batch.out <- rbind(ref.batch.out, other.batch.out)
      output[[target]] <- t(other.batch.out)
    }

    mnn.list[[target]] <- S4Vectors::DataFrame(
      current.cell = s2,
      other.cell = S4Vectors::Rle(s1),
      other.batch = S4Vectors::Rle(original.batch[s1])
    )

    original.batch <- c(original.batch, rep(target, nrow(other.batch.in)))

    rm(pair_result, correction.in, other.batch.in)
    if (exists("correction.out", inherits = FALSE)) rm(correction.out)
    if (exists("other.batch.out", inherits = FALSE)) rm(other.batch.out)
    invisible(gc(verbose = FALSE))
  }

  names(output) <- names(batches)
  names(mnn.list) <- names(batches)

  list(corrected = output, pairs = mnn.list)
}

# ============================================================
# 2. Parse arguments
# ============================================================

parser <- ArgumentParser(description = "Run supervised SMNN batch correction")

parser$add_argument(
  "--input_dir",
  type = "character",
  required = TRUE,
  help = paste(
    "Directory containing matrix.mtx, barcodes.csv,",
    "features.csv, metadata.csv"
  )
)

parser$add_argument(
  "--batch_key",
  type = "character",
  default = "batch"
)

parser$add_argument(
  "--celltype_key",
  type = "character",
  required = TRUE,
  help = "Cell-type annotation column used by supervised SMNN"
)

parser$add_argument(
  "--seed",
  type = "integer",
  default = 42
)

args <- parser$parse_args()
set.seed(args$seed)

# ============================================================
# 3. Load data
# ============================================================

cat("Loading data from mtx + csv...\n")

counts <- readMM(file.path(args$input_dir, "matrix.mtx"))
counts <- as(counts, "CsparseMatrix")

barcodes <- read.csv(
  file.path(args$input_dir, "barcodes.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

features <- read.csv(
  file.path(args$input_dir, "features.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

metadata <- read.csv(
  file.path(args$input_dir, "metadata.csv"),
  row.names = 1,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

rownames(counts) <- features$feature
colnames(counts) <- barcodes$barcode

missing_cells <- setdiff(colnames(counts), rownames(metadata))
if (length(missing_cells) > 0L) {
  stop(sprintf("metadata.csv is missing %d cells from matrix.mtx", length(missing_cells)))
}
metadata <- metadata[colnames(counts), , drop = FALSE]

if (!(args$batch_key %in% colnames(metadata))) {
  stop(sprintf("batch_key '%s' is not present in metadata.csv", args$batch_key))
}
if (!(args$celltype_key %in% colnames(metadata))) {
  stop(sprintf("celltype_key '%s' is not present in metadata.csv", args$celltype_key))
}

batch_values <- as.character(metadata[[args$batch_key]])
label_values <- as.character(metadata[[args$celltype_key]])

if (anyNA(batch_values) || any(batch_values == "")) {
  stop("batch_key contains missing or empty values")
}
if (anyNA(label_values) || any(label_values == "")) {
  stop("celltype_key contains missing or empty values")
}

batch_levels <- unique(batch_values)
if (length(batch_levels) < 2L) {
  stop("SMNN requires at least two batches")
}

if (!py_module_available("mnnpy")) {
  stop(paste0(
    "Python module 'mnnpy' is unavailable. RETICULATE_PYTHON=",
    Sys.getenv("RETICULATE_PYTHON")
  ))
}

cat(sprintf(
  "Dataset: %d cells, %d genes, %d batches\n",
  ncol(counts),
  nrow(counts),
  length(batch_levels)
))

# ============================================================
# 4. Full-gene SMNN correction and HVG PCA (timed)
# ============================================================

timer_start <- proc.time()

seurat_obj <- CreateSeuratObject(counts = counts, meta.data = metadata)
seurat_list <- SplitObject(seurat_obj, split.by = args$batch_key)
seurat_list <- lapply(seurat_list, function(x) {
  x <- NormalizeData(x, verbose = FALSE)
  FindVariableFeatures(
    x,
    selection.method = "vst",
    nfeatures = min(2000L, nrow(x)),
    verbose = FALSE
  )
})

integration_features <- SelectIntegrationFeatures(
  object.list = seurat_list,
  nfeatures = min(2000L, nrow(counts)),
  verbose = FALSE
)
integration_features <- unique(intersect(integration_features, rownames(counts)))

if (length(integration_features) < SMNN_NPCS) {
  stop(sprintf(
    "Only %d integration features were selected; at least %d are required",
    length(integration_features),
    SMNN_NPCS
  ))
}

cat(sprintf(
  "Running full-gene SMNN on %d genes; final PCA uses %d features\n",
  nrow(counts),
  length(integration_features)
))

rm(seurat_obj, seurat_list)
invisible(gc(verbose = FALSE))

label_levels <- sort(unique(label_values))
label_codes <- match(label_values, label_levels)

batch_indices <- lapply(batch_levels, function(batch_name) {
  which(batch_values == batch_name)
})

batches <- lapply(batch_indices, function(idx) {
  batch_matrix <- as.matrix(counts[, idx, drop = FALSE])
  storage.mode(batch_matrix) <- "double"
  batch_matrix
})
names(batches) <- batch_levels

batch_cluster_labels <- lapply(batch_indices, function(idx) {
  label_codes[idx]
})
names(batch_cluster_labels) <- batch_levels

# Preserve the official default order: the first input batch is the reference.
batch_sizes <- vapply(batches, ncol, integer(1))
batch_order <- seq_along(batches)

cat(sprintf(
  "Reference batch (original input order): %s (%d cells)\n",
  names(batches)[[batch_order[[1]]]],
  batch_sizes[[batch_order[[1]]]]
))

# Allow every annotated type to participate; SMNNcorrect_patched determines
# which types are shared separately at each sequential integration step.
matched_clusters <- sort(unique(label_codes))
effective_k <- SMNN_K

cat(sprintf(
  paste0(
    "Available cell types: %d; maximum SMNN k: %d; ",
    "variance adjustment: TRUE; workers: %d\n"
  ),
  length(matched_clusters),
  effective_k,
  N_WORKERS
))

batch_cell_names <- lapply(batches, colnames)

smnn_result <- SMNNcorrect_patched(
  batches = batches,
  batch.cluster.labels = batch_cluster_labels,
  matched.clusters = matched_clusters,
  correct.others = FALSE,
  k = effective_k,
  sigma = SMNN_SIGMA,
  cos.norm.in = TRUE,
  cos.norm.out = TRUE,
  var.adj = TRUE,
  order = batch_order,
  n.jobs = N_WORKERS
)

corrected_list <- smnn_result$corrected

for (i in seq_along(corrected_list)) {
  corrected_list[[i]] <- as.matrix(corrected_list[[i]])
  rownames(corrected_list[[i]]) <- rownames(counts)
  colnames(corrected_list[[i]]) <- batch_cell_names[[i]]
}

rm(batches, batch_cluster_labels, smnn_result)
invisible(gc(verbose = FALSE))

corrected <- do.call(cbind, corrected_list)
corrected <- corrected[, colnames(counts), drop = FALSE]

if (any(!is.finite(corrected))) {
  stop("SMNN returned non-finite corrected values")
}

smnn_obj <- CreateSeuratObject(counts = counts, meta.data = metadata)
smnn_obj[["SMNN"]] <- CreateAssay5Object(data = corrected)
DefaultAssay(smnn_obj) <- "SMNN"
VariableFeatures(smnn_obj) <- integration_features

smnn_obj <- ScaleData(
  smnn_obj,
  features = integration_features,
  verbose = FALSE
)

smnn_obj <- RunPCA(
  smnn_obj,
  features = integration_features,
  npcs = SMNN_NPCS,
  verbose = FALSE,
  seed.use = args$seed
)

embeddings <- Embeddings(smnn_obj, "pca")

if (ncol(embeddings) != SMNN_NPCS) {
  stop(sprintf("Expected %d PCs, received %d", SMNN_NPCS, ncol(embeddings)))
}

algo_seconds <- unname((proc.time() - timer_start)[["elapsed"]])
cat(sprintf("Algorithm runtime: %.4f seconds\n", algo_seconds))

write.csv(
  data.frame(runtime_seconds = algo_seconds),
  file.path(args$input_dir, "timing.csv"),
  row.names = FALSE
)

# ============================================================
# 5. Export embedding
# ============================================================

write.csv(
  embeddings,
  file.path(args$input_dir, "embedding.csv")
)

cat("SMNN integration complete.\n")
