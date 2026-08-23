"""scGPT fine-tuning cell type annotation benchmark."""

import gc
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import model_impl
from utils.dataset import is_count_matrix
from utils.pipeline import (
    SCORE_CLASSES_KEY,
    SCORE_KEY,
    create_argument_parser,
    run_annotation_pipeline,
)

# Benchmark memory safeguards, not upstream scGPT defaults. They are fixed for
# every dataset so no method receives dataset-specific tuning.
MAX_TRAIN_CELLS = 20_000
PREDICTION_CHUNK_SIZE = 2_048


def _stratified_training_subset(adata, label_col, seed, max_cells=MAX_TRAIN_CELLS):
    """Bound scGPT fine-tuning memory while preserving every cell type."""
    if adata.n_obs <= max_cells:
        return adata

    labels = adata.obs[label_col].astype(str).to_numpy()
    classes = np.unique(labels)
    if len(classes) > max_cells:
        raise ValueError("More cell types than the scGPT training-cell limit")

    rng = np.random.default_rng(seed)
    quota = max(1, max_cells // len(classes))
    selected = []
    for cell_type in classes:
        indices = np.flatnonzero(labels == cell_type)
        if len(indices) > quota:
            indices = rng.choice(indices, size=quota, replace=False)
        selected.extend(indices.tolist())

    slots = max_cells - len(selected)
    if slots > 0:
        chosen = np.zeros(adata.n_obs, dtype=bool)
        chosen[np.asarray(selected, dtype=int)] = True
        remaining = np.flatnonzero(~chosen)
        if len(remaining) > slots:
            remaining = rng.choice(remaining, size=slots, replace=False)
        selected.extend(remaining.tolist())

    selected = np.sort(np.asarray(selected, dtype=int))
    print(
        f"[scGPT] stratified training sampling: "
        f"{adata.n_obs} -> {len(selected)} cells"
    )
    return adata[selected].copy()


def _preprocess_adata(adata, vocab):
    """Apply scGPT normalization/binning while respecting processed X."""
    adata.var["gene_name"] = adata.var_names.astype(str)
    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()
    if adata.n_vars == 0:
        raise ValueError("No query genes occur in the scGPT vocabulary")

    raw_counts = is_count_matrix(adata.X)
    # Keep the upstream normalization/log1p behavior, then bin with the
    # mathematically equivalent sparse implementation. Upstream scGPT first
    # densifies the complete matrix, which is unnecessarily slow and memory
    # intensive for this benchmark.
    preprocessor = model_impl.Preprocessor(
        normalize_total=1e4 if raw_counts else False,
        result_normed_key=None,
        log1p=raw_counts,
        result_log1p_key=None,
    )
    preprocessor(adata, batch_key=None)
    print(
        f"[scGPT] sparse-safe binning: {adata.n_obs} cells x {adata.n_vars} genes",
        flush=True,
    )
    binned, bin_edges = model_impl.sparse_safe_binning(
        adata.X, model_impl.DEFAULT_N_BINS
    )
    adata.layers["X_binned"] = binned
    adata.obsm["bin_edges"] = bin_edges
    return adata


def _predict_probabilities(model, loader, vocab, device):
    model.eval()
    probabilities = []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch_data in loader:
            gene_ids = batch_data["gene_ids"].to(device)
            values = batch_data["values"].to(device)
            padding_mask = gene_ids.eq(vocab[model_impl.PAD_TOKEN])
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(
                    gene_ids,
                    values,
                    src_key_padding_mask=padding_mask,
                    CLS=True,
                )
            probabilities.append(
                torch.softmax(output["cls_output"], dim=1).cpu().numpy()
            )
    return np.concatenate(probabilities, axis=0)


def train_scgpt(adata, seed, run_out_path, args):
    """Fine-tune the pretrained whole-human scGPT classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_impl._preprocess_adata = _preprocess_adata
    training = _stratified_training_subset(
        adata, args.train_celltype, seed
    )
    model_dir = run_out_path / "_model"
    return model_impl._finetune(
        training,
        args.train_celltype,
        args.scgpt_model_dir,
        model_impl.ensure_dir(model_dir),
        seed,
        args,
        device,
    )


def predict_scgpt(adata, model_handle, pred_key, seed, run_out_path, args):
    """Predict query labels and retain softmax probabilities for AUC."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_handle["model"]
    vocab = model_handle["vocab"]
    id_to_type = model_handle["id2type"]
    pad_value = model_handle["pad_value"]

    if adata.n_obs == 0:
        raise ValueError("Query dataset contains no cells")
    if adata.n_obs > PREDICTION_CHUNK_SIZE:
        n_chunks = (adata.n_obs + PREDICTION_CHUNK_SIZE - 1) // PREDICTION_CHUNK_SIZE
        print(
            f"[scGPT] predicting {adata.n_obs} cells in "
            f"{n_chunks} chunks"
        )

    model.to(device)
    probability_parts = []
    for start in range(0, adata.n_obs, PREDICTION_CHUNK_SIZE):
        stop = min(start + PREDICTION_CHUNK_SIZE, adata.n_obs)
        processed = _preprocess_adata(adata[start:stop].copy(), vocab)
        genes = processed.var["gene_name"].tolist()
        gene_ids = np.asarray(vocab(genes), dtype=int)
        tokenized = model_impl._tokenize(
            processed, gene_ids, vocab, pad_value
        )
        test_data = {
            "gene_ids": tokenized["genes"],
            "values": tokenized["values"],
            "celltype_labels": torch.zeros(
                processed.n_obs, dtype=torch.long
            ),
        }
        loader = model_impl._prepare_dataloader(
            test_data, args.batch_size, shuffle=False
        )
        probability_parts.append(
            _predict_probabilities(model, loader, vocab, device)
        )
        del processed, tokenized, test_data, loader
        gc.collect()

    probabilities = np.concatenate(probability_parts, axis=0)
    class_names = np.asarray(
        [str(id_to_type[index]) for index in range(len(id_to_type))], dtype=str
    )

    adata.obs[pred_key] = class_names[probabilities.argmax(axis=1)]
    adata.obsm[SCORE_KEY] = probabilities
    adata.uns[SCORE_CLASSES_KEY] = class_names
    adata.uns["annotation_score_type"] = "probabilities"
    return adata


def find_pretrained_model():
    configured = os.environ.get("SCGPT_MODEL_DIR", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path(__file__).parent / "models" / "scGPT_human",
        Path(__file__).parent / "models" / "whole_human",
    ]
    for candidate in candidates:
        if candidate is not None and all(
            (candidate / filename).exists()
            for filename in ["best_model.pt", "vocab.json", "args.json"]
        ):
            return candidate
    return None


if __name__ == "__main__":
    parser = create_argument_parser("scGPT")
    parser.add_argument("--scgpt_model_dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=model_impl.DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=model_impl.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=model_impl.DEFAULT_LR)
    parser.add_argument(
        "--schedule_ratio", type=float, default=model_impl.DEFAULT_SCHEDULE_RATIO
    )
    arguments = parser.parse_args()
    if arguments.scgpt_model_dir is None:
        arguments.scgpt_model_dir = find_pretrained_model()
    if arguments.scgpt_model_dir is None:
        raise FileNotFoundError(
            "scGPT checkpoint not found. Run scGPT/download_checkpoint.py first."
        )

    run_annotation_pipeline(
        "scGPT", "scgpt_prediction", train_scgpt, predict_scgpt, arguments
    )
