"""scGPT fine-tuning cell type annotation benchmark."""

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
    preprocessor = model_impl.Preprocessor(
        use_key="X",
        normalize_total=1e4 if raw_counts else False,
        log1p=raw_counts,
        binning=model_impl.DEFAULT_N_BINS,
    )
    preprocessor(adata, batch_key=None)
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
    model_dir = run_out_path / "_model"
    return model_impl._finetune(
        adata,
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

    processed = _preprocess_adata(adata.copy(), vocab)
    genes = processed.var["gene_name"].tolist()
    gene_ids = np.asarray(vocab(genes), dtype=int)
    tokenized = model_impl._tokenize(processed, gene_ids, vocab, pad_value)
    test_data = {
        "gene_ids": tokenized["genes"],
        "values": model_impl.random_mask_value(
            tokenized["values"], mask_ratio=0.0, pad_value=pad_value
        ),
        "target_values": tokenized["values"],
        "celltype_labels": torch.zeros(processed.n_obs, dtype=torch.long),
    }
    loader = model_impl._prepare_dataloader(
        test_data, args.batch_size, shuffle=False
    )
    model.to(device)
    probabilities = _predict_probabilities(model, loader, vocab, device)
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
