"""scGPT cell type annotation benchmark.

Fine-tune the whole-human pretrained scGPT model with a CLS (cell-type
classification) objective, following the official tutorial:
  https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html
"""

import copy
import gc
import json
import os
import shutil
import sys
import time
import types
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# torchtext compatibility shim
# ============================================================
# torchtext 0.18 (the last release) ships a C extension compiled against
# torch ≤2.3.  When a newer torch is installed the shared library fails to
# load (undefined symbol).  scGPT only uses torchtext for the Vocab class,
# so we provide a lightweight pure-Python replacement when the native
# extension is broken.

def _install_torchtext_shim() -> None:
    """Replace torchtext's C-backed Vocab with a pure-Python version."""
    try:                       # fast path: torchtext loads normally
        import torchtext       # noqa: F401
        return
    except (OSError, ImportError):
        pass
    # Remove partially-loaded torchtext submodules
    for key in [k for k in sys.modules if k == "torchtext" or k.startswith("torchtext.")]:
        del sys.modules[key]

    import torch.nn as _nn

    # --- Pure-Python VocabPybind replacement ---
    class _VocabPybind:
        def __init__(self, tokens, default_index=None):
            self.itos_ = list(tokens) if tokens else []
            self.stoi_ = {t: i for i, t in enumerate(self.itos_)}
            self.default_index_ = default_index

        def __len__(self):
            return len(self.itos_)

        def __contains__(self, token):
            return token in self.stoi_

        def __getitem__(self, token):
            if token in self.stoi_:
                return self.stoi_[token]
            if self.default_index_ is not None:
                return self.default_index_
            raise RuntimeError(f"Token '{token}' not found and default index is not set")

        def set_default_index(self, index):
            self.default_index_ = index

        def get_default_index(self):
            return self.default_index_

        def insert_token(self, token, index):
            if token in self.stoi_:
                raise RuntimeError(f"Token '{token}' already exists")
            if index < 0 or index > len(self.itos_):
                raise RuntimeError(f"Index {index} not in [0, {len(self.itos_)}]")
            self.itos_.insert(index, token)
            self.stoi_ = {t: i for i, t in enumerate(self.itos_)}

        def append_token(self, token):
            if token in self.stoi_:
                raise RuntimeError(f"Token '{token}' already exists")
            self.stoi_[token] = len(self.itos_)
            self.itos_.append(token)

        def lookup_token(self, index):
            return self.itos_[index]

        def lookup_tokens(self, indices):
            return [self.itos_[i] for i in indices]

        def lookup_indices(self, tokens):
            return [self[t] for t in tokens]

        def get_stoi(self):
            return dict(self.stoi_)

        def get_itos(self):
            return list(self.itos_)

    # --- Pure-Python Vocab (mirrors torchtext.vocab.Vocab) ---
    class _Vocab(_nn.Module):
        def __init__(self, vocab):
            super().__init__()
            self.vocab = vocab

        def __len__(self):
            return len(self.vocab)

        def __contains__(self, token):
            return token in self.vocab

        def __getitem__(self, token):
            return self.vocab[token]

        def forward(self, tokens):
            return self.vocab.lookup_indices(tokens)

        def set_default_index(self, index):
            self.vocab.set_default_index(index)

        def get_default_index(self):
            return self.vocab.get_default_index()

        def insert_token(self, token, index):
            self.vocab.insert_token(token, index)

        def append_token(self, token):
            self.vocab.append_token(token)

        def lookup_token(self, index):
            return self.vocab.lookup_token(index)

        def lookup_tokens(self, indices):
            return self.vocab.lookup_tokens(indices)

        def lookup_indices(self, tokens):
            return self.vocab.lookup_indices(tokens)

        def get_stoi(self):
            return self.vocab.get_stoi()

        def get_itos(self):
            return self.vocab.get_itos()

    def _vocab_factory(ordered_dict, min_freq=1, specials=None, special_first=True):
        specials = specials or []
        for tok in specials:
            ordered_dict.pop(tok, None)
        tokens = [tok for tok, freq in ordered_dict.items() if freq >= min_freq]
        if special_first:
            tokens[0:0] = specials
        else:
            tokens.extend(specials)
        return _Vocab(_VocabPybind(tokens, None))

    def _build_vocab_from_iterator(iterator, min_freq=1, specials=None,
                                    special_first=True, max_tokens=None):
        counter = Counter()
        for tokens in iterator:
            counter.update(tokens)
        specials = specials or []
        sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        if max_tokens is not None:
            sorted_items = sorted_items[: max_tokens - len(specials)]
        od = OrderedDict(sorted_items)
        return _vocab_factory(od, min_freq=min_freq, specials=specials,
                              special_first=special_first)

    def _noop_log(*_a, **_kw):
        pass

    # --- Assemble fake torchtext modules ---
    tt = types.ModuleType("torchtext")
    tt._WARN = False
    tt._TORCHTEXT_DEPRECATION_MSG = ""
    tt.disable_torchtext_deprecation_warning = lambda: None
    sys.modules["torchtext"] = tt

    ext = types.ModuleType("torchtext._extension")
    sys.modules["torchtext._extension"] = ext
    tt._extension = ext

    mu = types.ModuleType("torchtext._internal.module_utils")
    mu.is_module_available = lambda *a, **kw: True
    _internal = types.ModuleType("torchtext._internal")
    _internal.module_utils = mu
    sys.modules["torchtext._internal"] = _internal
    sys.modules["torchtext._internal.module_utils"] = mu
    tt._internal = _internal

    tt_tt = types.ModuleType("torchtext._torchtext")
    tt_tt.Vocab = _VocabPybind
    sys.modules["torchtext._torchtext"] = tt_tt
    tt._torchtext = tt_tt

    utils_mod = types.ModuleType("torchtext.utils")
    utils_mod._log_class_usage = _noop_log
    sys.modules["torchtext.utils"] = utils_mod
    tt.utils = utils_mod

    vocab_mod = types.ModuleType("torchtext.vocab")
    vocab_mod.Vocab = _Vocab
    vocab_mod.vocab = _vocab_factory
    vocab_mod.build_vocab_from_iterator = _build_vocab_from_iterator
    sys.modules["torchtext.vocab"] = vocab_mod
    sys.modules["torchtext.vocab.vocab"] = types.ModuleType("torchtext.vocab.vocab")
    sys.modules["torchtext.vocab.vocab"].Vocab = _Vocab
    sys.modules["torchtext.vocab.vocab_factory"] = types.ModuleType("torchtext.vocab.vocab_factory")
    sys.modules["torchtext.vocab.vocab_factory"].vocab = _vocab_factory
    sys.modules["torchtext.vocab.vocab_factory"].build_vocab_from_iterator = _build_vocab_from_iterator
    tt.vocab = vocab_mod

_install_torchtext_shim()

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader

from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed

from utils.label_utils import get_broad_type
from utils.scgpt_compat import (
    create_argument_parser, run_annotation_pipeline,
    save_subtype_results, load_adata, ensure_dir,
    resolve_label_col, seed_python_numpy,
    BROAD_TYPES_FOR_SUBTYPE,
)

warnings.filterwarnings("ignore")

# ============================================================
# Default Hyper-parameters
# ============================================================

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-4
DEFAULT_SCHEDULE_RATIO = 0.9
DEFAULT_MAX_SEQ_LEN = 3001
DEFAULT_N_BINS = 51
DEFAULT_DROPOUT = 0.2

PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]


# ============================================================
# Dataset
# ============================================================

class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# ============================================================
# Preprocessing
# ============================================================

def _preprocess_adata(adata, vocab):
    """Filter genes to vocab, run Preprocessor (normalize + bin)."""
    adata.var["gene_name"] = adata.var.index.tolist()
    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()

    preprocessor = Preprocessor(
        use_key="X",
        normalize_total=False,
        binning=DEFAULT_N_BINS,
    )
    preprocessor(adata, batch_key=None)
    return adata


def _tokenize(adata, gene_ids, vocab, pad_value):
    """Extract binned counts and tokenize."""
    layer = adata.layers["X_binned"]
    counts = layer.A if issparse(layer) else layer
    return tokenize_and_pad_batch(
        counts, gene_ids,
        max_len=DEFAULT_MAX_SEQ_LEN,
        vocab=vocab,
        pad_token=PAD_TOKEN,
        pad_value=pad_value,
    )


def _prepare_dataloader(data_pt, batch_size, shuffle=False):
    num_workers = min(len(os.sched_getaffinity(0)), max(1, batch_size // 2))
    return DataLoader(
        dataset=SeqDataset(data_pt),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# ============================================================
# Model Construction & Loading
# ============================================================

def _load_pretrained(model_dir, num_types, device):
    """Build TransformerModel and load pretrained weights from model_dir."""
    model_dir = Path(model_dir)
    vocab_file = model_dir / "vocab.json"
    config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"

    vocab = GeneVocab.from_file(vocab_file)
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)

    with open(config_file, "r") as f:
        model_configs = json.load(f)

    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs.get("n_layers_cls", 3)

    pad_value = -2

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=embsize,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=num_types,
        vocab=vocab,
        dropout=DEFAULT_DROPOUT,
        pad_value=pad_value,
        ecs_threshold=0.0,
        use_fast_transformer=True,
    )

    pretrained_dict = torch.load(model_file, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)

    return model, vocab, pad_value


# ============================================================
# Training Loop
# ============================================================

def _train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion_cls,
                     vocab, pad_value, device, use_amp):
    model.train()
    total_loss, total_error, total_num = 0.0, 0.0, 0

    for batch_data in loader:
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[PAD_TOKEN])

        with torch.cuda.amp.autocast(enabled=use_amp):
            output_dict = model(
                input_gene_ids, input_values,
                src_key_padding_mask=src_key_padding_mask,
                CLS=True,
            )
            loss = criterion_cls(output_dict["cls_output"], celltype_labels)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                       error_if_nonfinite=False if scaler.is_enabled() else True)
        scaler.step(optimizer)
        scaler.update()

        bs = input_gene_ids.size(0)
        total_loss += loss.item() * bs
        total_error += (1 - (output_dict["cls_output"].argmax(1) == celltype_labels
                             ).sum().item() / bs) * bs
        total_num += bs

    scheduler.step()
    return total_loss / total_num, total_error / total_num


def _evaluate(model, loader, criterion_cls, vocab, device, use_amp):
    model.eval()
    total_loss, total_error, total_num = 0.0, 0.0, 0

    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[PAD_TOKEN])

            with torch.cuda.amp.autocast(enabled=use_amp):
                output_dict = model(
                    input_gene_ids, input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=True,
                )
                loss = criterion_cls(output_dict["cls_output"], celltype_labels)

            bs = input_gene_ids.size(0)
            total_loss += loss.item() * bs
            total_error += (1 - (output_dict["cls_output"].argmax(1) == celltype_labels
                                 ).sum().item() / bs) * bs
            total_num += bs

    return total_loss / total_num, total_error / total_num


def _predict_labels(model, loader, vocab, device, use_amp):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[PAD_TOKEN])

            with torch.cuda.amp.autocast(enabled=use_amp):
                output_dict = model(
                    input_gene_ids, input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=True,
                )
            all_preds.append(output_dict["cls_output"].argmax(1).cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def _finetune(adata, label_col, pretrained_dir, save_dir, seed, args, device):
    """Full fine-tuning pipeline: preprocess, tokenize, train, return handle."""
    set_seed(seed)
    ensure_dir(save_dir)

    # Build label mapping
    celltypes = adata.obs[label_col].astype("category")
    id2type = dict(enumerate(celltypes.cat.categories))
    num_types = len(id2type)
    celltype_ids = celltypes.cat.codes.values.astype(int)
    adata.obs["celltype_id"] = celltype_ids

    # Load pretrained vocab & model
    model, vocab, pad_value = _load_pretrained(
        pretrained_dir, num_types, device,
    )

    # Copy vocab to save_dir
    shutil.copy(Path(pretrained_dir) / "vocab.json", save_dir / "vocab.json")

    # Preprocess
    adata = _preprocess_adata(adata, vocab)
    genes = adata.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    # Train/valid split
    layer = adata.layers["X_binned"]
    all_counts = layer.A if issparse(layer) else layer
    (train_counts, valid_counts,
     train_ct, valid_ct) = train_test_split(
        all_counts, celltype_ids,
        test_size=0.1, shuffle=True, random_state=seed,
    )

    # Tokenize
    tokenized_train = tokenize_and_pad_batch(
        train_counts, gene_ids,
        max_len=DEFAULT_MAX_SEQ_LEN, vocab=vocab,
        pad_token=PAD_TOKEN, pad_value=pad_value,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_counts, gene_ids,
        max_len=DEFAULT_MAX_SEQ_LEN, vocab=vocab,
        pad_token=PAD_TOKEN, pad_value=pad_value,
    )

    # Build data tensors
    def _make_pt(tok, ct):
        return {
            "gene_ids": tok["genes"],
            "values": random_mask_value(
                tok["values"], mask_ratio=0.0,
                pad_value=pad_value,
            ),
            "target_values": tok["values"],
            "celltype_labels": torch.from_numpy(ct).long(),
        }

    train_pt = _make_pt(tokenized_train, train_ct)
    valid_pt = _make_pt(tokenized_valid, valid_ct)

    batch_size = args.batch_size
    train_loader = _prepare_dataloader(train_pt, batch_size, shuffle=True)
    valid_loader = _prepare_dataloader(valid_pt, batch_size, shuffle=False)

    # Optimizer / scheduler
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 1, gamma=args.schedule_ratio,
    )
    use_amp = True
    scaler = torch.cuda.amp.GradScaler()

    # Training
    best_val_loss = float("inf")
    best_model = None
    patience = 3
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_err = _train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion_cls, vocab, pad_value, device, use_amp,
        )
        val_loss, val_err = _evaluate(
            model, valid_loader, criterion_cls, vocab, device, use_amp,
        )
        elapsed = time.time() - t0
        print(f"  epoch {epoch:3d} | {elapsed:5.1f}s | "
              f"train loss {train_loss:.4f} err {train_err:.4f} | "
              f"valid loss {val_loss:.4f} err {val_err:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Save
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    with open(save_dir / "label_map.json", "w") as f:
        json.dump({str(k): v for k, v in id2type.items()}, f)
    with open(save_dir / "gene_list.json", "w") as f:
        json.dump(genes, f)

    del model, train_loader, valid_loader, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": best_model,
        "vocab": vocab,
        "gene_ids": gene_ids,
        "id2type": id2type,
        "num_types": num_types,
        "pad_value": pad_value,
        "pretrained_dir": str(pretrained_dir),
        "save_dir": str(save_dir),
    }


# ============================================================
# Train / Predict  (pipeline interface)
# ============================================================

def train_scgpt(train_path, model_dir, label_col, seed, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = load_adata(train_path)
    resolved = resolve_label_col(adata, label_col)
    print(f"[train] {train_path} | label={resolved} | seed={seed}")

    handle = _finetune(
        adata, resolved, args.scgpt_model_dir,
        ensure_dir(model_dir), seed, args, device,
    )
    return handle


def predict_scgpt(test_path, model_handle, pred_col, seed, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = load_adata(test_path)

    model = model_handle["model"]
    vocab = model_handle["vocab"]
    id2type = model_handle["id2type"]
    pad_value = model_handle["pad_value"]

    # Preprocess test data with same vocab
    adata_proc = _preprocess_adata(adata.copy(), vocab)
    genes = adata_proc.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    tokenized = _tokenize(adata_proc, gene_ids, vocab, pad_value)
    test_pt = {
        "gene_ids": tokenized["genes"],
        "values": random_mask_value(
            tokenized["values"], mask_ratio=0.0,
            pad_value=pad_value,
        ),
        "target_values": tokenized["values"],
        "celltype_labels": torch.zeros(adata_proc.n_obs, dtype=torch.long),
    }

    loader = _prepare_dataloader(test_pt, args.batch_size, shuffle=False)
    model.to(device)
    use_amp = True
    preds_ids = _predict_labels(model, loader, vocab, device, use_amp)
    preds = pd.Series([id2type[int(p)] for p in preds_ids], index=adata.obs_names)

    out = adata.copy()
    out.obs[pred_col] = preds.values
    return out


# ============================================================
# Subtype (fine-tune per broad type)
# ============================================================

def train_subtype_scgpt(subtype_train_path, model_dir, seed, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = load_adata(subtype_train_path)
    subtype_col = "Celltype_subtraining"
    broad_col = ("Celltype_training"
                 if "Celltype_training" in adata.obs.columns else None)
    if broad_col is None:
        adata.obs["_derived_broad"] = (
            adata.obs[subtype_col].astype(str).map(get_broad_type)
        )
        broad_col = "_derived_broad"

    ensure_dir(model_dir)

    # Broad-type model
    print(f"[subtype-train] broad model ...")
    broad_handle = _finetune(
        adata.copy(), broad_col, args.scgpt_model_dir,
        ensure_dir(model_dir / "broad"), seed, args, device,
    )

    # Per-broad-type subtype models
    subtype_handles = {}
    for bt in BROAD_TYPES_FOR_SUBTYPE:
        mask = adata.obs[broad_col].astype(str) == bt
        subset = adata[mask].copy()
        if subset.n_obs == 0 or subset.obs[subtype_col].nunique() < 2:
            continue
        safe = bt.replace(" ", "_").replace("+", "plus")
        print(f"[subtype-train] {bt} ...")
        sub_handle = _finetune(
            subset, subtype_col, args.scgpt_model_dir,
            ensure_dir(model_dir / f"subtype_{safe}"), seed, args, device,
        )
        subtype_handles[bt] = sub_handle
        print(f"[subtype-train] {bt}: done")

    return broad_handle, subtype_handles


def subtype_predict_scgpt(predict_fn, test_path, broad_handle,
                          subtype_handles, output_dir, tool_name,
                          pred_col, seed, args):
    """Predict broad then refine per-type with subtype models."""
    prefix = pred_col.replace("_pred", "")
    bp, sp = f"{prefix}_broad_pred", f"{prefix}_subtype_pred"

    # Broad prediction
    adata = predict_fn(test_path, broad_handle, bp, seed, args)
    out = adata.copy()
    out.obs[sp] = out.obs[bp].astype(str)
    out.obs[pred_col] = out.obs[bp].astype(str)

    # Per-broad-type subtype refinement
    ensure_dir(output_dir)
    for bt, sub_h in subtype_handles.items():
        mask = out.obs[bp].astype(str) == bt
        if mask.sum() == 0:
            continue
        subset = out[mask].copy()
        safe = bt.replace(" ", "_").replace("+", "plus")
        tmp = output_dir / f"_tmp_{safe}.h5ad"
        subset.write_h5ad(tmp)

        a_sub = predict_fn(tmp, sub_h, sp, seed, args)
        preds = a_sub.obs[sp].astype(str)
        broad_check = preds.map(get_broad_type)
        accepted = preds.where(broad_check == bt, bt)

        out.obs.loc[subset.obs_names, sp] = preds.values
        out.obs.loc[subset.obs_names, pred_col] = accepted.values
        tmp.unlink(missing_ok=True)

    save_subtype_results(out, pred_col, bp, sp, test_path, output_dir, tool_name)


# ============================================================
# Auto-discover scGPT checkpoint
# ============================================================

def _find_scgpt_model_dir():
    env = os.environ.get("SCGPT_MODEL_DIR", "").strip()
    if env and (Path(env) / "best_model.pt").exists():
        return Path(env)
    candidates = [
        Path(__file__).resolve().parent / "models" / "scGPT_human",
        Path(__file__).resolve().parent / "models" / "whole_human",
        Path(__file__).resolve().parent.parent / "scGPT" / "models" / "scGPT_human",
        Path(__file__).resolve().parent.parent / "scGPT" / "models" / "whole_human",
    ]
    for p in candidates:
        if (p / "best_model.pt").exists():
            return p
    return None


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = create_argument_parser("scGPT")
    parser.add_argument("--scgpt_model_dir", type=Path, default=None,
                        help="Path to whole-human pretrained scGPT checkpoint")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--schedule_ratio", type=float, default=DEFAULT_SCHEDULE_RATIO)
    args = parser.parse_args()

    if args.scgpt_model_dir is None:
        args.scgpt_model_dir = _find_scgpt_model_dir()
        if args.scgpt_model_dir is None:
            raise FileNotFoundError(
                "scGPT pretrained model dir not found. "
                "Set --scgpt_model_dir or SCGPT_MODEL_DIR env var."
            )
    print(f"[scGPT] pretrained model: {args.scgpt_model_dir}")

    run_annotation_pipeline(
        "scGPT", "scgpt_pred",
        train_scgpt, predict_scgpt, args,
        train_subtype_fn=train_subtype_scgpt,
        subtype_predict_fn=subtype_predict_scgpt,
    )
