"""Simple API for batch effect correction."""
import torch
import numpy as np
import scanpy as sc
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from scipy.sparse import issparse

from .config import Config, LossConfig, TrainerConfig
from .module import IMCModule, SeqModule
from .utils.dataset import BBNDataset
from .utils.tools import seq_preprocess


def correct_batch_effects(
    adata,
    batch_key='BATCH',
    cell_type_key=None,
    data_type='imc',
    latent_dim=20,
    epochs=100,
    lr=1e-4,
    batch_size=128,
    device='auto',
    loss_weights=None,
):
    """
    Batch effect correction for single-cell data.

    Args:
        adata: AnnData object with expression matrix and batch annotations.
        batch_key: Column name in adata.obs for batch labels.
        cell_type_key: Optional column name in adata.obs for cell type labels.
        data_type: 'imc' for Imaging Mass Cytometry, 'seq' for scRNA-seq.
            When 'seq', automatic preprocessing is applied (HVG 2000 + normalize + log1p).
        latent_dim: Latent space dimension (default: 20).
        epochs: Maximum training epochs (default: 100).
        lr: Learning rate (default: 1e-4).
        batch_size: Training batch size (default: 128).
        device: 'cuda', 'cpu', or 'auto' (default: 'auto').
        loss_weights: Optional dict to override default loss weights.
            Keys: 'recon', 'discriminator', 'classifier', 'kl_bio',
                  'kl_batch', 'ortho', 'kl_size' (seq only).

    Returns:
        bio_embeddings: np.ndarray of shape (n_cells, latent_dim),
            batch-corrected biological embeddings.
        batch_embeddings: np.ndarray of shape (n_cells, latent_dim),
            batch-specific embeddings.

    Example:
        >>> import scanpy as sc
        >>> from biobatchnet import correct_batch_effects
        >>> adata = sc.read_h5ad('data.h5ad')
        >>> bio_emb, batch_emb = correct_batch_effects(adata, batch_key='BATCH', data_type='imc')
        >>> adata.obsm['X_biobatchnet'] = bio_emb
    """
    # Auto-preprocess for scRNA-seq: HVG selection + normalize + log1p
    if data_type == 'seq':
        adata = seq_preprocess(adata)

    # Extract expression matrix
    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = np.array(X, dtype=np.float32)

    # Encode batch labels as integers
    batch_labels_raw = adata.obs[batch_key].values
    unique_batches = np.unique(batch_labels_raw)
    batch_to_int = {b: i for i, b in enumerate(unique_batches)}
    batch_labels = np.array([batch_to_int[b] for b in batch_labels_raw])
    batch_names = np.array([str(b) for b in batch_labels_raw])

    # Encode cell type labels if provided
    cell_types, cell_type_names = None, None
    if cell_type_key and cell_type_key in adata.obs:
        ct_raw = adata.obs[cell_type_key].values
        unique_cts = np.unique(ct_raw)
        ct_to_int = {c: i for i, c in enumerate(unique_cts)}
        cell_types = np.array([ct_to_int[c] for c in ct_raw])
        cell_type_names = np.array([str(c) for c in ct_raw])

    input_dim = X.shape[1]
    num_batches = len(unique_batches)

    # Build loss config
    if data_type == 'imc':
        loss_config = LossConfig(
            recon=10.0, discriminator=0.3, classifier=1.0,
            kl_bio=0.005, kl_batch=0.1, ortho=0.01,
        )
    else:
        loss_config = LossConfig(
            recon=10.0, discriminator=0.04, classifier=1.0,
            kl_bio=1e-6, kl_batch=0.01, ortho=0.0002, kl_size=0.002,
        )

    if loss_weights:
        for k, v in loss_weights.items():
            if hasattr(loss_config, k):
                setattr(loss_config, k, v)

    trainer_config = TrainerConfig(
        epochs=epochs, batch_size=batch_size, lr=lr,
        early_stop=100,
    )

    config = Config(
        mode=data_type,
        loss=loss_config,
        trainer=trainer_config,
    )
    config.model.latent_sz = latent_dim

    # Dataset and dataloader
    dataset = BBNDataset(X, batch_labels, cell_types, batch_names, cell_type_names)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create model
    Module = IMCModule if data_type == 'imc' else SeqModule
    model = Module(config, input_dim, num_batches)

    # Train
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[EarlyStopping(monitor='loss', patience=5, mode='min')],
        accelerator=device,
        devices=1,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, dataloader)

    # Extract embeddings
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    bio_embeddings, batch_embeddings = model.get_embeddings(eval_loader)

    return bio_embeddings, batch_embeddings
