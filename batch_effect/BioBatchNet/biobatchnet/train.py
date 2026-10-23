import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import scanpy as sc
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from .config import Config
from .module import IMCModule, SeqModule
from .utils.dataset import BBNDataset
from .utils.tools import load_adata, evaluate, visualization_with_leiden

def train(config: Config, seed: int = 42, run_name: Optional[str] = None, do_eval: bool = True, devices: int = 1, save_root: Optional[str] = None):
    pl.seed_everything(seed, workers=True)
    config.seed = seed

    save_root = Path(save_root) if save_root else Path.cwd() / 'saved'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{run_name}" if run_name else timestamp

    run_dir = save_root / config.preset / run_name
    save_dir = run_dir / f'seed_{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    preprocess = config.data.preprocess if config.data.preprocess is not None else (config.mode == 'seq')
    adata = sc.read_h5ad(str(config.data.path))
    adata_raw = adata.copy()

    data, batch_labels, cell_types, batch_names, cell_type_names = load_adata(
        str(config.data.path),
        data_type=config.mode,
        preprocess=preprocess,
        batch_key=config.data.batch_key,
        cell_type_key=config.data.cell_type_key,
    )

    in_sz = data.shape[1]
    num_batch = len(np.unique(batch_labels))

    dataset = BBNDataset(data, batch_labels, cell_types, batch_names, cell_type_names)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.trainer.batch_size,
        shuffle=True, num_workers=4, persistent_workers=True,
    )

    # Model and trainer
    Module = IMCModule if config.mode == 'imc' else SeqModule
    model = Module(config, in_sz, num_batch, save_dir=save_dir)

    # Try wandb, fall back to no logger
    try:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project='biobatchnet', name=f'{config.preset}_{run_name}_seed{seed}')
    except Exception:
        logger = False

    trainer = pl.Trainer(
        max_epochs=config.trainer.epochs,
        callbacks=[EarlyStopping(monitor='loss', patience=config.trainer.early_stop, mode='min')],
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator='auto',
        strategy='ddp' if devices > 1 else 'auto',
        devices=devices,
        deterministic=True,
    )

    config.to_yaml(run_dir / 'config.yaml')
    trainer.fit(model, dataloader)

    # Evaluation (single GPU / rank 0 only)
    eval_metrics = {}
    if trainer.is_global_zero:
        eval_loader = torch.utils.data.DataLoader(dataset, batch_size=config.trainer.batch_size, shuffle=False)
        z_bio, z_batch = model.get_embeddings(eval_loader)
        adata.obsm['X_biobatchnet'] = z_bio
        adata.obsm['X_batch'] = z_batch

        if do_eval:
            eval_metrics = evaluate(
                adata, adata_raw,
                embed='X_biobatchnet',
                batch_key=config.data.batch_key,
                label_key=config.data.cell_type_key,
                fraction=config.trainer.sampling_fraction,
            )
            for k, v in eval_metrics.items():
                print(f"{k}: {v:.4f}")

            visualization_with_leiden(
                adata, save_path=save_dir / 'umap.png',
                batch_key=config.data.batch_key,
                label_key=config.data.cell_type_key,
                seed=seed,
            )

        adata.write(save_dir / 'biobatchnet.h5ad')
        eval_metrics = {k: float(v) for k, v in eval_metrics.items()}

        # Save metrics
        metrics_path = save_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({'preset': config.preset, 'run_name': run_name, 'seed': seed, **eval_metrics}, f, indent=2)

    return model, adata, eval_metrics


def _set_nested_attr(obj, key: str, value):
    parts = key.split('.')
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def main():
    parser = argparse.ArgumentParser(description='BioBatchNet Training')
    parser.add_argument('--config', type=str, required=True, help='Preset name or YAML path')
    parser.add_argument('--data', type=str, required=True, help='Path to .h5ad file')
    parser.add_argument('--batch_key', type=str, default=None, help='Batch column in obs (default: BATCH)')
    parser.add_argument('--cell_type_key', type=str, default=None, help='Cell type column in obs (default: celltype)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_root', type=str, default=None, help='Root dir for outputs (default: ./saved)')
    parser.add_argument('--no-eval', action='store_true', help='Skip evaluation')
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs')

    # Overrides
    for key in ['loss.recon', 'loss.discriminator', 'loss.classifier',
                'loss.ortho', 'loss.kl_bio', 'loss.kl_batch']:
        parser.add_argument(f'--{key}', type=float, default=None)
    parser.add_argument('--trainer.epochs', type=int, default=None)
    parser.add_argument('--trainer.batch_size', type=int, default=None)
    parser.add_argument('--trainer.lr', type=float, default=None)

    args = parser.parse_args()

    config = Config.load(args.config)
    config.data.path = args.data
    if args.batch_key:
        config.data.batch_key = args.batch_key
    if args.cell_type_key:
        config.data.cell_type_key = args.cell_type_key

    # Apply numeric overrides
    override_keys = ['loss.recon', 'loss.discriminator', 'loss.classifier',
                     'loss.ortho', 'loss.kl_bio', 'loss.kl_batch',
                     'trainer.epochs', 'trainer.batch_size', 'trainer.lr']
    for key in override_keys:
        value = getattr(args, key)
        if value is not None:
            _set_nested_attr(config, key, value)

    train(config, args.seed, run_name=args.run_name, do_eval=not args.no_eval, devices=args.devices, save_root=args.save_root)


if __name__ == '__main__':
    main()
