import os
import random
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scanpy as sc
import torch
from scipy.sparse import csr_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def _copy_source_expression(adata):
    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    else:
        adata.layers["counts"] = adata.X.copy()
    return adata


def preprocess_scdecorr(adata, label_col, batch_col):
    """Apply the preprocessing reported for scDecorr."""
    del label_col
    adata = _copy_source_expression(adata)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(2_000, adata.n_vars),
        flavor="seurat_v3",
        batch_key=batch_col,
    )
    hvg_mask = adata.var["highly_variable"].to_numpy(copy=True)
    return adata[:, hvg_mask].copy()


def _load_official_components():
    package_dir = Path(__file__).resolve().parent / "scdecorr"
    if not package_dir.is_dir():
        raise FileNotFoundError(
            f"Official scDecorr source is missing: {package_dir}"
        )
    sys.path.insert(0, str(package_dir.parent))
    sys.path.insert(0, str(package_dir))
    common_utils = sys.modules.get("utils")
    spec = spec_from_file_location("utils", package_dir / "utils.py")
    official_utils = module_from_spec(spec)
    sys.modules["utils"] = official_utils
    try:
        spec.loader.exec_module(official_utils)
        from augment import Transform
        from dataset import MultiBatchDataset
        from models.btwins import DSBNBarlowTwins
    finally:
        if common_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = common_utils
    return Transform, MultiBatchDataset, DSBNBarlowTwins


def integrate_scdecorr(adata, seed, run_out_path, args):
    input_path = os.path.join(run_out_path, "scdecorr_input.h5ad")
    batch_codes = adata.obs[args.batch_key].astype("category").cat.codes

    exchange = sc.AnnData(
        X=csr_matrix(adata.X),
        obs={"_scdecorr_batch": batch_codes.to_numpy()},
        var=adata.var.iloc[:, 0:0].copy(),
    )
    exchange.obs_names = adata.obs_names.copy()
    exchange.write_h5ad(input_path)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    Transform, MultiBatchDataset, DSBNBarlowTwins = _load_official_components()
    n_domains = exchange.obs["_scdecorr_batch"].nunique()
    parameters = SimpleNamespace(
        arch="densenet11",
        in_features=exchange.n_vars,
        dim=64,
        projector="512-512-512",
        n_domains=n_domains,
        batch_size=512,
        lambd=0.0051,
    )

    dataset = MultiBatchDataset(
        input_path,
        n_domains,
        "_scdecorr_batch",
        transform=Transform(),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=parameters.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DSBNBarlowTwins(parameters).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1_000,
        eta_min=0,
    )

    model.train()
    for epoch in range(1_000):
        epoch_loss = 0.0
        for domain_views in loader:
            domain_views = [
                (
                    views[0].to(device=device, dtype=torch.float32),
                    views[1].to(device=device, dtype=torch.float32),
                )
                for views in domain_views
            ]
            domain_labels = [
                torch.tensor([index], device=device)
                for index in range(n_domains)
            ]
            losses = [
                model(
                    domain_views[index][0],
                    domain_views[index][1],
                    domain_labels[index],
                )
                for index in range(n_domains)
            ]
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch >= 10:
            scheduler.step()
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(
                f"[Epoch {epoch + 1}] "
                f"loss={epoch_loss / len(loader):.6f}"
            )

    matrix = exchange.X.toarray()
    # The official downstream extractor returns the signed DenseNet feature
    # vector before its final ReLU/classification head.
    model.backbone.return_feature = True
    model.eval()
    embedding = []
    with torch.no_grad():
        for start in range(0, exchange.n_obs, 256):
            batch = torch.from_numpy(
                np.asarray(matrix[start:start + 256], dtype=np.float32)
            ).to(device)
            embedding.append(model.backbone(batch).cpu().numpy())
    embedding = np.concatenate(embedding, axis=0)
    if embedding.shape[0] != adata.n_obs:
        raise ValueError(
            f"scDecorr returned {embedding.shape[0]} cells; expected {adata.n_obs}"
        )
    adata.obsm["X_scDecorr"] = embedding.astype(np.float32)
    return adata


if __name__ == "__main__":
    parser = create_argument_parser("scDecorr")
    args = parser.parse_args()
    run_integration_pipeline(
        "scDecorr",
        integrate_scdecorr,
        "X_scDecorr",
        args,
        run_pca=False,
        preprocess_fn=preprocess_scdecorr,
    )
