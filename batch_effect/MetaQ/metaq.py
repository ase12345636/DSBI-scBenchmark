import os
import random
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import harmonypy
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import create_argument_parser, run_integration_pipeline


def preprocess_metaq(adata, label_col, batch_col):
    """Retain raw counts; MetaQ performs its own RNA preprocessing."""
    del label_col, batch_col
    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    return adata


def _load_official_metaq():
    source_root = Path(__file__).resolve().parent / "MetaQ"
    entry_path = source_root / "MetaQ.py"
    if not entry_path.is_file():
        raise FileNotFoundError(f"Official MetaQ source is missing: {entry_path}")

    sys.path.insert(0, str(source_root))
    spec = spec_from_file_location("official_metaq_entry", entry_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _MappingNetwork(nn.Module):
    """Paper-specified M-256-50 map from expression to integrated PCs."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, expression):
        return self.layers(expression)


def _metacell_means(expression, assignments, nonempty_ids):
    membership = sp.csr_matrix(
        (
            np.ones(assignments.size, dtype=np.float32),
            (assignments, np.arange(assignments.size)),
        ),
        shape=(int(assignments.max()) + 1, assignments.size),
    )
    sizes = np.bincount(
        assignments,
        minlength=membership.shape[0],
    )[nonempty_ids]
    means = membership[nonempty_ids] @ expression
    means = sp.diags(1.0 / sizes.astype(np.float32)) @ means
    if sp.issparse(means):
        means = means.toarray()
    return np.asarray(means, dtype=np.float32)


def _map_original_cells(expression, targets, assignments, seed, device):
    nonempty_ids = np.unique(assignments)
    metacell_expression = _metacell_means(
        expression,
        assignments,
        nonempty_ids,
    )
    target_tensor = torch.from_numpy(
        np.asarray(targets, dtype=np.float32)
    )
    dataset = TensorDataset(
        torch.from_numpy(metacell_expression),
        target_tensor,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(512, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    network = _MappingNetwork(
        metacell_expression.shape[1],
        target_tensor.shape[1],
    ).to(device)
    optimizer = torch.optim.Adam(network.parameters())
    criterion = nn.MSELoss()
    network.train()
    for _ in range(1_000):
        for batch_expression, batch_target in loader:
            batch_expression = batch_expression.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad()
            loss = criterion(network(batch_expression), batch_target)
            loss.backward()
            optimizer.step()

    network.eval()
    mapped = []
    with torch.no_grad():
        for start in range(0, expression.shape[0], 512):
            batch = expression[start:start + 512]
            if sp.issparse(batch):
                batch = batch.toarray()
            batch = torch.from_numpy(
                np.asarray(batch, dtype=np.float32)
            ).to(device)
            mapped.append(network(batch).cpu().numpy())
    return np.concatenate(mapped, axis=0)


def integrate_metaq(adata, seed, run_out_path, args):
    input_path = os.path.join(run_out_path, "metaq_input.h5ad")
    workdir = os.path.join(run_out_path, "metaq_work")

    exchange = sc.AnnData(
        X=adata.X.copy(),
        obs=adata.obs[[args.batch_key]].copy(),
        var=adata.var.iloc[:, 0:0].copy(),
    )
    exchange.write_h5ad(input_path)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    official_metaq = _load_official_metaq()
    os.makedirs(workdir, exist_ok=True)
    if adata.n_obs < 2:
        raise ValueError("MetaQ requires at least two cells")
    if args.metacell_num is None:
        metacell_num = min(adata.n_obs, max(2, round(adata.n_obs / 50)))
    else:
        metacell_num = args.metacell_num
        if not 2 <= metacell_num <= adata.n_obs:
            raise ValueError(
                "--metacell_num must be between 2 and the number of cells"
            )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    official_args = SimpleNamespace(
        data_path=[os.path.abspath(input_path)],
        data_type=["RNA"],
        save_name="metaq",
        metacell_num=metacell_num,
        type_key="__unused_celltype__",
        codebook_init="Random",
        train_epoch=300,
        batch_size=min(512, adata.n_obs),
        num_workers=0,
        skip_pairwise_metrics=True,
        skip_visualization=True,
        chunk_size=5_000,
        cache_path=None,
        converge_threshold=10,
        random_seed=seed,
        device=device.type,
    )

    previous_cwd = os.getcwd()
    try:
        os.chdir(workdir)
        official_metaq.main(official_args)
    finally:
        os.chdir(previous_cwd)

    save_dir = Path(workdir) / "save"
    metacells = sc.read_h5ad(
        save_dir / f"metaq_RNA_{metacell_num}metacell.h5ad"
    )
    assignments = sc.read_h5ad(
        save_dir / f"metaq_{metacell_num}metacell_ids.h5ad"
    ).obs["metacell"].to_numpy().astype(int)
    if assignments.shape[0] != adata.n_obs or np.any(assignments < 0):
        raise ValueError("MetaQ assignments do not match the input cells")

    batch_codes = adata.obs[args.batch_key].astype("category").cat.codes.to_numpy()
    nonempty_ids = np.unique(assignments)
    metacell_batches = np.array([
        np.bincount(batch_codes[assignments == index]).argmax()
        for index in nonempty_ids
    ])
    metacells.obs[args.batch_key] = metacell_batches.astype(str)

    n_comps = min(50, metacells.n_obs - 1, metacells.n_vars - 1)
    if n_comps < 2:
        raise ValueError("MetaQ integration requires at least three metacells")
    sc.pp.pca(metacells, n_comps=n_comps, random_state=seed)
    harmony = harmonypy.run_harmony(
        metacells.obsm["X_pca"],
        metacells.obs,
        args.batch_key,
        max_iter_harmony=30,
        nclust=15,
        theta=10.0,
        random_state=seed,
        verbose=False,
    )
    harmony_embedding = np.asarray(harmony.Z_corr.T, dtype=np.float32)
    if harmony_embedding.shape[0] != nonempty_ids.size:
        raise ValueError("MetaQ Harmony output does not match the metacells")

    # The MetaQ paper recovers single-cell embeddings by fitting an
    # M-256-50 Adam/MSE mapping on raw metacells for 1,000 epochs, then
    # applying that network to the original cells. Copying one metacell
    # coordinate to all of its members would erase within-metacell structure.
    embedding = _map_original_cells(
        adata.X,
        harmony_embedding,
        assignments,
        seed,
        device,
    )
    adata.obsm["X_MetaQ"] = embedding.astype(np.float32)
    return adata


if __name__ == "__main__":
    parser = create_argument_parser("MetaQ")
    parser.add_argument(
        "--metacell_num",
        type=int,
        default=None,
        help=(
            "Target metacell count; default is approximately one metacell "
            "per 50 cells"
        ),
    )
    args = parser.parse_args()
    run_integration_pipeline(
        "MetaQ",
        integrate_metaq,
        "X_MetaQ",
        args,
        run_pca=False,
        preprocess_fn=preprocess_metaq,
    )
