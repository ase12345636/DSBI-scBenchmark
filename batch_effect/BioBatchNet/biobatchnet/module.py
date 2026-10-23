"""PyTorch Lightning modules for BioBatchNet."""
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from .models.model import IMCBioBatchNet, SeqBioBatchNet
from .utils.loss import kl_divergence, orthogonal_loss, ZINBLoss
from .utils.tools import visualization, independence_eval
from .config import Config


class BaseBBNModule(pl.LightningModule):
    def __init__(self, config: Config, in_sz: int, num_batch: int, save_dir=None):
        super().__init__()
        self.save_hyperparameters(ignore=['config', 'save_dir'])
        self.config = config
        self.in_sz = in_sz
        self.num_batch = num_batch
        self.lw = config.loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.save_dir = save_dir

    def forward(self, x):
        return self.model(x)

    def _log_metrics(self, loss, recon_loss, discriminator_loss, classifier_loss, kl_bio, kl_batch, ortho_loss, bio_acc, batch_acc):
        self.log('loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('recon_loss', recon_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('discriminator_loss', discriminator_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('classifier_loss', classifier_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('kl_bio', kl_bio, on_step=True, on_epoch=True, sync_dist=True)
        self.log('kl_batch', kl_batch, on_step=True, on_epoch=True, sync_dist=True)
        self.log('ortho_loss', ortho_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('bio_acc', bio_acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('batch_acc', batch_acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        tc = self.config.trainer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=tc.lr,
            weight_decay=tc.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=tc.epochs,
            eta_min=1e-6,
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_embeddings(self, dataloader):
        """Extract bio and batch embeddings."""
        self.eval()
        bio_z_list, batch_z_list = [], []

        with torch.no_grad():
            for batch in dataloader:
                data = batch[0].to(self.device)
                bio_z, batch_z  = self.model.get_embeddings(data)
                bio_z_list.append(bio_z)
                batch_z_list.append(batch_z)

        bio_z = np.concatenate(bio_z_list, axis=0) if bio_z_list else None
        batch_z = np.concatenate(batch_z_list, axis=0) if batch_z_list else None
        return bio_z, batch_z

    def on_train_epoch_end(self):
        if self.trainer.global_rank != 0:
            return
        epoch = self.current_epoch + 1
        dataset = self.trainer.train_dataloader.dataset

        # UMAP visualization
        if self.save_dir and epoch % self.config.trainer.save_period == 0:
            if dataset.cell_type_names is not None:
                loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
                bio_z, batch_z = self.get_embeddings(loader)
                save_path = self.save_dir / f'umap_epoch{epoch}.png'
                visualization(bio_z, batch_z, dataset.batch_names, dataset.cell_type_names, save_path)


class IMCModule(BaseBBNModule):
    def __init__(self, config: Config, in_sz: int, num_batch: int, save_dir=None):
        super().__init__(config, in_sz, num_batch, save_dir)
        self.model = IMCBioBatchNet(config.model, in_sz, in_sz, num_batch)
        self.recon_loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        data, batch_id, *_ = batch
        out = self(data)

        recon_loss = self.recon_loss(data, out.reconstruction)
        kl_bio = kl_divergence(out.bio_mu, out.bio_logvar).mean()
        kl_batch = kl_divergence(out.batch_mu, out.batch_logvar).mean()
        discriminator_loss = self.ce_loss(out.bio_batch_pred, batch_id)
        classifier_loss = self.ce_loss(out.batch_batch_pred, batch_id)
        ortho_loss = orthogonal_loss(out.bio_z, out.batch_z)

        loss = (
            self.lw.recon * recon_loss +
            self.lw.discriminator * discriminator_loss +
            self.lw.classifier * classifier_loss +
            self.lw.kl_bio * kl_bio +
            self.lw.kl_batch * kl_batch +
            self.lw.ortho * ortho_loss
        )

        bio_acc = (out.bio_batch_pred.argmax(1) == batch_id).float().mean()
        batch_acc = (out.batch_batch_pred.argmax(1) == batch_id).float().mean()

        self._log_metrics(loss, recon_loss, discriminator_loss, classifier_loss, kl_bio, kl_batch, ortho_loss, bio_acc, batch_acc)
        return loss


class SeqModule(BaseBBNModule):
    def __init__(self, config: Config, in_sz: int, num_batch: int, save_dir=None):
        super().__init__(config, in_sz, num_batch, save_dir)
        self.model = SeqBioBatchNet(config.model, in_sz, in_sz, num_batch)
        self.recon_loss = ZINBLoss()

    def training_step(self, batch, batch_idx):
        data, batch_id, *_ = batch
        out = self(data)

        recon_loss = self.recon_loss(data, out.mean, out.disp, out.pi)
        kl_bio = kl_divergence(out.bio_mu, out.bio_logvar).mean()
        kl_batch = kl_divergence(out.batch_mu, out.batch_logvar).mean()

        discriminator_loss = self.ce_loss(out.bio_batch_pred, batch_id)
        classifier_loss = self.ce_loss(out.batch_batch_pred, batch_id)
        ortho_loss = orthogonal_loss(out.bio_z, out.batch_z)

        kl_size = kl_divergence(out.size_mu, out.size_logvar).mean()

        loss = (
            self.lw.recon * recon_loss +
            self.lw.discriminator * discriminator_loss +
            self.lw.classifier * classifier_loss +
            self.lw.kl_bio * kl_bio +
            self.lw.kl_batch * kl_batch +
            self.lw.ortho * ortho_loss +
            self.lw.kl_size * kl_size
        )

        bio_acc = (out.bio_batch_pred.argmax(1) == batch_id).float().mean()
        batch_acc = (out.batch_batch_pred.argmax(1) == batch_id).float().mean()

        self._log_metrics(loss, recon_loss, discriminator_loss, classifier_loss, kl_bio, kl_batch, ortho_loss, bio_acc, batch_acc)
        return loss


