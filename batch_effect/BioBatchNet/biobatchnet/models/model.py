"""VAE models for batch effect correction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

from .layers import Encoder, Decoder, Classifier, GRL
from ..config import Config, ModelConfig


@dataclass
class IMCBioBatchNetOutput:
    bio_z: torch.Tensor
    bio_mu: torch.Tensor
    bio_logvar: torch.Tensor
    batch_z: torch.Tensor
    batch_mu: torch.Tensor
    batch_logvar: torch.Tensor
    bio_batch_pred: torch.Tensor
    batch_batch_pred: torch.Tensor
    reconstruction: torch.Tensor


@dataclass
class SeqBioBatchNetOutput:
    bio_z: torch.Tensor
    bio_mu: torch.Tensor
    bio_logvar: torch.Tensor
    batch_z: torch.Tensor
    batch_mu: torch.Tensor
    batch_logvar: torch.Tensor
    bio_batch_pred: torch.Tensor
    batch_batch_pred: torch.Tensor
    mean: torch.Tensor
    disp: torch.Tensor
    pi: torch.Tensor
    size_factor: torch.Tensor
    size_mu: torch.Tensor
    size_logvar: torch.Tensor


class IMCBioBatchNet(nn.Module):
    def __init__(self, config: ModelConfig, in_sz: int, out_sz: int, num_batch: int):
        super().__init__()
        self.config = config

        # Build layer sizes
        bio_enc_sizes = [in_sz] + config.bio_encoder_layers + [config.latent_sz]
        batch_enc_sizes = [in_sz] + config.batch_encoder_layers + [config.latent_sz]
        dec_sizes = [2 * config.latent_sz] + config.decoder_layers + [out_sz]
        bio_clf_sizes = [config.latent_sz] + config.bio_classifier_layers + [num_batch]
        batch_clf_sizes = [config.latent_sz] + config.batch_classifier_layers + [num_batch]

        self.bio_encoder = Encoder(bio_enc_sizes, use_bn=config.use_bn, dropout=config.dropout)
        self.batch_encoder = Encoder(batch_enc_sizes, use_bn=config.use_bn, dropout=config.dropout)
        self.decoder = Decoder(dec_sizes, use_bn=config.use_bn, dropout=config.dropout)
        self.bio_classifier = Classifier(bio_clf_sizes, use_bn=config.use_bn, dropout=config.dropout)
        self.batch_classifier = Classifier(batch_clf_sizes, use_bn=config.use_bn, dropout=config.dropout)

        self.grl = GRL(alpha=1.0)

    def forward(self, x):
        # Bio encoding
        bio_z, bio_mu, bio_logvar = self.bio_encoder(x)

        # Batch encoding
        batch_z, batch_mu, batch_logvar = self.batch_encoder(x)

        # Combine (detach batch_z to prevent gradients)
        z_combine = torch.cat([bio_z, batch_z.detach()], dim=1)

        # Adversarial classification
        bio_z_grl = self.grl(bio_z)
        bio_batch_pred = self.bio_classifier(bio_z_grl)

        # Batch classification
        batch_batch_pred = self.batch_classifier(batch_z)

        # Reconstruction
        reconstruction = self.decoder(z_combine)

        return IMCBioBatchNetOutput(
            bio_z=bio_z,
            bio_mu=bio_mu,
            bio_logvar=bio_logvar,
            batch_z=batch_z,
            batch_mu=batch_mu,
            batch_logvar=batch_logvar,
            bio_batch_pred=bio_batch_pred,
            batch_batch_pred=batch_batch_pred,
            reconstruction=reconstruction,
        )

    def get_embeddings(self, data):
        """Get bio and batch embeddings."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data)
            data = data.to(device)
            bio_z, _, _ = self.bio_encoder(data)
            batch_z, _, _ = self.batch_encoder(data)
            return bio_z.cpu().numpy(), batch_z.cpu().numpy()


class SeqBioBatchNet(nn.Module):
    def __init__(self, config: ModelConfig, in_sz: int, out_sz: int, num_batch: int):
        super().__init__()
        self.config = config

        # Build layer sizes
        bio_enc_sizes = [in_sz] + config.bio_encoder_layers + [config.latent_sz]
        batch_enc_sizes = [in_sz] + config.batch_encoder_layers + [config.latent_sz]
        size_enc_sizes = [in_sz] + config.bio_encoder_layers + [1]
        dec_sizes = [2 * config.latent_sz] + config.decoder_layers + [1000]
        bio_clf_sizes = [config.latent_sz] + config.bio_classifier_layers + [num_batch]
        batch_clf_sizes = [config.latent_sz] + config.batch_classifier_layers + [num_batch]

        self.bio_encoder = Encoder(bio_enc_sizes, use_bn=config.use_bn, dropout=config.dropout)
        self.batch_encoder = Encoder(batch_enc_sizes, use_bn=config.use_bn, dropout=config.dropout)
        self.size_encoder = Encoder(size_enc_sizes, use_bn=config.use_bn, dropout=config.dropout)
        self.decoder = Decoder(dec_sizes, use_bn=config.use_bn, dropout=config.dropout)

        # ZINB output layers
        self.mean_decoder = nn.Sequential(nn.Linear(1000, out_sz), MeanAct())
        self.disp_decoder = nn.Sequential(nn.Linear(1000, out_sz), DispAct())
        self.dropout_decoder = nn.Sequential(nn.Linear(1000, out_sz), nn.Sigmoid())

        self.bio_classifier = Classifier(bio_clf_sizes, use_bn=config.use_bn, dropout=config.dropout)
        self.batch_classifier = Classifier(batch_clf_sizes, use_bn=config.use_bn, dropout=config.dropout)

        self.grl = GRL(alpha=1.0)

    def forward(self, x):
        # Bio encoding
        bio_z, bio_mu, bio_logvar = self.bio_encoder(x)

        # Size factor encoding
        size_factor, size_mu, size_logvar = self.size_encoder(x)

        # Batch encoding
        batch_z, batch_mu, batch_logvar = self.batch_encoder(x)

        # Combine
        z_combine = torch.cat([bio_z, batch_z.detach()], dim=1)

        # Adversarial classification
        bio_z_grl = self.grl(bio_z)
        bio_batch_pred = self.bio_classifier(bio_z_grl)

        # Batch classification
        batch_batch_pred = self.batch_classifier(batch_z)

        # ZINB decoding
        h = self.decoder(z_combine)
        size_factor = torch.clamp(size_factor, min=-5, max=5)
        mean = self.mean_decoder(h) * torch.exp(size_factor)
        mean = torch.clamp(mean, 1e-6, 1e8)
        disp = self.disp_decoder(h)
        pi = self.dropout_decoder(h)
        pi = torch.clamp(pi, 1e-6, 1.0 - 1e-6)

        return SeqBioBatchNetOutput(
            bio_z=bio_z,
            bio_mu=bio_mu,
            bio_logvar=bio_logvar,
            batch_z=batch_z,
            batch_mu=batch_mu,
            batch_logvar=batch_logvar,
            bio_batch_pred=bio_batch_pred,
            batch_batch_pred=batch_batch_pred,
            mean=mean,
            disp=disp,
            pi=pi,
            size_factor=size_factor,
            size_mu=size_mu,
            size_logvar=size_logvar,
        )

    def get_embeddings(self, data):
        """Get bio and batch embeddings."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data)
            data = data.to(device)
            bio_z, _, _ = self.bio_encoder(data)
            batch_z, _, _ = self.batch_encoder(data)
            return bio_z.cpu().numpy(), batch_z.cpu().numpy()


class MeanAct(nn.Module):
    """Activation for mean parameter."""
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-3, max=1e3)


class DispAct(nn.Module):
    """Activation for dispersion parameter."""
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-3, max=1e3)
