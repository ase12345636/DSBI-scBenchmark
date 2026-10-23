import torch
import torch.nn as nn


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return _GradientReversal.apply(x, self.alpha)


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, dropout=0.1):
        super().__init__()
        self.use_bn = use_bn
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class Encoder(nn.Module):
    """VAE encoder, returns (z, mu, logvar)."""
    def __init__(self, layer_sizes, use_bn=False, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(ResBlock(layer_sizes[i], layer_sizes[i + 1], use_bn=use_bn, dropout=dropout))
        self.layers = nn.Sequential(*layers)

        latent_dim = layer_sizes[-1]
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        h = self.layers(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        logvar = torch.clamp(logvar, min=-5, max=5)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Decoder(nn.Module):
    def __init__(self, layer_sizes, use_bn=False, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(ResBlock(layer_sizes[i], layer_sizes[i + 1], use_bn=use_bn, dropout=dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        return self.layers(z)


class Classifier(nn.Module):
    def __init__(self, layer_sizes, use_bn=False, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(ResBlock(layer_sizes[i], layer_sizes[i + 1], use_bn=use_bn, dropout=dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
