import torch
import torch.nn.functional as F


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             kld_weight: float = 1.0) -> torch.Tensor:
    """
    BCE reconstruction loss + KL divergence, normalised by batch size.
    recon_x and x must both be in [0, 1].
    """
    batch_size = x.size(0)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return recon_loss + kld_weight * kld