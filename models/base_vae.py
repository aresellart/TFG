from abc import abstractmethod
import torch
from torch import nn
from typing import List, Any


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, *args: Any, **kwargs: Any) -> dict:
        raise NotImplementedError

    def sample(self, num_samples: int, device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x)[0]