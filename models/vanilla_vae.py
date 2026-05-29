import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from .base_vae import BaseVAE


class VanillaVAE(BaseVAE): #i define the vanillaVAE class which inherits from the base vae class

    def __init__(self,
                 in_channels: int = 1,        # 1 for grayscale (our stamps)
                 latent_dim: int = 128, #size of the latent vector 
                 hidden_dims: List[int] = None, 
                 **kwargs) -> None: #returns nothing
        super().__init__() #initialize parent class

        self.latent_dim = latent_dim
        self.hidden_dims = list(hidden_dims) if hidden_dims else [32, 64, 128, 256, 512,512] #default number of hidden dimensions for the encoder and decoder

        #ENCODER --------------------------------------------------
        modules = [] #for the conv layers 
        ch = in_channels
        for h_dim in self.hidden_dims: #for each hidden dimension, add a conv layer with stride 2 to downsample the image, followed by batch norm and leaky relu activation
            modules.append(nn.Sequential(
                nn.Conv2d(ch, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2, inplace=True), #leaky relu with a negative slope of 0.2 (helps prevent dying neurons and allows some gradient flow for negative inputs)
            ))
            ch = h_dim
        self.encoder = nn.Sequential(*modules) #pack all conv blocks

        #after 5 stride-2 convs on 64×64 input → 2×2 feature map
        self.fc_mu  = nn.Linear(self.hidden_dims[-1] * 2 * 2, latent_dim) #fully connected layer to map the flattened feature map to the mean of the latent distribution
        self.fc_var = nn.Linear(self.hidden_dims[-1] * 2 * 2, latent_dim) #fully connected layer to map the flattened feature map to the log variance of the latent distribution

        #DECODER --------------------------------------------------
        self.decoder_input = nn.Linear(latent_dim, self.hidden_dims[-1] * 2 * 2) #maps z back from latent space to a tensor (i can reshape it) 

        dec_dims = list(reversed(self.hidden_dims)) # mirror the encoder in reverse so now it will be [512, 512, 256, 128, 64, 32]
        modules = []
        for i in range(len(dec_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(dec_dims[i], dec_dims[i + 1], #conv transpose is a inverse convolution (doubles the spatial size)
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.BatchNorm2d(dec_dims[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            ))
        self.decoder = nn.Sequential(*modules)

        #final layer outputs 1 channel (grayscale), Sigmoid → [0,1]
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(dec_dims[-1], dec_dims[-1],
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(dec_dims[-1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dec_dims[-1], out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid(),   # sigmoid squashes output to [0,1] range. necessary bc the BCE loss expects values between 0 and 1 ( also the stamps are normalized to [0,1] )
        )

    # FORDWARD PASSES --------------------------------------------------
    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return [self.fc_mu(h), self.fc_var(h)]

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor: #THE KEY VAE TRICK! --> allows gradients to flow through the sampling. problem: sampling z ~ N(μ,σ) is not differentiable. solution: z = μ + ε·σ where ε ~ N(0,1) is separate random noise. now gradients can flow through μ and σ, ε is just noise
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor: #takes z and returns the reconstruced image 
        h = self.decoder_input(z)
        h = h.view(-1, self.hidden_dims[-1], 2, 2)
        h = self.decoder(h)
        return self.final_layer(h)

    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]: #full forward pass: encode → reparameterize → decode
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]

    # LOSS ---------------------------------------------------
    def loss_function(self, *args, **kwargs) -> dict: #compute TOTAL VAE LOSS = L_recon + λ_KL * L_KL
        recon, x, mu, log_var = args[0], args[1], args[2], args[3]
        kld_weight = kwargs.get('M_N', 1.0)  #thisi is what controls the low-KL vs high-KL behaviour! low value (0.1) = reconstruction dominates // high value (1.0) = KL pushes the latent space toward gaussian

        recon_loss = F.binary_cross_entropy(recon, x, reduction='sum') / x.size(0) #BCE between reconstructed and original image 
        kld_loss   = -0.5 * torch.mean(
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )
        loss = recon_loss + kld_weight * kld_loss
        return {
            'loss': loss,
            'Reconstruction_Loss': recon_loss.detach(),
            'KLD': kld_loss.detach(),
        }
    

    