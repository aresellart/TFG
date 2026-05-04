import torch
from torch import nn
from transformers import ResNetModel
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-50').to(device)
        hidden_size = self.resnet.config.hidden_sizes[-1]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 16)
        )

    def forward(self, img):
        ret = self.resnet(img)
        feat = ret.pooler_output.squeeze(-1).squeeze(-1) # batch, 512
        fin = self.classifier(feat)
        return feat, fin

