import os
from typing import Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from face2vec.models.mtcnn import MTCNN
from face2vec.models.vae import VariationalAutoEncoder


class Face2Vec(nn.Module):
    def __init__(self, device: torch.device = None):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.to(device)

        self.mtcnn = MTCNN(image_size=128, device=self.device)
        self.vae = VariationalAutoEncoder(device=self.device)

    def forward(self, image: Union[torch.Tensor, Image, np.ndarray]) -> torch.Tensor:
        faces = self.mtcnn(image).unsqueeze(0).to(self.device)
        mu, log_var = self.vae.encode(faces)
        return self.vae.reparametrize(mu, log_var)

    def reconstruct(self, image: Union[torch.Tensor, Image, np.ndarray]) -> torch.Tensor:
        faces = self.mtcnn(image).unsqueeze(0).to(self.device)
        x, x_hat, mu, log_var = self.vae(faces)

        return x_hat