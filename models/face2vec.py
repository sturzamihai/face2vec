import os
from typing import List, Dict, Any, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from models.mtcnn import MTCNN
from models.vae import VariationalAutoEncoder


class Face2Vec(nn.Module):
    def __init__(self, device: torch.device = None):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device is not None:
            self.device = device

        self.mtcnn = MTCNN(image_size=128, device=self.device)

        self.vae = VariationalAutoEncoder()
        state_dict_path = os.path.join(os.path.dirname(__file__), '../weights/vae_epoch15.pt')
        self.vae.load_state_dict(torch.load(state_dict_path, map_location=self.device))

    def forward(self, image: Union[torch.Tensor, Image, np.ndarray]) -> torch.Tensor:
        faces = self.mtcnn(image).unsqueeze(0).to(self.device)
        mu, log_var = self.vae.encode(faces)
        return self.vae.reparametrize(mu, log_var)
