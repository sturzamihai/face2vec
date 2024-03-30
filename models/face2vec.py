from typing import List, Dict, Any

import torch
import torch.nn as nn
from models.mtcnn import MTCNN
from models.vae import VariationalAutoEncoder


class Face2Vec(nn.Module):
    def __init__(self):
        super().__init__()

        self.mtcnn = MTCNN()
        self.vae = VariationalAutoEncoder()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pass
