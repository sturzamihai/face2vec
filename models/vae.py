from typing import List, Dict, Any

import torch
import os
import torch.nn as nn
from torch import Tensor


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_channels: int = 3, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()

        self.embedding_dim = embedding_dim

        encoder_modules = []
        hidden_dims = [32, 64, 128, 256, 512]

        for dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU(),
                )
            )
            input_channels = dim

        self.encoder = nn.Sequential(*encoder_modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4 * 4, embedding_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1] * 4 * 4, embedding_dim)
        self.decoder_input = nn.Linear(embedding_dim, hidden_dims[-1] * 4 * 4)

        decoder_modules = []
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hidden_dims[i],
                        out_channels=hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*decoder_modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_dims[-1],
                out_channels=hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=hidden_dims[-1], out_channels=3, kernel_size=3, padding=1
            ),
            nn.Tanh(),
        )

        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__), '../weights/vae.pt')
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return [mu, log_var]

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(x)
        x = x.view(-1, 512, 4, 4)
        x = self.decoder(x)
        x = self.final_layer(x)

        return x

    def reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        x_hat = self.decode(z)

        return [x, x_hat, mu, log_var]

    def loss(self, x, x_hat, mu, log_var, **kwargs) -> List[Tensor]:
        kld_weight = 0.00025
        if "kld_weight" in kwargs:
            kld_weight = kwargs["kld_weight"]

        recons_loss = nn.functional.mse_loss(x_hat, x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss

        return [loss, recons_loss.detach(), kld_loss.detach()]
