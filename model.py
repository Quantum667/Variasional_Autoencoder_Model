import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, dim=16):
        super().__init__()

        self.encode1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.encode2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)

        self.f_mu = nn.Linear(64*7*7, dim)
        self.f_logvar = nn.Linear(64*7*7, dim)

        self.f_decode = nn.Linear(dim, 64*7*7)

        self.decode1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decode2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def encoder(self, x):
        x = F.relu(self.encode1(x))
        x = F.relu(self.encode2(x))
        x = x.view(x.size(0), -1)

        mu = self.f_mu(x)
        logvar = self.f_logvar(x)

        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decoder(self, z):
        x = F.relu(self.f_decode(z))
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.decode1(x))
        x = torch.sigmoid(self.decode2(x))

        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        x_regen = self.decoder(z)
        return x_regen, mu, logvar

def loss_function(x, x_reg, mu, logvar):
    reg_loss = F.binary_cross_entropy(x_reg, x, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reg_loss + kl_div
