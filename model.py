import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, dim=64):
        super().__init__()

        self.encode1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.encode2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.encode3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.encode4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        self.ebn1 = nn.BatchNorm2d(32)
        self.ebn2 = nn.BatchNorm2d(64)
        self.ebn3 = nn.BatchNorm2d(128)
        self.ebn4 = nn.BatchNorm2d(256)

        self.f_mu = nn.Linear(256*2*2, dim)
        self.f_logvar = nn.Linear(256*2*2, dim)

        self.f_decode = nn.Linear(dim, 256*2*2)

        self.decode1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.decode2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decode3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decode4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

        self.dbn1 = nn.BatchNorm2d(128)
        self.dbn2 = nn.BatchNorm2d(64)
        self.dbn3 = nn.BatchNorm2d(32)

    def encoder(self, x):
        x = F.relu(self.ebn1(self.encode1(x)))
        x = F.relu(self.ebn2(self.encode2(x)))
        x = F.relu(self.ebn3(self.encode3(x)))
        x = F.relu(self.ebn4(self.encode4(x)))
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
        x = x.view(-1, 256, 2, 2)
        x = F.relu(self.dbn1(self.decode1(x)))
        x = F.relu(self.dbn2(self.decode2(x)))
        x = F.relu(self.dbn3(self.decode3(x)))
        x = torch.sigmoid(self.decode4(x))

        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        x_regen = self.decoder(z)
        return x_regen, mu, logvar

def loss_function(x, x_reg, mu, logvar):
    reg_loss = F.mse_loss(x_reg, x, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reg_loss + kl_div
