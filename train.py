import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from model import Model, loss_function
from config import EPOCHS, LEARNING_RATE
from load_data import data_loader

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_load = data_loader()
    
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for bi, (x, _) in enumerate(train_load):
            x = x.to(device)

            optimizer.zero_grad()
            x_reg, mu, logvar = model(x)
            loss = loss_function(x_reg, x, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_load.dataset):.4f}')

    torch.save(model.state_dict(), "VAE_model.pth")
