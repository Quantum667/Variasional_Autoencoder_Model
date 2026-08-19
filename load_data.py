import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from config import BATCH_SIZE

train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

train_load = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    shuffle=True
)
