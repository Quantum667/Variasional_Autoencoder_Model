from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from config import BATCH_SIZE, NUM_WORKERS

def data_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):


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

    return train_load
