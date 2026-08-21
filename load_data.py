from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from config import BATCH_SIZE, NUM_WORKERS

def data_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform_train
    )

    train_load = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=True
    )

    return train_load
