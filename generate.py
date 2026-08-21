import os
import torch
import matplotlib.pyplot as plt
from model import Model

def generate(path, num_img=16):
    model = Model(dim=64)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()

    z = torch.randn(num_img, 64)

    with torch.no_grad():
        gen = model.decoder(z)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img = gen[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis("off")

    os.makedirs("result", exist_ok=True)

    plt.suptitle("Gen IMG", fontsize=14)
    plt.tight_layout()
    plt.savefig("result/gen.png", dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    generate("vae_cifar10_epoch_50.pth", num_img=16)
