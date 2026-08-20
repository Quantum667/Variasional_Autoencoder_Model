import os
import torch
import matplotlib.pyplot as plt
from model import Model

def generate(path, num_img=16):
    model = Model(dim=16)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()

    z = torch.randn(num_img, 16)

    with torch.no_grad():
        gen = model.decoder(z)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(gen[i].squeeze(), cmap="gray")
        ax.axis("off")

    os.makedirs("result", exist_ok=True)

    plt.suptitle("Gen IMG", fontsize=14)
    plt.tight_layout()
    plt.savefig("result/gen.png", dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    generate("VAE_model.pth", num_img=16)
