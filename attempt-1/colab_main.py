from models import NCSNpp
import torch
import torch.nn as nn
import torchvision
import torchsummary
from runner import train
import sys

def summary(device):
    model = NCSNpp(3, 128, nn.SiLU(), device).to(device)

    print("Input shape: [-1, 3, 32, 32]")
    torchsummary.summary(model, (3, 32, 32))

def load_cifar10(img_width, train, batch_size=64):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10("./data", train=train, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(img_width)
        ])),
    shuffle=True, batch_size=batch_size, drop_last=True)

    return train_loader

def main(device, batch_size):
    train_loader = load_cifar10(img_width=32, batch_size=batch_size, train=True)
    eval_loader = load_cifar10(img_width=32, batch_size=batch_size, train=False)

    train(device, train_loader, eval_loader, vis=None, batch_size=batch_size, n_epochs=1300001, snapshot_freq=1000)

if __name__ == "__main__":
    bs = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    print(f"Using batch size {bs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    main(device, batch_size=bs)
    