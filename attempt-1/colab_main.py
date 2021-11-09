from models import NCSNpp
import torch
import torch.nn as nn
import torchvision
import torchsummary
from runner import train

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

def main(device):
    train_loader = load_cifar10(img_width=32, batch_size=32, train=True)
    eval_loader = load_cifar10(img_width=32, batch_size=32, train=False)

    train(device, train_loader, eval_loader, vis=None, n_epochs=1300001, snapshot_freq=1000)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    main(device)
    