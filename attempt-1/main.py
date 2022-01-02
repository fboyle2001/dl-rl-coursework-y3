from models import NCSNpp
import torch
import torch.nn as nn
import torchvision
import torchinfo
from runner import train, visualise
import visdom

def load_cifar10(img_width, train, batch_size=64):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../data', train=train, download=False, transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(img_width)
        ])),
    shuffle=True, batch_size=batch_size, drop_last=True)

    return train_loader

def main(device, vis):
    train_loader = load_cifar10(img_width=32, batch_size=32, train=True)
    eval_loader = load_cifar10(img_width=32, batch_size=32, train=False)

    train(device, train_loader, eval_loader, vis, batch_size=32, n_epochs=1300001, snapshot_freq=1000)
    # visualise(32, train_loader, device)

def summary():
    device = "cuda:0"
    batch_size = 32
    model = NCSNpp(in_ch=3, nf=128, activation_fn=nn.SiLU(), device=device).to(device)
    torchinfo.summary(model, [(batch_size, 3, 32, 32), (batch_size,)])

summary()

# if __name__ == "__main__":
#     # vis = visdom.Visdom()
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")
#     main(device, vis=None)
    