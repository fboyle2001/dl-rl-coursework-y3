import model as score_models
import torch
import torch.nn as nn
import torchvision
import torchsummary
import run_model
import sys

def load_cifar10(img_width, batch_size=64):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../data', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(img_width)
        ])),
    shuffle=True, batch_size=batch_size, drop_last=True)

    return train_loader

def load_stl10(img_width, batch_size=32):
    stl10 = torchvision.datasets.STL10('../data', split="train+unlabeled", download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(img_width)
    ]))

    train_loader = torch.utils.data.DataLoader(stl10, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cifar10"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    img_width = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    previous_save = sys.argv[4] if len(sys.argv) > 4 else None
    print("Using", dataset)
    print("Using batch size", batch_size)
    print("Using image size", img_width)

    train_loader = None

    if dataset == "cifar10":
        train_loader = load_cifar10(img_width=img_width, batch_size=batch_size)
    elif dataset == "stl10":
        train_loader = load_stl10(img_width=img_width, batch_size=batch_size)
    else:
        print("Invalid dataset", dataset)
        sys.exit(-1)

    device = "cuda:0"

    run_model.train(train_loader=train_loader, device=device, colab=True, previous_save=previous_save)

# eps=1e-6
# model = score_models.NCSNpp(num_features=128, in_ch=3).to(device)
# batch, _ = next(train_iterator)
# batch = batch.to(device)
# t = torch.rand(batch.shape[0], device=batch.device) * (1 - eps) + eps
# model(batch, t)