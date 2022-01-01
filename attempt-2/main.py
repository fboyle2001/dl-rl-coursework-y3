import model as score_models
import torch
import torch.nn as nn
import torchvision
import torchsummary

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

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

train_loader = load_cifar10(img_width=32, train=True)
train_iterator = iter(cycle(train_loader))

device = "cuda"

eps=1e-6
model = score_models.NCSNpp(num_features=128, in_ch=3).to(device)
batch, _ = next(train_iterator)
batch = batch.to(device)
t = torch.rand(batch.shape[0], device=batch.device) * (1 - eps) + eps
model(batch, t)