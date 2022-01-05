import model as score_models
import torch
import torch.nn as nn
import torchvision
import torchinfo
import run_model
import matplotlib.pyplot as plt
import numpy as np
import time

"""
Cite CIFAR-10 Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
""" 
def load_cifar10(img_width, train, batch_size=32):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../data', train=train, download=False, transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(img_width)
        ])),
    shuffle=True, batch_size=batch_size, drop_last=True)

    return train_loader

def load_stl10(img_width=96, batch_size=32):
    stl10 = torchvision.datasets.STL10('../data', split="train+unlabeled", download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(img_width)
    ]))

    train_loader = torch.utils.data.DataLoader(stl10, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader

def load_combined_cifar10_stl10(img_width=96, batch_size=32):
    cifar10 = torchvision.datasets.CIFAR10('../data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(img_width)
    ]))

    stl10 = torchvision.datasets.STL10('../data', split="train+unlabeled", download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(img_width)
    ]))

    combined = torch.utils.data.ConcatDataset([cifar10, stl10])

    train_loader = torch.utils.data.DataLoader(combined, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader

# Lectures
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def check_params():
    device = "cuda:0"
    train_loader = load_cifar10(img_width=32, train=True, batch_size=32)
    train_iterator = iter(cycle(train_loader))
    eps=1e-6
    model = score_models.NCSNpp(num_features=128, in_ch=3).to(device)
    batch, _ = next(train_iterator)
    batch = batch.to(device)
    t = torch.rand(batch.shape[0], device=batch.device) * (1 - eps) + eps

    #print(batch.shape, t.shape)
    #return
    model(batch, t)

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def summary(batch_size=64, img_size=32):
    device = "cuda:0"
    model = score_models.NCSNpp(num_features=128, in_ch=3).to(device)
    torchinfo.summary(model, [(batch_size, 3, img_size, img_size), (batch_size,)])

def run(dataset, sample=False):
    device = "cuda:0"

    assert dataset.lower() in ["cifar10", "stl10"], "CIFAR10 or STL10 are supported"

    train_loader = None

    if dataset.lower() == "cifar10":
        train_loader = load_cifar10(img_width=32, train=True, batch_size=32)
    else: 
        train_loader = load_stl10(img_width=64, batch_size=8)

    if sample:
        it = iter(cycle(train_loader))
        batch, _ = next(it)
        nrow = int(np.sqrt(batch.shape[0]))
        image_grid = torchvision.utils.make_grid(batch, nrow=nrow)
        torchvision.utils.save_image(image_grid, f"{dataset}-{time.time()}.png")
        plt.imshow(image_grid.data.permute(0, 2, 1).contiguous().permute(2, 1, 0))

    run_model.train(train_loader=train_loader, device=device)

def combined_dataset_runner():
    device = "cuda:0"
    train_loader = load_combined_cifar10_stl10()

    it = iter(cycle(train_loader))
    batch, _ = next(it)

    print(batch.shape)
    
    nrow = int(np.sqrt(batch.shape[0]))
    image_grid = torchvision.utils.make_grid(batch, nrow=nrow)
    torchvision.utils.save_image(image_grid, f"{time.time()}.png")
    plt.imshow(image_grid)
    
run("cifar10", sample=True)

# summary(batch_size=16, img_size=64)