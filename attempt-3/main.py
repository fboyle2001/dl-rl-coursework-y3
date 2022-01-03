import model as score_models
import torch
import torch.nn as nn
import torchvision
import torchinfo
import run_model

"""
Cite CIFAR-10 Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
""" 
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

def summary():
    device = "cuda:0"
    batch_size = 64
    model = score_models.NCSNpp(num_features=128, in_ch=3).to(device)
    torchinfo.summary(model, [(batch_size, 3, 32, 32), (batch_size,)])

def run():
    device = "cuda:0"
    train_loader = load_cifar10(img_width=32, train=True, batch_size=32)
    run_model.train(train_loader=train_loader, device=device, previous_save="state-epoch-10000.model")

run()