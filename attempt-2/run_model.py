import torch
import torch.nn as nn
import torch.optim as optim
from sde import VESDE
from model import NCSNpp
import time
import os
import numpy as np
import old_models
import sampling

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train(train_loader, epochs=1300001, device="cuda:0", colab=False):
    model = NCSNpp(num_features=128, in_ch=3).to(device)
    # model = old_models.NCSNpp(in_ch=3, nf=128, activation_fn=nn.SiLU(), device=device).to(device)
    score_opt = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8)
    sde = VESDE()
    train_iterator = iter(cycle(train_loader))
    total_loss = 0
    loss_freq = 25
    start = time.time()
    save_freq = 1500
    
    save_folder = f"./models/{start}/saves"
    sample_folder = f"./models/{start}/samples"

    if colab:
        save_folder = f"/content/gdrive/My Drive/models/{start}/saves"
        sample_folder = f"/content/gdrive/My Drive/models/{start}/samples"
    
    os.makedirs(save_folder)
    os.makedirs(sample_folder)

    for epoch in range(epochs):
        batch, _ = next(train_iterator)
        batch = batch.to(device)
        
        loss = step_fn(sde, model, batch) #TODO: Calculate the loss

        score_opt.zero_grad() # Might need to go above loss = 
        loss.backward()
        optimise_fn(score_opt, model.parameters(), epoch) # TODO: Change, scale learning rate and optimise parameters
        total_loss += loss.detach().cpu()

        if epoch % loss_freq == 0 and epoch != 0:
            avg_loss = total_loss / loss_freq
            duration = time.time() - start

            print(f"[Epoch {epoch}] Avg: {avg_loss}, Took {duration:.3f}s")

            start = time.time()
            total_loss = 0

        if (epoch % save_freq == 0 or epoch == epochs - 1) and epoch != 0:
            print(f"Reached sampling epoch {epoch}")
            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            sampling.sample_from_model(model, path=sample_folder)

            
            # save_model(epoch, model, score_opt, f"{save_folder}/state-epoch-{epoch}.model")

def save_model(epoch, model, opt, path):
    trainable_state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimiser": opt.state_dict()
    }

    torch.save(trainable_state, path)

def optimise_fn(optimiser, params, step, lr=2e-4, warmup=5000, grad_clip=True):
    if warmup > 0:
        for g in optimiser.param_groups:
            g['lr'] = lr * np.minimum(step / warmup, 1.0)

    if grad_clip >= 0:
        torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

    optimiser.step()

def step_fn(sde, model, batch, eps=1e-6):
    # Uniformly get a tensor of batch_size random numbers in the range [eps, 1 - eps)
    t = torch.rand(batch.shape[0], device=batch.device) * (1 - eps) + eps #sde.T?
    # Get the noise for the Brownian motion, we will scale using the perturbation kernel 
    z = torch.randn_like(batch)
    # Perturbation kernel, p_t(x), which is used to bring the data closer to noise
    mean, std = sde.perturbation_kernel(batch, t)
    # Perturb the data and shift it closer to noise using the kernel
    perturbed_data = mean + std[:, None, None, None] * z

    # What is labels? Need to figure out what marginal prob is
    # marginal prob is p_t(x)
    labels = std # sde.marginal_prob(torch.zeros_like(perturbed_data), t)[1]
    
    # Put the model into train mode this won't be necessary, just call once at start
    model.train()
    # Do a forward pass on the model with the perturbed data
    score = model(perturbed_data, labels)

    # Calculate the loss objective
    # Essentially difference between 
    reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    losses = torch.square(score * std[:, None, None, None] + z)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)

    # This is what we will backprop on
    loss = torch.mean(losses)
    return loss