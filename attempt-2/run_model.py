import torch
import torch.nn as nn
import torch.optim as optim
from sde import VESDE

def train(epochs=1300001):
    score_model = None #TODO: Make the score model
    score_opt = optim.Adam(score_model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8)
    sde = VESDE()

    for epoch in range(epochs):
        images = None #TODO: Select batch of images and send to GPU
        
        loss = None #TODO: Calculate the loss

        score_opt.zero_grad() # Might need to go above loss = 
        loss.backward()
        #TODO: scale learning rate and optimise parameters


def step_fn(sde, model, batch):
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