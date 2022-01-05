import torch
import torch.nn as nn
import torch.optim as optim
from sde import VESDE
from model import NCSNpp
import time
import os
import numpy as np

def load_state(path):
    # Load to CPU due to surge in memory usage when loading
    # otherwise get a CUDA runtime error on GPU
    model_info = torch.load(path, map_location="cpu")
    return model_info["epoch"], model_info["model_state"], model_info["optimiser"]

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train(train_loader, epochs=1300001, device="cuda:0", colab=False, previous_save=None):
    model = NCSNpp(num_features=128, in_ch=3).to(device)
    # model = old_models.NCSNpp(in_ch=3, nf=128, activation_fn=nn.SiLU(), device=device).to(device)
    score_opt = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8)
    start_epoch = 0

    sde = VESDE()
    train_iterator = iter(cycle(train_loader))
    total_loss = 0
    loss_freq = 25
    start = time.time()
    save_freq = 2000
    
    gdrive_prefix = "/content/gdrive/MyDrive/"
    save_folder = f"models/{start}/saves"
    sample_folder = f"models/{start}/samples"

    if colab:
        save_folder = f"{gdrive_prefix}{save_folder}"
        sample_folder = f"{gdrive_prefix}{sample_folder}"
    else:
        save_folder = f"./{save_folder}"
        sample_folder = f"./{sample_folder}"
    
    os.makedirs(save_folder)
    os.makedirs(sample_folder)

    if previous_save is not None:
        print("Restoring training from previous saved state")
        load_path = previous_save

        if colab:
            load_path = f"{gdrive_prefix}{load_path}"
        else:
            load_path = f"./{load_path}"

        print(f"Location: {load_path}")

        saved_epoch, model_state, opt_state = load_state(load_path)

        model.load_state_dict(model_state)
        model.to(device)
        score_opt.load_state_dict(opt_state)

        start_epoch = int(saved_epoch) + 1

        print(f"Starting from epoch {start_epoch}")

    epsilon = 1e-6
    time_sample_distribution = torch.distributions.uniform.Uniform(0, 1)
    z_normal_distribution = torch.distributions.normal.Normal(0, 1)
    max_learning_rate = 2e-4
    warmup_period = 5000
    calculate_learning_rate = lambda epoch: min((epoch / warmup_period) * max_learning_rate, max_learning_rate)

    model.train()

    for epoch in range(start_epoch, epochs):
        batch, _ = next(train_iterator)
        batch = batch.to(device)

        score_opt.zero_grad()

        # Eq 6 of Song's NCSN paper
        # t is uniformly sampled over [0, T] for VESDE T = 1 so we sample over uniform on range [0, 1) since 
        # probability of sampling any specific x ∈ [0, 1] uniformly is 0. Generates batch_size samples.
        t = time_sample_distribution.sample((batch.shape[0],)).to(device)
        
        # We compute the value of sigma(t) for each value t 
        time_sigmas = sde.sigma(t)
        # Will need to add this to the mean (the batch in this case) so reshape so it can be done
        time_sigmas_reshaped = time_sigmas.reshape(time_sigmas.shape[0], 1, 1, 1)

        # importantly, the transition kernel is always Gaussian with known mean and variance
        # for the VESDE the mean is simply the data and the standard deviation is sigma as defined in VESDE
        # this is because the drift coefficient of the SDE is affine since f(x, t) = 0 
        # cite Sarkka and Solin 2019
        gaussian_noise = z_normal_distribution.sample(batch.shape).to(device)

        # Equivalent to sampling from N(x(0), sigma(t)**2) to get the perturbed data with the noise
        # what about subtracting sigma(0)**2 ??? 

        # print("A")

        # print(batch.shape)
        # print(time_sigmas_reshaped.shape)
        # print(gaussian_noise.shape)

        # print((time_sigmas_reshaped * gaussian_noise).shape)

        perturbed_batch = batch + time_sigmas_reshaped * gaussian_noise
        # compute the estimate of the score using the model that is being trained
        score = model(perturbed_batch, time_sigmas)

        # compute the loss, need to link back to an equation
        expectation_objective = torch.square(time_sigmas_reshaped * score + gaussian_noise)
        # take the half sum
        # merge the final 3 dimensions and then take the half sum over the new final axis
        loss_per_instance = 0.5 * torch.sum(expectation_objective.reshape(expectation_objective.shape[0], -1), -1)
        # average over the batch
        batch_loss = torch.mean(loss_per_instance)
        # back prop
        batch_loss.backward()

        if epoch <= warmup_period:
            new_learning_rate = calculate_learning_rate(epoch)

            for param_group in score_opt.param_groups:
                param_group["lr"] = new_learning_rate

        # Clip the gradients and then optimise
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        score_opt.step()

        # TODO: Replace this
        # optimise_fn(score_opt, model.parameters(), epoch) # TODO: Change, scale learning rate and optimise parameters
        total_loss += batch_loss.detach().cpu()

        if epoch % loss_freq == 0 and epoch != 0:
            avg_loss = total_loss / loss_freq
            duration = time.time() - start

            print(f"[Epoch {epoch}] Avg: {avg_loss}, Took {duration:.3f}s")

            start = time.time()
            total_loss = 0

        if (epoch % save_freq == 0 or epoch == epochs - 1) and epoch != 0:
            # print(f"Reached sampling epoch {epoch}")
            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            # sampling.sample_from_model(model, path=sample_folder)
            save_model(epoch, model, score_opt, f"{save_folder}/state-epoch-{epoch}.model")
            print("Model saved")

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
    print("T shape", t.shape) # [batch_size]
    # Get the noise for the Brownian motion, we will scale using the perturbation kernel 
    z = torch.randn_like(batch) #[32, 3, 32, 32]
    # Perturbation kernel, p_t(x), which is used to bring the data closer to noise
    mean, std = sde.perturbation_kernel(batch, t) # mean is [32, 3, 32, 32] std is [32]
    print("mean, std shape", mean.shape, std.shape)
    print("std :NNN shape", std[:, None, None, None].shape) #[32, 1, 1, 1]
    print(std[:, None, None, None] * z)
    # Perturb the data and shift it closer to noise using the kernel
    perturbed_data = mean + std[:, None, None, None] * z
    print("perturbed_data shape", perturbed_data.shape) #[32, 3, 32, 32]

    # What is labels? Need to figure out what marginal prob is
    # marginal prob is p_t(x)
    labels = std # sde.marginal_prob(torch.zeros_like(perturbed_data), t)[1] #[32]
    
    # Put the model into train mode this won't be necessary, just call once at start
    model.train()
    # Do a forward pass on the model with the perturbed data
    score = model(perturbed_data, labels) #[32, 3, 32, 32]
    print("Score shape", score.shape)

    # Calculate the loss objective
    # Essentially difference between 
    reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    l = score * std[:, None, None, None] + z #[32, 3, 32, 32]
    print("L0 shape", l.shape)
    losses = torch.square(l) #[32, 3, 32, 32]
    print("L1 shape", losses.shape)
    losses = losses.reshape(losses.shape[0], -1) #[32, 3072]
    print("L2 shape", losses.shape)
    losses = 0.5 * torch.sum(losses, dim=-1) #[32]
    print("L3 shape", losses.shape)

    # 0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
    

    # This is what we will backprop on
    loss = torch.mean(losses)
    print("L* shape", loss.shape) #[1] or just float

    return loss