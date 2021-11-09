from models import NCSNpp
import torch
import torch.nn as nn
import torch.optim as optim
from vesde import VESDE
import numpy as np
import time
import os
import sampler
import torchvision
import time

def optimise_fn(optimiser, params, step, lr=2e-4, warmup=5000, grad_clip=True):
    if warmup > 0:
        for g in optimiser.param_groups:
            g['lr'] = lr * np.minimum(step / warmup, 1.0)

    if grad_clip >= 0:
        torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

    optimiser.step()

def get_model_fn(model, train=False):
    def model_fn(x, labels):
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)
    
    return model_fn

def get_score_fn(sde, model, train=False):
    model_fn = get_model_fn(model, train=train)

    def score_fn(x, t):
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
        score = model_fn(x, labels)
        return score
    
    return score_fn

def get_sde_loss_fn(sde, train, eps=1e-5):
    reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        score_fn = get_score_fn(sde, model, train=train)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score = score_fn(perturbed_data, t)

        losses = torch.square(score * std[:, None, None, None] + z)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)

        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_step_fn(sde, train, optimise_fn=None):
    loss_fn = get_sde_loss_fn(sde, train)

    def step_fn(batch, model, optimiser, step):
        if train:
            optimiser.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimise_fn(optimiser, model.parameters(), step)
        else:
            with torch.no_grad():
                loss = loss_fn(model, batch)
        
        return loss
    
    return step_fn

def train(device, train_loader, eval_loader, vis, batch_size, n_epochs=1300001, snapshot_freq=100):
    score_model = NCSNpp(3, 128, nn.SiLU(), device).to(device)
    opt = optim.Adam(score_model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8)
    sde = VESDE()
    opt_fn = optimise_fn
    # cont = True

    train_step_fn = get_step_fn(sde, train=True, optimise_fn=opt_fn)
    eval_step_fn = get_step_fn(sde, train=False, optimise_fn=opt_fn)
    sampling_fn = sampler.get_sampling_fn(sde, (batch_size, 3, 32, 32))

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    current_time = time.time()

    snapshot_folder = f"./runs/{current_time}/snapshots"
    os.makedirs(snapshot_folder)

    sample_folder = f"./runs/{current_time}/samples"
    os.makedirs(sample_folder)

    train_iterator = iter(cycle(train_loader))
    eval_iterator = iter(cycle(eval_loader))
    total_loss = 0

    start_time = time.time()

    for step in range(n_epochs):
        images, _ = next(train_iterator)
        images = images.to(device)

        loss = train_step_fn(images, score_model, opt, step)

        if step == 0:
            if vis:
                vis.scatter([[0, loss.detach().cpu()]], opts=dict(
                    title="Avg. Loss",
                    xlabel="Epoch",
                    ylabel="Avg. Loss"
                ), win="avg_loss")
        else:
            total_loss += loss.detach().cpu()

        if step != 0 and step % 25 == 0:
            it = step // 25
            avg_loss = total_loss / 25
            duration = time.time() - start_time
            print(f"[Step {step}] Loss: {loss}, Avg: {avg_loss}, Took {duration:.3f}s")

            if vis:
                vis.scatter([[it, avg_loss]], win="avg_loss", update="append")

            total_loss = 0

        if step != 0 and step % snapshot_freq == 0 or step == n_epochs - 1 or step == n_epochs:
            save_step = step // snapshot_freq
            torch.save(score_model.state_dict(), f"{snapshot_folder}/snapshot-{save_step}.model")

            print("Sampling...")
            sample, n = sampling_fn(score_model)
            print("Sampled")
            nrow = int(np.sqrt(sample.shape[0]))
            image_grid = torchvision.utils.make_grid(sample, nrow=nrow)

            if vis:
                vis.images(image_grid, nrow=nrow, win="sample_images")

            torchvision.utils.save_image(image_grid, f"{sample_folder}/sample-{save_step}.png")
        
        del images

    
