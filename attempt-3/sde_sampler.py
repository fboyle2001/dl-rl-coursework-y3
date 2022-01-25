import numpy as np
import torch
import scipy.integrate
import math
import time

def predictor_corrector_sampling(img_count, img_width, model, sde, prior_sample=None, img_channels=3, snr=0.16, steps=1000, verbose=True):
    """
    Uses Reverse Diffusion for prediction and Lanegvin Dynamics for correction
    1 step of each per timestep for a default of 1000 iterations
    """
    with torch.no_grad():
        x_i_one = prior_sample
        
        if prior_sample is None:
            x_i_one = sde.sample_from_prior(img_count, img_width)
        else:
            print("Using known prior_sample")

        x_i_one = x_i_one.to(model.device)
        last_denoised = None
        timesteps = torch.linspace(1e-3, 1, steps + 1).to(model.device)
        start_time = time.time()

        for i in range(steps - 1, -1, -1):
            if i % 100 == 0 and verbose:
                print(f"Done {steps - i} samples of {steps} time elapsed is {time.time()-start_time:.3f} seconds")

            l_start_time = time.time()
            # Correct
            x_i, denoised_x_i = compute_langevin_corrector_step(x_i_one, timesteps[i], sde, model, snr)
            l_end_time = time.time() - l_start_time

            r_start_time = time.time()
            # Predict
            x_i, denoised_x_i = compute_reverse_diff_predictor_step(x_i, timesteps[i], timesteps[i + 1], sde, model)
            r_end_time = time.time() - r_start_time

            print(f"L Predictor took {l_end_time:.5f} s")
            print(f"R Corrector took {r_end_time:.5f} s")

            x_i_one = x_i
            last_denoised = denoised_x_i

        return x_i_one, last_denoised

    """
    N = discrete steps for RSDE, M = no of corrector steps
    for i = N - 1 to 0 do
        x_i = previous_x_i + (sigma(i + 1) ** 2 - sigma(i) ** 2) * score(previous_x_i, sigma(i + 1))
        z ~ N(0, 1)
        x_i = x_i + sqrt(sigma(i + 1) ** 2 - sigma(i) ** 2) * z

        for j = 1 to M do
            z ~ N(0, 1)
            x_i = x_i + epsilon * score(x_i, sigma(i)) + sqrt(2 * epsilon) * z
    
        previous_x_i = x_i
    
    return previous_x_i
    """

def compute_reverse_diff_predictor_step(x, t, last_t, sde, model):
    sigma_i_plus_one = sde.sigma(last_t)
    sigma_i = sde.sigma(t)
    sigma_sqr = sigma_i_plus_one ** 2 - sigma_i ** 2
    time_sigmas = torch.full((x.shape[0],), sigma_i_plus_one).to(model.device)

    denoised_x = x + sigma_sqr * model(x, time_sigmas)
    noise = torch.randn(denoised_x.shape).to(model.device)
    x = denoised_x + math.sqrt(sigma_sqr) * noise
    
    return x, denoised_x

def compute_langevin_corrector_step(x, t, sde, model, snr):
    # Algorithm 4
    noise = torch.randn(x.shape).to(model.device)
    t = torch.full((x.shape[0],), t).to(model.device)
    score = model(x, t)
    epsilon = 2 * (snr * torch.linalg.norm(noise) / torch.linalg.norm(score)) ** 2
    denoised_x = x + epsilon * score
    x = denoised_x + math.sqrt(2 * epsilon) * noise

    return x, denoised_x