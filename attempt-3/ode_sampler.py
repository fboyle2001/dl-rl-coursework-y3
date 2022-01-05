import numpy as np
import torch
import scipy.integrate
import math
import sde_sampler 

def probability_flow_sampler(img_count, img_width, model, sde, prior_sample=None, img_channels=3, stability=1e-5, verbose=False):
    """
    Sample from the distribution using the probability flow ODE:
    dx / dt = F(t, x) := [f(x, t) - 0.5 * g(t) ** 2 * s(x, t)]
    where f(x, t) and g(t) are the drift and diffusion coefficients of the forward SDE
    """

    # Don't want to backprop over the sample at any point
    # Without run out of memory very quickly
    with torch.no_grad():
        # Can specify prior_sample e.g. for interpolating between two prior samples
        if prior_sample is None:
            prior_sample = sde.sample_from_prior(img_count, img_width)
        else:
            print("Using prior sample given", prior_sample.shape)

        prior_shape = prior_sample.shape
        prior_sample = prior_sample.detach().cpu().numpy().reshape(-1)

        def reverse_probability_flow_ode(t, x):
            """
            The ODE to actually solve using scipy
            Note that solve_ivp takes F(t, x) as input such that dx / dt = F(t, x)
            """

            if verbose:
                print(t)
                
            # Input comes as a flat numpy array since we are using scipy
            tensor_x = torch.from_numpy(x).reshape(prior_shape).to(model.device, dtype=torch.float32)
            tensor_t = torch.full((prior_shape[0],), t).to(model.device)

            fwd_drift = sde.drift_coeff(tensor_x, tensor_t)
            fwd_diffusion = sde.diffusion_coeff(tensor_t).reshape(prior_shape[0], 1, 1, 1)

            # Estimate scores and calculate RHS of the ODE
            time_sigmas = sde.sigma(tensor_t)
            # reverse diffusion is 0 in the probability flow ODE
            rev_drift = fwd_drift - 0.5 * torch.square(fwd_diffusion) * model(tensor_x, time_sigmas)

            # Detach back to CPU and flatten back to a numpy array for scipy
            return rev_drift.detach().cpu().numpy().reshape((-1,))

        # Can't sample at t = 0 for the VESDE so the integration range is (1, s) where s is small
        integration_range = (1, stability)
        ivp_solution = scipy.integrate.solve_ivp(reverse_probability_flow_ode, integration_range, prior_sample, rtol=1e-3, atol=1e-3)
        
        return ivp_solution, prior_shape

def convert_ivp_solution_shape(ivp_solution, prior_shape, scale=False):
    """
    The IVP solution return result contains more information than we need.
    Namely, ivp_solution.y contains the values of x at each timestep t. 
    We only need the last one (unless we want to visualise the transition of noise to images)
    """

    # Select the final timestep
    image_data = ivp_solution.y[:, -1].reshape(prior_shape)

    if scale:
        image_data = (image_data + 1) / 2.
        
    image_tensor = torch.tensor(image_data)
    # print("IT", image_tensor)

    return image_tensor

def find_prior_from_image(x_0, model, sde, stability=1e-5, verbose=False):
    x_0_shape = x_0.shape
    x_0 = x_0.detach().cpu().numpy().reshape(-1)

    def reverse_probability_flow_ode(t, x,stability=1e-5):
        """
        The ODE to actually solve using scipy
        Note that solve_ivp takes F(t, x) as input such that dx / dt = F(t, x)
        """

        if verbose:
            print(t)
            
        # Input comes as a flat numpy array since we are using scipy
        tensor_x = torch.from_numpy(x).reshape(x_0_shape).to(model.device, dtype=torch.float32)
        tensor_t = torch.full((x_0_shape[0],), t).to(model.device)

        fwd_drift = sde.drift_coeff(tensor_x, tensor_t)
        fwd_diffusion = sde.diffusion_coeff(tensor_t).reshape(x_0_shape[0], 1, 1, 1)

        # Estimate scores and calculate RHS of the ODE
        time_sigmas = sde.sigma(tensor_t)
        # reverse diffusion is 0 in the probability flow ODE
        rev_drift = fwd_drift - 0.5 * torch.square(fwd_diffusion) * model(tensor_x, time_sigmas)

        # Detach back to CPU and flatten back to a numpy array for scipy
        return rev_drift.detach().cpu().numpy().reshape((-1,))

    integration_range = (stability, 1)
    ivp_solution = scipy.integrate.solve_ivp(reverse_probability_flow_ode, integration_range, x_0, rtol=1e-5, atol=1e-5)
    
    return ivp_solution, x_0_shape