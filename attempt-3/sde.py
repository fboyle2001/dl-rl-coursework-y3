import numpy as np
import torch

class VESDE:
    """
    Forward SDES are of the form dX = f(X, t) * dt + g(t) * dW
    Reverse SDEs are of the form dX = [f(X, t) - g(t)^2 * S(X, t)] * dt + g(t) * d(~W)
    where X a tensor consisting of the batch of images, t is the timestamp scalar,
    W is Brownian motion and ~W is Reverse-Time Brownian motion, and S is the score (∇_X[ln_(p_t) (x)])
    (or S is an approximation thereof such as a deep neural network)
    f is known as the drift and g is the diffusion

    For the Variance Exploding SDE (VESDE), f(X, t) = 0 (in the shape of the tensor X) and g(t) = σ * sqrt(2 * (ln(σ_max) - ln(σ_min)))
    where σ = (σ_max)^t / (σ_min)^(t-1)

    Song et al. used σ_min = 0.01 and σ_max = 50 in their repo
    (see https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py#L208)
    """
    def __init__(self, sigma_min=0.01, sigma_max=50):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.ln_sqr_sigma_ratio = np.log((sigma_max / sigma_min) ** 2)

    def sigma(self, t):
        """
        t may be a tensor here, in that case we get a tensor out with each component being the power
        """
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def drift_coeff(self, x, t):
        """
        f: Drift coefficient of the SDE
        """

        return torch.zeros_like(x)

    def diffusion_coeff(self, t):
        """
        g: Diffusion coefficient of the SDE
        """

        return self.sigma(t) * self.ln_sqr_sigma_ratio

    def reverse_drift_coeff(self, x, t, score_function):
        """
        f(X, t) - g(t)^2 * S(X, t)
        """

        return self.drift_coeff(x, t) - (self.diffusion_coeff(t) ** 2) * score(x, t)

    def reverse_diffusion_coeff(self, t):
        """
        g(t)
        """

        return self.diffusion_coeff(t)

    def perturbation_kernel(self, x, t):
        """
        p_t(x)
        See Eq(29) in Appendix B from Song et al. x(0) is simply the data i.e. x in this case

        (Note in the next line x is a general variable not the batch) 
        Recall z = (x - μ) / σ (maybe cite prob 1 notes?) ⇔ σ * z + μ = x 

        Returns the Gaussian coefficients for rescaling N(0, 1) to match the perturbation kernel
        at a given timestep t
        """
        
        return x, self.sigma(t)


"""
Notes

Upsampling and Downsampling:
- Use Finite Impulse Response (FIR) from (Zhang, 2019)
- Use the same implementation and hyper-parameters as in StyleGAN-2
    => note this is rough to setup because of the Nvidia stuff, worth leaving until I have a working version

Skip Connections:
- Need to see if these are even used in NCSN++

Architecture of NCSN++:
- Uses FIR up/down sampling
- Rescales skip connections by 1 / sqrt(2)
- Uses BigGAN-type residual blocks
- 4 residual blocks per resolution instead of 2 (double to achieve deep)
- No progressive growing architecture for output
- "Residual" for input 

Continuous:
- Tancik et al. 2020 => use random Fourier feature embeddings with scale fixed to 16 

Sampling:
- One corrector step per update of the predictor for VE SDEs with SNR = 0.16

Interpolation:
- Need to use the Probability Flow ODE dX = [f(X, t) - 1/2 * g(t) ** 2 * S(X, t)] * dt => determined from SDE
- Neural ODE
- By integrating the Flow ODE can encode X(0) into latent space X(T)
- Decoding is integrating the corresponding ODE for the reverse-time SDE 
- Allows slerping to interpolate images
"""