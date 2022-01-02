import abc
import runner
import torch

class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.
        Args:
        x: A PyTorch tensor representing the current state
        t: A Pytorch tensor representing the current time step.
        Returns:
        x: A PyTorch tensor of the next state.
        x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.
        Args:
            x: A PyTorch tensor representing the current state
            t: A PyTorch tensor representing the current time step.
        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean

class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr

        alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean

def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = runner.get_score_fn(sde, model, train=False)
    predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)

def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = runner.get_score_fn(sde, model, train=False)
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)

def get_pc_sampler(sde, shape, predictor, corrector, snr=0.16, n_steps=1, eps=1e-3, denoise=True, device="cuda"):
    def pc_sampler(model):
        with torch.no_grad():
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):
                if i % 100 == 0:
                    print(f"Done {i} of {sde.N} samples")
                    
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = shared_corrector_update_fn(x, vec_t, sde, model, corrector, continuous=True, snr=snr, n_steps=n_steps)
                x, x_mean = shared_predictor_update_fn(x, vec_t, sde, model, predictor, probability_flow=False, continuous=True)

            return x_mean, sde.N * (n_steps + 1)
    
    return pc_sampler

def get_sampling_fn(sde, shape):
    predictor = ReverseDiffusionPredictor
    print("Have predictor")
    corrector = LangevinCorrector
    print("Have corrector")
    sampling_fn = get_pc_sampler(sde, shape, predictor, corrector)
    print("Have sampling_fn")
    return sampling_fn