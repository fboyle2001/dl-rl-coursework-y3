import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from sde import VESDE
from model import NCSNpp
import ode_sampler
import math
import time
import sde_sampler
import sys

class TestModel:
    def __init__(self, img_width=48, verbose=False, strict_checks=True):
        self.device = "cuda:0"
        self.model = NCSNpp(num_features=128, in_ch=3).to(self.device)
        self.sde = VESDE()
        self.img_width = img_width
        self.verbose = verbose
        self.strict_checks = strict_checks

    def load_parameters(self, path):
        model_info = torch.load(path, map_location="cpu")
        self.model.load_state_dict(model_info["model_state"])

    def _eval_check(self):
        if self.strict_checks:
            assert not self.model.training, "Model cannot be in training mode for this function"
        else:
            if self.model.training:
                print("[WARN] Model is in training mode. Is this intentional?")

    def sample_probability_flow_ode(self, batch_size, prior_sample=None, show=True, return_raw=False):
        self._eval_check()
        
        time_start = time.time()
        print("Starting sampling from ODE")
        solution, shape = ode_sampler.probability_flow_sampler(batch_size, self.img_width, self.model, self.sde, prior_sample=prior_sample, verbose=self.verbose)
        print(f"Finished ODE sampling took {time.time() - time_start:.3f} seconds")

        if not show and return_raw:
            return solution, shape 

        images = ode_sampler.convert_ivp_solution_shape(solution, shape, scaled=True)

        if not show and not return_raw:
            return images

        grid = torchvision.utils.make_grid(images, nrow=int(math.sqrt(images.shape[0])))
        grid = grid.detach().cpu().permute(1, 2, 0)

        plt.imshow(grid)
        plt.show(block=True)

        if return_raw:
            return solution, shape

        return images

    def sample_rsde(self, batch_size, prior_sample=None):
        self._eval_check()

        x, denoised_x = sde_sampler.predictor_corrector_sampling(batch_size, self.img_width, self.model, self.sde)

        np.save("x_normal.npy", x.detach().cpu().numpy())
        np.save("dnx.npy", denoised_x.detach().cpu().numpy())

def load_and_display_npy(path, shape=None):
    data = npy.load(path)

    if shape is not None:
        data = data.reshape(shape)
    
    data = torch.tensor(data)
    grid = torchvision.utils.make_grid(data, nrow=int(math.sqrt(data.shape[0])))
    grid = grid.detach().cpu().permute(1, 2, 0)

    plt.imshow(grid)
    plt.show(block=True)

load_and_display_npy("dnx.npy")
sys.exit(0)

test = TestModel()
test.verbose = True
test.load_parameters("../attempt-1/model_states/3/state-epoch-90000.model")
test.model.eval()
test.sample_rsde(128)
# test.sample_probability_flow_ode(32)