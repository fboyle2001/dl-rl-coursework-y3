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
import torchinfo
import scipy

class TestModel:
    def __init__(self, img_width=48, verbose=False, strict_checks=True):
        self.device = "cuda:0"
        self.model = NCSNpp(num_features=128, in_ch=3).to(self.device)
        self.sde = VESDE()
        self.img_width = img_width
        self.verbose = verbose
        self.strict_checks = strict_checks
        self.loaded_parameters = False

    def load_parameters(self, path):
        model_info = torch.load(path, map_location="cpu")
        self.model.load_state_dict(model_info["model_state"])

        if self.verbose:
            print("Loaded network parameters")

        self.loaded_parameters = True

    def _eval_check(self):
        if self.strict_checks:
            assert not self.model.training, "Model cannot be in training mode for this function"
            assert self.loaded_parameters, "No parameters loaded!"
        else:
            if self.model.training:
                print("[WARN] Model is in training mode. Is this intentional?")
            
            if not self.model.loaded_parameters:
                print("[WARN] No model parameters loaded. Is this intentional?")

    def sample_probability_flow_ode(self, batch_size, prior_sample=None, show=True, return_raw=False):
        self._eval_check()
        
        time_start = time.time()
        print("Starting sampling from ODE")
        solution, shape = ode_sampler.probability_flow_sampler(batch_size, self.img_width, self.model, self.sde, prior_sample=prior_sample, verbose=self.verbose)
        print(f"Finished ODE sampling took {time.time() - time_start:.3f} seconds")

        if not show and return_raw:
            return solution, shape 

        images = ode_sampler.convert_ivp_solution_shape(solution, shape)

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

        x, denoised_x = sde_sampler.predictor_corrector_sampling(batch_size, self.img_width, self.model, self.sde, prior_sample=prior_sample)

        np.save("x_normal_1.npy", x.detach().cpu().numpy())
        np.save("dnx_1.npy", denoised_x.detach().cpu().numpy())

    def summarise_model(self, batch_size):
        torchinfo.summary(self.model, [(batch_size, 3, self.img_width, self.img_width), (batch_size,)])

    def batch_to_prior(self, batch):
        self._eval_check()

        solution, shape = ode_sampler.find_prior_from_image(batch, self.model, self.sde, verbose=self.verbose)
        images = ode_sampler.convert_ivp_solution_shape(solution, shape)

        np.save("z_latent_test.npy", images.detach().cpu().numpy())

        grid = torchvision.utils.make_grid(images, nrow=int(math.sqrt(images.shape[0])))
        grid = grid.detach().cpu().permute(1, 2, 0)

        plt.imshow(grid)
        plt.show(block=True)

    def likelihood_test(self, a, b):
        self._eval_check()
        
        x_0 = torch.tensor(np.load("noise_images.npy"))#.to(self.device)
        a = a.squeeze()

        # ode_sampler.find_prior_from_image(a, self.model, self.sde)
        # print("Done")

        b = b.squeeze()

        print(a.shape, b.shape)

        # print(a)

        a_n = a.reshape(-1).numpy()
        b_n = b.reshape(-1).numpy()
        num_steps = 31

        #linear_interpolation = lambda step: a_n + step * (b_n - a_n) / num_steps
        o_interpolation = lambda step: a_n * np.sin((math.pi / 2) * (step / num_steps)) + b_n * (1 - np.sin((math.pi / 2) * (step / num_steps)))
        #interp_steps = np.array([linear_interpolation(step) for step in range(num_steps + 1)])
        interp_steps = np.array([o_interpolation(step) for step in range(num_steps + 1)])
        # interp_steps = scipy.interpolate.RBFInterpolator()
        interp_steps = interp_steps.reshape((num_steps + 1, 3, self.img_width, self.img_width))
        interp_steps = torch.tensor(interp_steps)

        # sample = self.sde.sample_from_prior(1, 48)
        # print("ss", sample.shape)
        # interp_steps = torch.cat([interp_steps, sample])

        print(interp_steps.shape)
        
        time_start = time.time()
        print("Starting sampling from ODE for interpolation")
        solution, shape = ode_sampler.probability_flow_sampler(interp_steps.shape[0], self.img_width, self.model, self.sde, prior_sample=interp_steps, verbose=self.verbose)
        print(f"Finished ODE sampling took {time.time() - time_start:.3f} seconds")

        images = ode_sampler.convert_ivp_solution_shape(solution, shape)

        grid = torchvision.utils.make_grid(images, nrow=int(math.sqrt(images.shape[0])))
        grid = grid.detach().cpu().permute(1, 2, 0)

        plt.imshow(grid)
        plt.show(block=True)

        torchvision.utils.save_image(grid, f"interpolation-{time.time()}.png")

        print(x_0.shape)
        ode_sampler.find_prior_from_image(x_0, self.model, self.sde)

# https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network/
# https://github.com/soumith/dcgan.torch/issues/14#issuecomment-200025792
def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

def load_and_display_npy(path, shape=None):
    data = np.load(path)

    if shape is not None:
        data = data.reshape(shape)
    
    data = torch.tensor(data)
    grid = torchvision.utils.make_grid(data, nrow=int(math.sqrt(data.shape[0])))
    grid = grid.detach().cpu().permute(1, 2, 0)

    plt.imshow(grid)
    plt.show(block=True)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def load_stl10(img_width=48, batch_size=16):
    stl10 = torchvision.datasets.STL10('../data', split="train+unlabeled", download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(img_width)
    ]))

    train_loader = torch.utils.data.DataLoader(stl10, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader

# load_and_display_npy("x_normal_1.npy")
# load_and_display_npy("dnx_1.npy")
# load_and_display_npy("prior_images_1.npy")
# sys.exit(0)
test = TestModel(verbose=True)
test.model.eval()
test.load_parameters("../attempt-1/model_states/3/state-epoch-142000.model")

# train_loader = load_stl10()
# it = iter(cycle(train_loader))
# batch, _ = next(it)
# batch = batch.to(test.device)

# test.batch_to_prior(batch)

noise = torch.tensor(np.load("z_latent_test.npy"))

last = noise[-1, :, :, :].unsqueeze(0)
second_last = noise[-2, :, :, :].unsqueeze(0)

print(last.shape, second_last.shape)

data = torch.cat([last, second_last])
print(data.shape)

data = torch.tensor(data)
grid = torchvision.utils.make_grid(data, nrow=int(math.sqrt(data.shape[0])))
grid = grid.detach().cpu().permute(1, 2, 0)

plt.imshow(grid)
plt.show(block=True)
test.likelihood_test(last, second_last)

# print("NS", noise.shape)

# test.sample_probability_flow_ode(batch_size=noise.shape[0], prior_sample=noise)

# test.likelihood_test()
# test.sample_probability_flow_ode(32)
sys.exit(0)
test.likelihood_test()
test.verbose = True
test.model.eval()

prior_sample = test.sde.sample_from_prior(128, 48)
np.save("prior_images_1.npy", prior_sample.detach().cpu().numpy())
test.sample_rsde(128)
# test.sample_probability_flow_ode(32)