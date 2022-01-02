import torch 
from vesde import VESDE
import sampler
import sys
import numpy as np
import torchvision
import time

sys.path.append("F:/Documents/Development/GitHub/dl-rl-coursework-y3/attempt-2")

import model as score_models

def load_state(path):
    model_info = torch.load(path)
    return model_info["epoch"], model_info["model_state"], model_info["optimiser"]

def sample_from_model(sde, model, batch_size=128):
    sample_fn = sampler.get_sampling_fn(sde, (batch_size, 3, 32, 32))
    print("Sampling...")
    sample, n = sample_fn(model)
    print("Sampled")
    nrow = int(np.sqrt(sample.shape[0]))
    image_grid = torchvision.utils.make_grid(sample, nrow=nrow)

    torchvision.utils.save_image(image_grid, f"./model_states/1/sample-{time.time()}.png")

model_state_path = "./model_states/1/state-epoch-14000.model"
device = "cuda:0"
model = score_models.NCSNpp(num_features=128, in_ch=3).to(device)
_, model_state, _ = load_state(model_state_path)

model.load_state_dict(model_state)
model.eval()
sde = VESDE()

sample_from_model(sde, model)