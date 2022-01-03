import torch 
import vesde as oldsde
import sampler
import sys
import numpy as np
import torchvision
import time

sys.path.insert(1, "F:/Documents/Development/GitHub/dl-rl-coursework-y3/attempt-3")

import model as score_models

def load_state(path):
    model_info = torch.load(path)
    return model_info["epoch"], model_info["model_state"], model_info["optimiser"]

def sample_from_model(model, epoch, batch_size=128, path="./model_states/2/"):
    sde = oldsde.VESDE()
    sample_fn = sampler.get_sampling_fn(sde, (batch_size, 3, 32, 32))
    print("Sampling...")
    sample, n = sample_fn(model)
    print("Sampled")
    nrow = int(np.sqrt(sample.shape[0]))
    image_grid = torchvision.utils.make_grid(sample, nrow=nrow)

    torchvision.utils.save_image(image_grid, f"{path}sample-{epoch}.png")

epoch = 10000
model_state_path = f"./model_states/2/state-epoch-{epoch}.model"
print(f"Sampling {model_state_path}")
device = "cuda:0"
model = score_models.NCSNpp(num_features=128, in_ch=3).to(device)
_, model_state, _ = load_state(model_state_path)

model.load_state_dict(model_state)
model.eval()

sample_from_model(model, epoch)