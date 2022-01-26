from typing import List, Optional, Callable, Tuple

import torch 
import torch.nn as nn
import torch.optim as optim

import numpy as np

activations = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh()
}

initialisers = {
    "xavier": nn.init.xavier_uniform_
}

# Idea from https://github.com/ku2482/rltorch/blob/master/rltorch/network/builder.py
def create_initialiser(initialiser_key: str) -> Callable:
    def initialiser(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            initialisers[initialiser_key](module.weight)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    return initialiser

def create_fully_connected_network(input_size: int, output_size: int, hidden_layers: List[int], output_activation: Optional[str], initialiser: str = "xavier") -> nn.Sequential:
    assert initialiser in initialisers.keys(), f"Currently only supports {initialisers.keys()} for initialisers"
    assert output_activation is None or output_activation in activations.keys(), f"Currently only supports {activations.keys()} for activations"
    assert len(hidden_layers) != 0, "Requires at least one hidden layer"

    current_neurons = hidden_layers[0]

    layers = [
        nn.Linear(input_size, current_neurons),
        nn.ReLU()
    ]

    for connected_neurons in hidden_layers[1:]:
        layers.append(nn.Linear(current_neurons, connected_neurons))
        layers.append(nn.ReLU())
        current_neurons = connected_neurons 
    
    layers.append(nn.Linear(current_neurons, output_size))

    if output_activation is not None:
        layers.append(activations[output_activation])

    layers = nn.Sequential(*layers).apply(create_initialiser(initialiser))
    return layers

class Critic(nn.Module):
    def __init__(self, input_size, output_size, connected_size=256):
        super().__init__()
        self.layers = create_fully_connected_network(input_size, output_size, [connected_size, connected_size], output_activation=None)
    
    def forward(self, states, actions):
        inp = torch.cat([states, actions], dim=1)
        return self.layers(inp)

class TwinnedCritics(nn.Module):
    def __init__(self, input_size, output_size, connected_size=256):
        super().__init__()
        self.Q1 = Critic(input_size, output_size, connected_size)
        self.Q2 = Critic(input_size, output_size, connected_size)
    
    def forward(self, states, actions):
        q1_out = self.Q1(states, actions)
        q2_out = self.Q2(states, actions)
        
        return q1_out, q2_out
        
class SACGaussianPolicy(nn.Module):
    # TODO: Understand what these actually mean
    # Need to rewrite all of this to understand really
    # https://github.com/ku2482/soft-actor-critic.pytorch/blob/1688d3595817d85c2bcd279bb8ec71221b502899/code/model.py#L17
    # https://github.com/seungeunrho/minimalRL/blob/7597b9af94ee64536dfd261446d795854f34171b/sac.py#L48
    eps = 1e-6
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, input_size, output_size, hidden_layers=[256, 256]):
        super().__init__()
        self.layers = create_fully_connected_network(input_size, output_size * 2, hidden_layers, output_activation=None)
        print(self.layers)
    
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = torch.chunk(self.layers(states), 2, dim=-1)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std
    
    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Gaussian distribution
        means, log_stds = self.forward(states)
        stds = log_stds.exp()
        normals = torch.distributions.Normal(means, stds)

        # Sample actions
        xs = normals.rsample()
        actions = torch.tanh(xs)

        # Entropies
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(means)

        