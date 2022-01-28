from typing import List, Optional, Callable, Tuple

import torch 
import torch.nn as nn

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

class Critic(nn.Sequential):
    def __init__(self, input_size: int, output_size: int, connected_size: int = 256):
        super().__init__(
            *create_fully_connected_network(input_size, output_size, [connected_size, connected_size], output_activation=None)
        )

class Actor(nn.Sequential):
    def __init__(self, input_size: int, output_size: int, connected_size: int = 256):
        super().__init__(
            *create_fully_connected_network(input_size, output_size, [connected_size, connected_size], output_activation="tanh")
        )

class GaussianActor(nn.Module):
    def __init__(self, state_dims: int, action_dims: int, connected_size: int = 256, sigma_upper_bound: float = 8, action_scale: float = 1, action_offset: float = 0):
        """
        The Gaussian Actor takes in a state and predicts the parameters of
        a Normal distribution for each dimension in the action dimensions.
        This provides some stochasticisty for the actions. As such, the output
        of the network is the mean and standard deviation of the Normals.

        For numerical stability we predict the natural logarithm of the 
        standard deviations since as σ -> 0, ln(σ) -> -inf providing improvements
        for the stability
        """
        super().__init__()

        self.action_scale = action_scale
        self.action_offset = action_offset

        # If the environment does not have actions in the range [-1, 1]
        # then we will never be able to get those outside the bounds since
        # we squash using tanh (R -> [-1, 1]) thus we may need to do a linear
        # transform to resolve this. For BipedalWalker this is fine
            
        # Stability
        self._epsilon = 1e-8

        # Clip the sigmas to prevent too great variation, essentially arbitrary
        self._log_sigma_lower_bound = torch.FloatTensor([np.log(self._epsilon)])
        self._log_sigma_upper_bound = torch.FloatTensor([np.log(8)])

        print(self._log_sigma_lower_bound)
        print(self._log_sigma_upper_bound)

        self.layers = create_fully_connected_network(state_dims, action_dims * 2, [connected_size, connected_size], output_activation=None)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # NN output
        forward_pass = self.layers(states)

        # The forward_pass has 2 * A elements so split down the middle
        # autograd will handle figuring out which ones to backprop on
        mu, log_sigma = torch.chunk(forward_pass, 2, dim=-1)

        # Clamp the sigmas, can't do in-place due to autograd
        log_sigma = log_sigma.clamp(self._log_sigma_lower_bound, self._log_sigma_upper_bound)
        
        # Map the log sigmas to the real sigmas
        sigma = log_sigma.exp()
        
        # These are now the real parameters for the Normal distribution
        return mu, sigma

    def _compute_actions(self, states: torch.Tensor, stochastic: bool = True) -> Tuple[torch.Tensor, torch.distributions.distribution.Distribution, torch.Tensor]:
        mu, sigma = self.forward(states)

        # Need to apply the reparameterisation trick
        dist = torch.distributions.Normal(mu, sigma)
        unbounded_actions = dist.rsample()

        # If we don't want any stochasticity in our actions
        # e.g. when evaluating the agent then we can just take the mean
        # of the Normal to be the action since this is the expectation
        # of the policy over that action dimension

        if stochastic:
            actions = torch.tanh(unbounded_actions)
        else:
            actions = torch.tanh(mu)

        # Squash into the range [-action_scale + action_offset, action_scale + action_offset]
        actions = self.action_scale * actions + self.action_offset

        return actions, dist, unbounded_actions

    def compute(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions, dist, unbounded_actions = self._compute_actions(states)

        # Appendix C of Haaraja 2018
        log_prob_actions = dist.log_prob(unbounded_actions)
        # For actions where x -> inf so tanh(x) -> 1, log(1 - 1) = undefined 
        # so log(1 - 1 + eps) stabilises the calculation
        correction = torch.log(1 - actions.square() + self._epsilon)
        log_prob_actions -= correction

        # At this point we have a tensor of the shape [B, dim(A)] and want [B, 1]
        log_prob_actions = log_prob_actions.sum(dim=-1).unsqueeze(1)
        return actions, log_prob_actions

    def compute_actions(self, states: torch.Tensor, stochastic: bool = True) -> torch.Tensor:
        actions, _, _ = self._compute_actions(states, stochastic)
        return actions
        
    def compute_actions_log_probability(self, states: torch.Tensor) -> torch.Tensor:
        _, log_prob_actions = self.compute(states)
        return log_prob_actions

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
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.layers(state)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Gaussian distribution
        means, log_stds = self.forward(state)
        stds = log_stds.exp()
        normals = torch.distributions.Normal(means, stds)

        # Sample actions
        xs = normals.rsample()
        actions = torch.tanh(xs)

        # Entropies, think I need to remove the minus
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + self.eps)
        log_probs = log_probs.sum(1, keepdim=True)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions * 2, log_probs, torch.tanh(means) * 2

        