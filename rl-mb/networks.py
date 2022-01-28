from typing import List, Optional, Callable, Tuple

import torch 
import torch.nn as nn
import torch.nn.functional as F

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

class SingleGaussianDynamics(nn.Module):
    def __init__(self, state_dims: int, action_dims: int, connected_size: int = 256, reward_dims: int = 1, hidden_layers: int = 4):
        super().__init__()

        hidden = [connected_size for _ in range(hidden_layers + 1)]
        self.layers = create_fully_connected_network(state_dims + action_dims, 2 * (state_dims + reward_dims), hidden, output_activation=None)

        # Clip the sigmas to prevent too great variation, essentially arbitrary
        self._log_sigma_lower_bound = torch.FloatTensor([np.log(1e-8)])
        self._log_sigma_upper_bound = torch.FloatTensor([np.log(8)])
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        merged_input = torch.cat([states, actions], dim=-1)
        # NN output
        forward_pass = self.layers(merged_input)

        # The forward_pass has 2 * A elements so split down the middle
        # autograd will handle figuring out which ones to backprop on
        mu, log_sigma = torch.chunk(forward_pass, 2, dim=-1)

        # Clamp the sigmas, can't do in-place due to autograd
        log_sigma = log_sigma.clamp(self._log_sigma_lower_bound, self._log_sigma_upper_bound)
        
        # Map the log sigmas to the real sigmas
        sigma = log_sigma.exp()
        
        # These are now the real parameters for the Normal distribution
        return mu, sigma

    def predict(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, sigma = self.forward(states, actions)

        # Sample the actions based on the Normal parameters
        dist = torch.distributions.Normal(mu, sigma)
        predictions = dist.rsample()

        # Split into next state predictions and reward predictions
        state_predictions, reward_predictions = predictions[:, :-1], predictions[:, -1].unsqueeze(1)
        return state_predictions, reward_predictions

class EnsembleGaussianDynamics:
    def __init__(self, state_dims: int, action_dims: int, ensemble_size: int, connected_size: int = 256, reward_dims: int = 1, hidden_layers: int = 4, lr: float = 3e-4):
        self.ensemble_size = ensemble_size
        self.models = [SingleGaussianDynamics(state_dims, action_dims, connected_size, reward_dims, hidden_layers) for _ in range(ensemble_size)]
        self.optimisers = [torch.optim.Adam(model.parameters(), lr=lr) for model in self.models]
        self.losses = [0 for _ in range(ensemble_size)]

    def single_prediction(self, states: torch.Tensor, actions: torch.Tensor, grad: bool, model_index: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        assert model_index is None or 0 <= model_index < self.ensemble_size, f"Invalid model_index {model_index}"

        if model_index is None:
            model_index = np.random.choice(np.arange(self.ensemble_size))
        
        selected_model = self.models[model_index]

        if grad:
            state_predictions, reward_predictions = selected_model.predict(states, actions)
        else:
            with torch.no_grad():
                state_predictions, reward_predictions = selected_model.predict(states, actions)
            
        return state_predictions, reward_predictions

    def weighted_ensemble_prediction(self, states: torch.Tensor, actions: torch.Tensor, weights: Optional[torch.Tensor] = None):
        if weights is None:
            weights = F.softmin(torch.FloatTensor(self.losses))

        with torch.no_grad():
            state_predictions, reward_predictions = self.single_prediction(states, actions, grad=False) * weights[0]

            for model_index in range(1, self.ensemble_size):
                s, r = self.models[model_index].predict(states, actions) * weights[model_index]
                state_predictions += s
                reward_predictions += r
        
        return state_predictions, reward_predictions

    def ensemble_prediction(self, states, actions):
        return self.weighted_ensemble_prediction(states, actions, torch.ones(self.ensemble_size))

    def train(self, states: torch.Tensor, actions: torch.Tensor, true_next_states: torch.Tensor, true_next_rewards: torch.Tensor) -> None:
        eval_masks = [np.random.choice(states.shape[0], size=int(0.2 * states.shape[0])) for _ in range(len(self.models))]
        # print("EM", eval_mask[0].shape)
        # print(true_next_states.shape, true_next_rewards.shape)

        steps_since_last_improvement = [0 for _ in range(len(self.models))]
        best_eval_losses = [None for _ in range(len(self.models))]
        improvement_threshold = 0.01
        steps_without_improvement = 5
        done = False

        while not done:
            done = True
            print(steps_since_last_improvement)
            
            for model_index in range(self.ensemble_size):
                if steps_since_last_improvement[model_index] >= steps_without_improvement:
                    continue
                    
                done = False
                eval_mask = eval_masks[model_index]

                training_mask = np.ones(states.shape[0], dtype=bool)
                training_mask[eval_mask] = False
                training_states, training_actions = states[training_mask], actions[training_mask]

                merged_truth = torch.cat([true_next_states[training_mask], true_next_rewards[training_mask]], dim=-1)

                state_predictions, reward_predictions = self.single_prediction(training_states, training_actions, grad=True)
                merged_predict = torch.cat([state_predictions, reward_predictions], dim=-1)
                loss = F.mse_loss(merged_predict, merged_truth)

                optimiser = self.optimisers[model_index]
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                with torch.no_grad():
                    eval_merged_truth = torch.cat([true_next_states[eval_mask], true_next_rewards[eval_mask]], dim=-1)
                    eval_states, eval_actions = states[eval_mask], actions[eval_mask]
                    eval_pred_states, eval_pred_rewards = self.single_prediction(eval_states, eval_actions, grad=True)
                    eval_merged_predict = torch.cat([eval_pred_states, eval_pred_rewards], dim=-1)
                    eval_loss = F.mse_loss(eval_merged_predict, eval_merged_truth)

                    if best_eval_losses[model_index] is None:
                        best_eval_losses[model_index] = eval_loss.item() # type: ignore
                    else:
                        eval_ratio = (best_eval_losses[model_index] - eval_loss) / best_eval_losses[model_index]

                        if eval_ratio < improvement_threshold:
                            steps_since_last_improvement[model_index] += 1
                        else:
                            steps_since_last_improvement[model_index] = 0
                        
                        if eval_loss < best_eval_losses[model_index]:
                            best_eval_losses[model_index] = eval_loss.item() # type: ignore
        
        self.losses = best_eval_losses

                
