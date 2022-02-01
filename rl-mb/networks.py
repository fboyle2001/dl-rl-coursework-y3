from mimetypes import init
from optparse import Option
from typing import List, Optional, Callable, Tuple, Union
from unicodedata import bidirectional

import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

activations = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh()
}

initialisers = {
    "xavier": nn.init.xavier_uniform_,
    "orthogonal": nn.init.orthogonal_
}

# Idea from https://github.com/ku2482/rltorch/blob/master/rltorch/network/builder.py
def create_initialiser(initialiser_key: str) -> Callable:
    def initialiser(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            initialisers[initialiser_key](module.weight)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    return initialiser

def create_fully_connected_network(input_size: int, output_size: int, hidden_layers: List[int], output_activation: Optional[str], initialiser: str = "orthogonal") -> nn.Sequential:
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
    def __init__(self, input_size: int, output_size: int, connected_size: int = 512, hidden_count = 2):
        super().__init__(
            *create_fully_connected_network(input_size, output_size, [connected_size for _ in range(hidden_count + 1)], output_activation=None)
        )

        print(self)

class Actor(nn.Sequential):
    def __init__(self, input_size: int, output_size: int, connected_size: int = 512, hidden_count = 2):
        super().__init__(
            *create_fully_connected_network(input_size, output_size, [connected_size for _ in range(hidden_count + 1)], output_activation="tanh")
        )

class GaussianActor(nn.Module):
    def __init__(self, state_dims: int, action_dims: int, device: Union[str, torch.device], connected_size: int = 512,
                 sigma_upper_bound: float = 8, action_scale: float = 1, action_offset: float = 0):
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
        self.device = device

        self.action_scale = action_scale
        self.action_offset = action_offset

        # If the environment does not have actions in the range [-1, 1]
        # then we will never be able to get those outside the bounds since
        # we squash using tanh (R -> [-1, 1]) thus we may need to do a linear
        # transform to resolve this. For BipedalWalker this is fine
            
        # Stability
        self._epsilon = 1e-8

        # Clip the sigmas to prevent too great variation, essentially arbitrary
        self._log_sigma_lower_bound = torch.tensor([-5], device=self.device)
        self._log_sigma_upper_bound = torch.tensor([2], device=self.device)

        print(self._log_sigma_lower_bound)
        print(self._log_sigma_upper_bound)

        self.layers = create_fully_connected_network(state_dims, action_dims * 2, [connected_size, connected_size, connected_size], output_activation=None).to(self.device)

        print(self.layers)

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

    def compute(self, states: torch.Tensor, stochastic: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        actions, dist, unbounded_actions = self._compute_actions(states, stochastic)

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

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self.eps = 1e-8

    def fit(self, data: torch.Tensor) -> None:
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None and self.std is not None, "Must fit before transforming"
        return (data - self.mean) / (self.std + self.eps)

class SingleGaussianDynamics(nn.Module):
    def __init__(self, state_dims: int, action_dims: int, device: Union[str, torch.device],
                 connected_size: int = 256, reward_dims: int = 1, hidden_layers: int = 4):
        super().__init__()
        self.device = device

        hidden = [connected_size for _ in range(hidden_layers + 1)]
        self.layers = create_fully_connected_network(state_dims + action_dims, 2 * (state_dims + reward_dims), hidden, output_activation=None)

        # Clip the sigmas to prevent too great variation, essentially arbitrary
        self._log_sigma_lower_bound = torch.tensor([np.log(1e-8)], device=self.device)
        self._log_sigma_upper_bound = torch.tensor([np.log(8)], device=self.device)
    
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
    def __init__(self, state_dims: int, action_dims: int, ensemble_size: int, device: Union[str, torch.device],
                 connected_size: int = 256, reward_dims: int = 1, hidden_layers: int = 4, lr: float = 3e-4):
        self.device = device
        self.ensemble_size = ensemble_size
        self.models = [SingleGaussianDynamics(state_dims, action_dims, device, connected_size, reward_dims, hidden_layers).to(self.device) for _ in range(ensemble_size)]
        self.optimisers = [torch.optim.Adam(model.parameters(), lr=lr) for model in self.models]
        self.losses = [0 for _ in range(ensemble_size)]
        self.scaler = StandardScaler()
        self._has_trained = False

    def single_predict(self, states: torch.Tensor, actions: torch.Tensor, grad: bool, model_index: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        assert model_index is None or 0 <= model_index < self.ensemble_size, f"Invalid model_index {model_index}"

        if model_index is None:
            model_index = np.random.choice(np.arange(self.ensemble_size))
        
        selected_model = self.models[model_index]
        # Scale
        states, actions = self.scaler.transform(torch.cat([states, actions], dim=-1)).chunk(2, dim=-1)

        if grad:
            state_predictions, reward_predictions = selected_model.predict(states, actions)
        else:
            with torch.no_grad():
                state_predictions, reward_predictions = selected_model.predict(states, actions)
            
        return state_predictions, reward_predictions

    def weighted_ensemble_predict(self, states: torch.Tensor, actions: torch.Tensor, weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if weights is None:
            weights = F.softmin(torch.tensor(self.losses))

        with torch.no_grad():
            state_predictions, reward_predictions = self.single_predict(states, actions, grad=False, model_index=0) * weights[0]

            for model_index in range(1, self.ensemble_size):
                s, r = self.models[model_index].predict(states, actions) * weights[model_index]
                state_predictions += s
                reward_predictions += r
        
        return state_predictions, reward_predictions

    def ensemble_predict(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.weighted_ensemble_predict(states, actions, torch.ones(self.ensemble_size))

    def batch_predict(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = np.arange(states.shape[0])
        np.random.shuffle(indices)
        spaced_indices = np.linspace(0, states.shape[0], num=self.ensemble_size + 1, dtype=np.uint32)

        state_predictions = None
        reward_predictions = None

        for model_index in range(self.ensemble_size):
            batch_indices = indices[spaced_indices[model_index] : spaced_indices[model_index + 1]]
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]

            batch_state_predictions, batch_reward_predictions = self.single_predict(batch_states, batch_actions, grad=False, model_index=model_index)

            if state_predictions is None or reward_predictions is None:
                state_predictions = batch_state_predictions
                reward_predictions = batch_reward_predictions
            else:
                state_predictions = torch.cat([state_predictions, batch_state_predictions], dim=0)
                reward_predictions = torch.cat([reward_predictions, batch_reward_predictions], dim=0)
        
        assert state_predictions is not None and reward_predictions is not None
        return state_predictions, reward_predictions

    @property
    def absolute_average_loss(self) -> float:
        if not self._has_trained:
            return 100
        
        return np.array(self.losses).mean()

    def train(self, states: torch.Tensor, actions: torch.Tensor, true_next_states: torch.Tensor, true_next_rewards: torch.Tensor) -> None:
        self._has_trained = True
        eval_masks = [np.random.choice(states.shape[0], size=int(0.2 * states.shape[0])) for _ in range(len(self.models))]

        # Fit and transform
        joined_states_actions = torch.cat([states, actions], dim=-1)
        self.scaler.fit(joined_states_actions)
        states, actions = self.scaler.transform(joined_states_actions).chunk(2, dim=-1)

        steps_since_last_improvement = [0 for _ in range(len(self.models))]
        best_eval_losses = [None for _ in range(len(self.models))]
        improvement_threshold = 0.01
        steps_without_improvement = 5
        done = False

        while not done:
            done = True
            
            for model_index in range(self.ensemble_size):
                if steps_since_last_improvement[model_index] >= steps_without_improvement:
                    continue
                    
                done = False
                eval_mask = eval_masks[model_index]

                training_mask = np.ones(states.shape[0], dtype=bool)
                training_mask[eval_mask] = False
                training_states, training_actions = states[training_mask], actions[training_mask]
                training_true_states, training_true_rewards = true_next_states[training_mask], true_next_rewards[training_mask]

                for batch_start in range(0, training_states.shape[0], 256):
                    merged_truth = torch.cat([training_true_states[batch_start : batch_start + 256], training_true_rewards[batch_start : batch_start + 256]], dim=-1)

                    state_predictions, reward_predictions = self.single_predict(training_states[batch_start : batch_start + 256], training_actions[batch_start : batch_start + 256], grad=True)
                    merged_predict = torch.cat([state_predictions, reward_predictions], dim=-1)
                    loss = F.mse_loss(merged_predict, merged_truth)

                    optimiser = self.optimisers[model_index]
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                with torch.no_grad():
                    eval_merged_truth = torch.cat([true_next_states[eval_mask], true_next_rewards[eval_mask]], dim=-1)
                    eval_states, eval_actions = states[eval_mask], actions[eval_mask]
                    eval_pred_states, eval_pred_rewards = self.single_predict(eval_states, eval_actions, grad=True)
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

def get_params(models):
    for m in models:
        for p in m.parameters():
            yield p

class GRUDynamicsModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, horizon: int, device: Union[str, torch.device], clip_grad: Optional[float],
                    latent_dim: int = 512, coder_hidden_dim: int = 512, encoder_hidden_depth: int = 2,
                    decoder_hidden_depth: int = 0, gru_layers: int = 2, lr: float = 1e-3):
        super().__init__()
        
        """
        Encoder:
            in: obs_dim + action_dim
            hidden_dim: encoder_hidden_dim = 512
            out: latent_dim = 512
            depth: encoder_hidden_depth = 2
        
        GRU:
            in: latent_dim = 512
            out: latent_dim = 512
            num_layers: rec_num_layers = 2

        Decoder:
            in: latent_dim = 512
            hidden_dim: decoder_hidden_dim = 512
            out: obs_dim
            depth: decoder_hidden_depth = 0

        Weights: Initialise encoder and decoder but not the GRU
        Optimiser: Optimise encoder, GRU and decoder all together

        Others:
            detach_xt = True
            clip_grad_norm = 1.0
            learning_rate = 1e-3
            horizon = agent.horizon =??
            action_range = ??
        """
        
        self.device = device
        self.gru_layers = gru_layers
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.clip_grad = clip_grad

        self.encoder = create_fully_connected_network(input_size, latent_dim, hidden_layers=[coder_hidden_dim for _ in range(encoder_hidden_depth + 1)], output_activation=None)
        self.model = nn.GRU(input_size=latent_dim, hidden_size=latent_dim, num_layers=gru_layers)
        self.decoder = create_fully_connected_network(latent_dim, output_size, hidden_layers=[coder_hidden_dim for _ in range(decoder_hidden_depth + 1)], output_activation=None)

        params = get_params([self.encoder, self.model, self.decoder])
        self.optimiser = torch.optim.Adam(params, lr=lr) # type: ignore

        print("Encoder")
        print(self.encoder)

        print("Model")
        print(self.model)

        print("Decoder")
        print(self.decoder)
    
    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.predict_states(state, actions)

    def unroll_horizon(self, initial_state: torch.Tensor, actor: GaussianActor, stochastic=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert initial_state.dim() == 2
        hidden_state = torch.zeros(self.gru_layers, initial_state.shape[0], self.latent_dim).to(self.device)

        predicted_states = []
        action_sequence = []
        log_probs = []
        current_state = initial_state

        for _ in range(self.horizon - 1):
            # Sample the action to take
            actions, action_log_prob =  actor.compute(current_state, stochastic)
            action_sequence.append(actions)
            log_probs.append(action_log_prob)

            current_state = current_state.detach()

            # Estimate the change in state
            joined_states_actions = torch.cat([current_state, actions], dim=-1)
            encoded = self.encoder(joined_states_actions).unsqueeze(0) # Need to unsqueeze?
            latent_delta_next_state, hidden_state = self.model(encoded, hidden_state)

            # Move to the next state
            next_state = current_state + self.decoder(latent_delta_next_state.squeeze(0)) # Need to squeeze?
            # print("Next state shape:", next_state.shape)

            # Track the predicted states
            predicted_states.append(next_state)
            current_state = next_state
        
        # Believe the final one will be used to check termination?
        final_action, final_log_prob = actor.compute(current_state, stochastic)
        action_sequence.append(final_action)
        log_probs.append(final_log_prob)

        # print("PSL", len(predicted_states))

        policy_actions = torch.stack(action_sequence)
        predicted_states = torch.stack(predicted_states)
        log_probs = torch.stack(log_probs).squeeze(2) # Need to squeeze?

        # Log probs are need for V_(0:H)^(pi, alpha)[x] for optimisation
        return predicted_states, policy_actions, log_probs

    def predict_states(self, initial_state: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        predicted_states = []
        hidden_state = torch.zeros(self.gru_layers, initial_state.shape[0], self.latent_dim).to(self.device)
        current_state = initial_state

        for action_index in range(action_sequence.shape[0]):
            current_action = action_sequence[action_index]
            current_state = current_state.detach()

            joined_states_actions = torch.cat([current_state, current_action], dim=-1)
            encoded = self.encoder(joined_states_actions).unsqueeze(0) # Need to unsqueeze?
            latent_delta_next_state, hidden_state = self.model(encoded, hidden_state)

            # Move to the next state
            next_state = current_state + self.decoder(latent_delta_next_state.squeeze(0)) # Need to squeeze?

            # Track the predicted states
            predicted_states.append(next_state)
            current_state = next_state
        
        predicted_states = torch.stack(predicted_states)
        return predicted_states

    def train(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        # Predict states from the first state 
        predicted_states = self.forward(states[0], actions[:-1])
        # Want to match it to the real next states
        target_states = states[1:] 

        assert predicted_states.shape == target_states.shape

        # Loss function is simply the MSE
        loss = F.mse_loss(predicted_states, target_states)

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.clip_grad is not None:
            assert len(self.optimiser.param_groups) == 1
            torch.nn.utils.clip_grad_norm_(self.optimiser.param_groups[0]["params"], self.clip_grad) # type: ignore

        self.optimiser.step()

        return loss.detach().item()


class RewardModel(nn.Sequential):
    """
    hidden_dim = 512
    hidden_depth = 2
    lr = 1e-3
    """
    def __init__(self, input_size: int, output_size: int, connected_size: int = 512, hidden_count = 2):
        super().__init__(
            *create_fully_connected_network(input_size, output_size, [connected_size for _ in range(hidden_count + 1)], output_activation="tanh")
        )

class TerminationModel(nn.Sequential):
    """
    hidden_dim = 512
    hidden_depth = 2
    lr = 1e-3
    ctrl_accum = True ???? 
    """
    def __init__(self, input_size: int, output_size: int, connected_size: int = 512, hidden_count = 2):
        super().__init__(
            *create_fully_connected_network(input_size, output_size, [connected_size for _ in range(hidden_count + 1)], output_activation="tanh")
        )