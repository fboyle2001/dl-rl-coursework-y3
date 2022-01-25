from typing import Any, Dict, Optional, Union
import abc

import numpy as np
import torch
import torch.nn.functional as F
import copy
import buffers
import networks
import utils
from dotmap import DotMap

class RLAgent(abc.ABC):
    def __init__(self, state_dim, action_dim, device: Union[str, torch.device], parameters: DotMap):
        self.device = device
        self.parameters = parameters

        self._state_dim = state_dim
        self._action_dim = action_dim

        self._log_info = dict()
        self._steps = 0

    def store_memory(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, is_terminal: bool) -> None:
        """
        Store a replay if this agent supports experience replays
        """
        raise NotImplementedError("Agent does not support experience replays")

    def _append_log_info(self, key: str, duration: float, additional: Dict[str, Any]) -> None:
        info = {
            duration: duration,
            **additional
        }

        self._log_info[key] = info

    def get_log(self) -> Dict[str, Dict[str, Any]]:
        return self._log_info
    
    def clear_log(self) -> None:
        self._log_info = dict()

    @abc.abstractmethod
    def sample_action(self, state: np.ndarray) -> np.ndarray:
        """
        Sample an action from the agent's policy given the current state
        """
        pass

    @abc.abstractmethod
    def train(self) -> None:
        """
        Train the agent's policy
        """
        self._steps += 1

class StandardTD3Agent(RLAgent):
    def __init__(self, state_dim, action_dim, device: Union[str, torch.device], parameters: DotMap):
        """
        Required parameters:
        buffer_size => defines max number of experiences able to hold in memory
        lr
        noise_sigma
        tau
        replay_batch_size
        noise_clip
        gamma
        policy_update_frequency
        """
        super().__init__(state_dim, action_dim, device, parameters)

        # Replay buffer
        self.replay_buffer = buffers.StandardReplayBuffer(state_dim, action_dim, self.parameters.buffer_size, self.device)

        # Critics predicts the reward of taking action A from state S
        self.critic_1 = networks.Critic(input_size=state_dim + action_dim, output_size=1).to(self.device)
        self.critic_2 = networks.Critic(input_size=state_dim + action_dim, output_size=1).to(self.device)

        # Actor predicts the action to take based on the current state
        self.actor = networks.Actor(input_size=state_dim, output_size=action_dim).to(self.device)

        # Establish the target networks
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_1.eval()
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)
        self.target_critic_2.eval()
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_actor.eval()

        # Establish optimisers
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=self.parameters.lr)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=self.parameters.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.parameters.lr)

        # Establish sampling distributions
        self.noise_distribution = torch.distributions.normal.Normal(0, self.parameters.noise_sigma)

        self.steps = 0
    
    def store_memory(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, is_terminal: bool) -> None:
        self.replay_buffer.store_replay(state, action, reward, next_state, is_terminal)
    
    def update_target_parameters(self):
        # Update frozen targets, taken from https://github.com/sfujim/TD3/blob/385b33ac7de4767bab17eb02ade4a268d3e4e24f/TD3.py
        # theta' = tau * theta + (1 - tau) * theta'
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.parameters.tau * param.data + (1 - self.parameters.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.parameters.tau * param.data + (1 - self.parameters.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.parameters.tau * param.data + (1 - self.parameters.tau) * target_param.data)

    def sample_action(self, np_state: np.ndarray) -> np.ndarray:
        # Given state S what is the action A to take?
        np_state = np_state.reshape(1, -1)
        state = torch.FloatTensor(np_state).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self) -> None:
        super().train()
        
        if self.replay_buffer.count < self.parameters.replay_batch_size:
            return

        # Sample the replay buffer
        buffer_sample = self.replay_buffer.sample_buffer(self.parameters.replay_batch_size)

        # TD3 
        with torch.no_grad():
            noise = self.noise_distribution.sample(buffer_sample.actions.shape).clamp(-self.parameters.noise_clip, self.parameters.noise_clip).to(self.device)
            target_actions = self.target_actor(buffer_sample.next_states)
            target_actions = (target_actions + noise).clamp(-1, 1)

            target_input = torch.cat([buffer_sample.next_states, target_actions], dim=-1)

            target_Q1 = self.target_critic_1(target_input)
            target_Q2 = self.target_critic_2(target_input)
            min_Q_target = torch.min(target_Q1, target_Q2)

            # Temporal difference learning
            target_Q = buffer_sample.rewards.unsqueeze(1) + buffer_sample.terminals.unsqueeze(1) * self.parameters.gamma * min_Q_target 

        actual_input = torch.cat([buffer_sample.states, buffer_sample.actions], dim=-1)

        actual_Q1 = self.critic_1(actual_input)
        actual_Q2 = self.critic_2(actual_input)

        critic_loss = F.mse_loss(actual_Q1, target_Q) + F.mse_loss(actual_Q2, target_Q)

        self.critic_1_opt.zero_grad()
        self.critic_2_opt.zero_grad()

        critic_loss.backward()

        self.critic_1_opt.step()
        self.critic_2_opt.step()

        # Delayed policy updates
        if self.steps % self.parameters.policy_update_frequency == 0:
            actor_input = torch.cat([buffer_sample.states, self.actor(buffer_sample.states)], dim=-1)
            actor_loss = -self.critic_1(actor_input).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.update_target_parameters()

class PrioritisedTD3Agent(RLAgent):
    def __init__(self, state_dim, action_dim, device: Union[str, torch.device], parameters: DotMap):
        """
        Required parameters:
        leaf_count => defines max number of experiences able to hold in memory (2 ** leaf_count)
        lr
        noise_sigma
        tau
        replay_batch_size
        noise_clip
        gamma
        policy_update_frequency
        """
        super().__init__(state_dim, action_dim, device, parameters)

        # Replay buffer
        self.replay_buffer = buffers.PriorityReplayBuffer(state_dim, action_dim, self.parameters.leaf_count, self.device)

        # Critics predicts the reward of taking action A from state S
        self.critic_1 = networks.Critic(input_size=state_dim + action_dim, output_size=1).to(self.device)
        self.critic_2 = networks.Critic(input_size=state_dim + action_dim, output_size=1).to(self.device)

        # Actor predicts the action to take based on the current state
        self.actor = networks.Actor(input_size=state_dim, output_size=action_dim).to(self.device)

        # Establish the target networks
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_1.eval()
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)
        self.target_critic_2.eval()
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_actor.eval()

        # Establish optimisers
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=self.parameters.lr)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=self.parameters.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.parameters.lr)

        # Establish sampling distributions
        self.noise_distribution = torch.distributions.normal.Normal(0, self.parameters.noise_sigma)

        self.steps = 0
    
    def store_memory(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, is_terminal: bool, td_error: Optional[float]) -> None:
        self.replay_buffer.store_replay(state, action, reward, next_state, is_terminal, td_error)
    
    def update_target_parameters(self):
        # Update frozen targets, taken from https://github.com/sfujim/TD3/blob/385b33ac7de4767bab17eb02ade4a268d3e4e24f/TD3.py
        # theta' = tau * theta + (1 - tau) * theta'
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.parameters.tau * param.data + (1 - self.parameters.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.parameters.tau * param.data + (1 - self.parameters.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.parameters.tau * param.data + (1 - self.parameters.tau) * target_param.data)

    def sample_action(self, np_state: np.ndarray) -> np.ndarray:
        # Given state S what is the action A to take?
        np_state = np_state.reshape(1, -1)
        state = torch.FloatTensor(np_state).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self) -> None:
        super().train()
        
        if self.replay_buffer.count < self.parameters.replay_batch_size:
            return

        # Sample the replay buffer
        buffer_sample, buffer_indexes, buffer_weights = self.replay_buffer.sample_buffer(self.parameters.replay_batch_size)

        # TD3 
        with torch.no_grad():
            noise = self.noise_distribution.sample(buffer_sample.actions.shape).clamp(-self.parameters.noise_clip, self.parameters.noise_clip).to(self.device)
            target_actions = self.target_actor(buffer_sample.next_states)
            target_actions = (target_actions + noise).clamp(-1, 1)

            target_input = torch.cat([buffer_sample.next_states, target_actions], dim=-1)

            target_Q1 = self.target_critic_1(target_input)
            target_Q2 = self.target_critic_2(target_input)
            min_Q_target = torch.min(target_Q1, target_Q2)

            # Temporal difference learning
            target_Q = buffer_sample.rewards.unsqueeze(1) + buffer_sample.terminals.unsqueeze(1) * self.parameters.gamma * min_Q_target 

        actual_input = torch.cat([buffer_sample.states, buffer_sample.actions], dim=-1)

        actual_Q1 = self.critic_1(actual_input)
        actual_Q2 = self.critic_2(actual_input)

        critic_loss = utils.importance_sampling_mse(actual_Q1, target_Q, buffer_weights) + utils.importance_sampling_mse(actual_Q2, target_Q, buffer_weights)

        self.critic_1_opt.zero_grad()
        self.critic_2_opt.zero_grad()

        critic_loss.backward()

        self.critic_1_opt.step()
        self.critic_2_opt.step()

        # Use the TD errors to update the priorities in the replay buffer
        td_errors = torch.abs(target_Q - actual_Q1).detach().cpu().numpy()
        self.replay_buffer.update_priorities(buffer_indexes, td_errors)

        # Delayed policy updates
        if self.steps % self.parameters.policy_update_frequency == 0:
            actor_input = torch.cat([buffer_sample.states, self.actor(buffer_sample.states)], dim=-1)
            actor_loss = -self.critic_1(actor_input).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.update_target_parameters()