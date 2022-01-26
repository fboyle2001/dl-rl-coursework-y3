from sys import intern
from typing import Any, Dict, Optional, Union, Callable
import abc
import time
import os

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import copy
import buffers
import networks
import utils
from dotmap import DotMap
import random

class RLAgent(abc.ABC):
    def __init__(self, agent_name: str, env_name: str, device: Union[str, torch.device], video_every: Optional[int],
                    max_episodes: int = 50000, log_to_file: bool = True, log_to_screen: bool = True, seed: int = 42):
        
        self._agent_name = agent_name
        self.env = gym.make(env_name)
        self._action_dim = self.env.action_space.shape[0] # type: ignore
        self._state_dim = self.env.observation_space.shape[0] # type: ignore

        self._seed = seed
        torch.manual_seed(self._seed)
        self.env.seed(self._seed)
        random.seed(self._seed)
        np.random.seed(self._seed)
        self.env.action_space.seed(self._seed)

        self._id = f"{time.time()}-{env_name}"
        self._home_directory = f"./info/{self._agent_name}/{self._id}/"
        self._tensorboard_directory = f"{self._home_directory}tensorboard/"
        self._video_directory = f"{self._home_directory}videos/"

        os.makedirs(self._home_directory)
        os.makedirs(self._tensorboard_directory)
        os.makedirs(self._video_directory)

        self._log_directory = f"{self._home_directory}logs/"
        self._log_to_screen = log_to_screen
        self._log_to_file = log_to_file

        if self._log_to_file:
            os.makedirs(self._log_directory)
            self._log_file = open(f"{self._log_directory}log.txt", "w+")
        else:
            self._log_directory = None
            self._log_file = None

        self._writer = SummaryWriter(log_dir=self._tensorboard_directory)

        if video_every is not None:
            self.env = gym.wrappers.Monitor(self.env, self._video_directory, video_callable=lambda ep: ep % video_every == 0, force=True)

        self.device = device

        self._warmup_steps = 0
        self._steps = 0
        self._episodes = 0
        self._episode_steps = 0
        self._max_episodes = max_episodes
        self._reward_log = []

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def total_steps(self) -> int:
        return self._steps + self._warmup_steps

    @property
    def episodes(self) -> int:
        return self._episodes

    def log(self, message):
        output = f"[{time.time()}][S: {self.total_steps}][E: {self.episodes} ({self._episode_steps})] {message}"

        if self._log_to_screen:
            print(output)

        if self._log_file is not None:
            self._log_file.write(f"{output}\n")
            self._log_file.flush()

    @abc.abstractmethod
    def sample_evaluation_action(self, states: np.ndarray) -> np.ndarray:
        """
        Sample an action for use in evaluation of the policy
        e.g. without any exploration noise added
        """
        pass

    @abc.abstractmethod
    def sample_regular_action(self, states: np.ndarray) -> np.ndarray:
        """
        Sample an action for use during training 
        e.g. with exploration noise added
        """
        pass

    @abc.abstractmethod
    def train_policy(self) -> None:
        """
        Train the agent's policy on a single episode
        """
        pass

    @abc.abstractmethod
    def run(self) -> None:
        pass

class StandardTD3Agent(RLAgent):
    def __init__(self, env_name: str, device: Union[str, torch.device], video_every: Optional[int],
                    buffer_size: int = int(1e6), lr: float = 3e-4, noise_sigma: float = 0.2, 
                    tau: float = 0.005, replay_batch_size: int = 256, noise_clip: float = 0.5,
                    gamma: float = 0.99, policy_update_frequency: int = 2, exploration_noise: float = 0.1,
                    min_warmup_steps: int = 0, warmup_function: Callable[[int], int] = lambda step: 0):
        """
        TD3 Implementation
        """
        super().__init__("StandardTD3", env_name, device, video_every)

        # Replay buffer
        self.replay_buffer = buffers.StandardReplayBuffer(self._state_dim, self._action_dim, buffer_size, self.device)

        # Critics predicts the reward of taking action A from state S
        self.critic_1 = networks.Critic(input_size=self._state_dim + self._action_dim, output_size=1).to(self.device)
        self.critic_2 = networks.Critic(input_size=self._state_dim + self._action_dim, output_size=1).to(self.device)

        # Actor predicts the action to take based on the current state
        self.actor = networks.Actor(input_size=self._state_dim, output_size=self._action_dim).to(self.device)

        # Establish the target networks
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_1.eval()
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)
        self.target_critic_2.eval()
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_actor.eval()

        # Establish optimisers
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Establish sampling distributions
        self.noise_distribution = torch.distributions.normal.Normal(0, noise_sigma)

        self.tau = tau
        self.replay_batch_size = replay_batch_size
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_update_frequency = policy_update_frequency
        self.min_warmup_steps = min_warmup_steps
        self.warmup_function = warmup_function
        self.exploration_noise = exploration_noise

    def run(self) -> None:
        super().run()

        if self.min_warmup_steps != 0:
            self.log(f"Have {self.min_warmup_steps} steps")
            done = False
            state = self.env.reset()
            
            while self._warmup_steps < self.min_warmup_steps or not done:
                if done:
                    state = self.env.reset()
                    done = False
                
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.store_replay(state, action, reward, next_state, done)

                self._warmup_steps += 1
            
            self.log(f"Complete warmup steps")

        while self._episodes <= self._max_episodes:
            self._episodes += 1

            state = self.env.reset()
            done = False
            warmup_steps = self.warmup_function(self._episodes)
            episode_reward = 0
            self._episode_steps = 0

            while not done:
                if self._episode_steps < warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.sample_regular_action(state)
                
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.store_replay(state, action, reward, next_state, done)

                self._steps += 1
                self._episode_steps += 1
                episode_reward += reward
                state = next_state

                if self._episode_steps >= warmup_steps:
                    self.train_policy()

            self._reward_log.append(episode_reward)
            self._writer.add_scalar("stats/reward", episode_reward, self.episodes)
            self.log(f"Episode Reward: {episode_reward}")
    
    def sample_evaluation_action(self, states: np.ndarray) -> np.ndarray:
        np_state = states.reshape(1, -1)
        state = torch.FloatTensor(np_state).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def sample_regular_action(self, states: np.ndarray) -> np.ndarray:
        action = self.sample_evaluation_action(states)
        action = (action + np.random.normal(0, self.exploration_noise, size=self._action_dim)).clip(-1, 1)
        return action

    def train_policy(self) -> None:
        # Fill the replay buffer before we try to train the policy
        if self.replay_buffer.count < self.replay_batch_size:
            return
        
        buffer_sample = self.replay_buffer.sample_buffer(self.replay_batch_size)

        # Q Target
        with torch.no_grad():
            noise = self.noise_distribution.sample(buffer_sample.actions.shape).clamp(-self.noise_clip, self.noise_clip).to(self.device)
            target_actions = self.target_actor(buffer_sample.next_states)
            target_actions = (target_actions + noise).clamp(-1, 1)

            target_input = torch.cat([buffer_sample.next_states, target_actions], dim=-1)

            target_Q1 = self.target_critic_1(target_input)
            target_Q2 = self.target_critic_2(target_input)
            min_Q_target = torch.min(target_Q1, target_Q2)

            # Temporal Difference Learning
            target_Q = buffer_sample.rewards.unsqueeze(1) + buffer_sample.terminals.unsqueeze(1) * self.gamma * min_Q_target 
            self._writer.add_scalar("stats/target_q", target_Q.detach().cpu().mean().item(), self._steps)

        actual_input = torch.cat([buffer_sample.states, buffer_sample.actions], dim=-1)

        actual_Q1 = self.critic_1(actual_input)
        self._writer.add_scalar("stats/critic_1", actual_Q1.detach().cpu().mean().item(), self._steps)
        actual_Q2 = self.critic_2(actual_input)
        self._writer.add_scalar("stats/critic_2", actual_Q2.detach().cpu().mean().item(), self._steps)

        critic_loss = F.mse_loss(actual_Q1, target_Q) + F.mse_loss(actual_Q2, target_Q)
        self._writer.add_scalar("stats/critic_loss", critic_loss.detach().cpu().item(), self._steps)

        self.critic_1_opt.zero_grad()
        self.critic_2_opt.zero_grad()

        critic_loss.backward()

        self.critic_1_opt.step()
        self.critic_2_opt.step()

        # Delayed policy updates
        if self._steps % self.policy_update_frequency == 0:
            actor_input = torch.cat([buffer_sample.states, self.actor(buffer_sample.states)], dim=-1)
            actor_loss = -self.critic_1(actor_input).mean()
            
            self._writer.add_scalar("stats/actor_loss", actor_loss.detach().cpu().item(), self._steps)

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.update_target_parameters()

    def update_target_parameters(self):
        # Update frozen targets, taken from https://github.com/sfujim/TD3/blob/385b33ac7de4767bab17eb02ade4a268d3e4e24f/TD3.py
        # theta' = tau * theta + (1 - tau) * theta'
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

a = StandardTD3Agent("BipedalWalker-v3", "cpu", 10, warmup_function=lambda episode: max(0, 300 - episode))
a.run()

class StandardSACAgent(RLAgent):
    def __init__(self, env_name: str, device: Union[str, torch.device], video_every: Optional[int],
                    buffer_size: int = int(1e6), lr: float = 3e-4, tau: float = 0.005,
                    replay_batch_size: int = 256, gamma: float = 0.99, gradient_steps: int = 1,
                    min_warmup_steps: int = 10000, target_update_interval: int = 2,
                    warmup_function: Callable[[int], int] = lambda step: 0):
        """
        TD3 Implementation
        """
        super().__init__("StandardSAC", env_name, device, video_every)

        # Replay buffer
        self.replay_buffer = buffers.StandardReplayBuffer(self._state_dim, self._action_dim, buffer_size, self.device)

        # Critics predicts the reward of taking action A from state S
        self.critic_1 = networks.Critic(input_size=self._state_dim + self._action_dim, output_size=1).to(self.device)
        self.critic_2 = networks.Critic(input_size=self._state_dim + self._action_dim, output_size=1).to(self.device)

        # Actor predicts the action to take based on the current state
        self.actor = networks.Actor(input_size=self._state_dim, output_size=self._action_dim).to(self.device)

        # Establish the target networks
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_1.eval()
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)
        self.target_critic_2.eval()
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_actor.eval()

        # Establish optimisers
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Automatic temperature optimisation
        self.target_entropy = -np.prod(self.env.action_space.shape) # type: ignore

        self.tau = tau
        self.replay_batch_size = replay_batch_size
        self.target_update_interval = target_update_interval
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.min_warmup_steps = min_warmup_steps
        self.warmup_function = warmup_function
    
    def run(self) -> None:
        super().run()

        if self.min_warmup_steps != 0:
            self.log(f"Have {self.min_warmup_steps} steps")
            done = False
            state = self.env.reset()
            
            while self._warmup_steps < self.min_warmup_steps or not done:
                if done:
                    state = self.env.reset()
                    done = False
                
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.store_replay(state, action, reward, next_state, done)

                self._warmup_steps += 1
            
            self.log(f"Complete warmup steps")

        while self._episodes <= self._max_episodes:
            self._episodes += 1

            state = self.env.reset()
            done = False
            warmup_steps = self.warmup_function(self._episodes)
            episode_reward = 0
            self._episode_steps = 0

            while not done:
                if self._episode_steps < warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.sample_regular_action(state)
                
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.store_replay(state, action, reward, next_state, done)

                self._steps += 1
                self._episode_steps += 1
                episode_reward += reward
                state = next_state

                if self._episode_steps >= warmup_steps:
                    for _ in range(self.gradient_steps):
                        self.train_policy()

            self._reward_log.append(episode_reward)
            self._writer.add_scalar("stats/reward", episode_reward, self.episodes)
            self.log(f"Episode Reward: {episode_reward}")

    def sample_evaluation_action(self, states: np.ndarray) -> np.ndarray:
        return np.array([])

    def sample_regular_action(self, states: np.ndarray) -> np.ndarray:
        return np.array([])

    def train_policy(self) -> None:
        pass
