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
    def sample_evaluation_action(self, states: torch.Tensor) -> np.ndarray:
        """
        Sample an action for use in evaluation of the policy
        e.g. without any exploration noise added
        """
        pass

    @abc.abstractmethod
    def sample_regular_action(self, states: torch.Tensor) -> np.ndarray:
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
    
    def sample_evaluation_action(self, states: torch.Tensor) -> np.ndarray:
        states = states.reshape(1, -1)
        return self.actor(states).cpu().data.numpy().flatten()

    def sample_regular_action(self, states: torch.Tensor) -> np.ndarray:
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

class StandardSACAgent(RLAgent):
    def __init__(self, env_name: str, device: Union[str, torch.device], video_every: Optional[int],
                        buffer_size: int = int(1e6), lr: float = 0.0003, tau: float = 0.005,
                        replay_batch_size: int = 256, gamma: float = 0.99, gradient_steps: int = 1,
                        warmup_steps: int = 10000, target_update_interval: int = 1):
        """
        SAC Implementation
        """
        super().__init__("SACRewrite", env_name, device, video_every)

        self.replay_buffer = buffers.StandardReplayBuffer(self._state_dim, self._action_dim, buffer_size, self.device)

        # Critics predicts the reward of taking action A from state S
        self.critic_1 = networks.Critic(input_size=self._state_dim + self._action_dim, output_size=1).to(self.device)
        self.critic_2 = networks.Critic(input_size=self._state_dim + self._action_dim, output_size=1).to(self.device)

        # Actor predicts the action to take based on the current state
        # Though in SAC it actually predicts the distribution of actions
        self.actor = networks.GaussianActor(self._state_dim, self._action_dim, self.device)

        # Establish the target networks
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)

        # Defined in the paper as -dim(A)
        # This line is taken from elsewhere
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item() # type: ignore

        # Alpha can get very close to 0 so for the same reason as in
        # the GaussianActor we optimise ln(alpha) instead for numerical stability
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp() # Or 0.2?

        # Optimisers
        self.opt_critic_1 = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.opt_critic_2 = torch.optim.Adam(self.critic_2.parameters(), lr=lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
        
        # Hyperparameters
        self.gamma = gamma 
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.replay_batch_size = replay_batch_size
        self.warmup_steps = warmup_steps
        self.gradient_steps = gradient_steps
    
    def sample_evaluation_action(self, states: torch.Tensor) -> np.ndarray:
        states = states.unsqueeze(0)

        with torch.no_grad():
            action = self.actor.compute_actions(states, stochastic=False)

        return action.detach().cpu().numpy()[0]

    def sample_regular_action(self, states: torch.Tensor) -> np.ndarray:
        states = states.unsqueeze(0)

        with torch.no_grad():
            action = self.actor.compute_actions(states, stochastic=True)

        return action.detach().cpu().numpy()[0]

    def train_policy(self) -> None:
        # Need to fill the buffer before we can do anything
        if self.replay_buffer.count < self.replay_batch_size:
            return

        # Firstly update the Q functions (the critics)
        buffer_sample = self.replay_buffer.sample_buffer(self.replay_batch_size)

        # Q Target
        # For this we need:
        # * (s_t, a_t, r_t) ~ buffer
        # * min{1,2}(Q(s_t, a_t))
        # * a'_t ~ policy(s_[t+1])
        # This maps to Eq 3 and Eq 5 from Haarnoja et al. 2018

        # Parameters of the target critics are frozen so don't update them!
        with torch.no_grad():
            target_actions = self.actor.compute_actions(buffer_sample.next_states)
            target_input = torch.cat([buffer_sample.next_states, target_actions], dim=-1)

            # Calculate both critic Qs and then take the min
            target_Q1 = self.target_critic_1(target_input)
            target_Q2 = self.target_critic_2(target_input)
            min_Q_target = torch.min(target_Q1, target_Q2)

            # Compute r(s_t, a_t) + gamma * E_{s_[t+1] ~ p}(V_target(s_[t+1]))
            target_Q = buffer_sample.rewards.unsqueeze(1) + buffer_sample.terminals.unsqueeze(1) * self.gamma * min_Q_target 
            
            # Log for TB
            self._writer.add_scalar("stats/target_q", target_Q.detach().cpu().mean().item(), self._steps)

        # Now calculate the MSE for the critics and update parameters
        critic_input = torch.cat([buffer_sample.states, buffer_sample.actions], dim=-1)

        actual_Q1 = self.critic_1(critic_input)
        self._writer.add_scalar("stats/critic_1", actual_Q1.detach().cpu().mean().item(), self._steps)
        actual_Q2 = self.critic_2(critic_input)
        self._writer.add_scalar("stats/critic_2", actual_Q2.detach().cpu().mean().item(), self._steps)

        # Eq 5
        critic_1_loss = 0.5 * F.mse_loss(actual_Q1, target_Q)
        critic_2_loss = 0.5 * F.mse_loss(actual_Q2, target_Q)

        self._writer.add_scalar("stats/critic_loss_1", critic_1_loss.detach().cpu().item(), self._steps)
        self._writer.add_scalar("stats/critic_loss_2", critic_1_loss.detach().cpu().item(), self._steps)

        self.opt_critic_1.zero_grad()
        critic_1_loss.backward()
        self.opt_critic_1.step()
        
        self.opt_critic_2.zero_grad()
        critic_2_loss.backward()
        self.opt_critic_2.step()

        # Q functions updated, now update the policy, need:
        # * a'_t ~ policy(s_t) (NOT s_[t+1] like in Q updates)
        # * log(a'_t | s_t) from policy
        # * current alpha
        # Maps to Eq 7

        actor_actions, actor_probs = self.actor.compute(buffer_sample.states)
        real_input = torch.cat([buffer_sample.states, actor_actions], dim=-1)

        # See what the critics think to the actor's prediction of the next action
        real_Q1 = self.critic_1(real_input)
        real_Q2 = self.critic_2(real_input)
        min_Q = torch.min(real_Q1, real_Q2)

        self._writer.add_scalar('loss/mean_Q1', real_Q1.detach().mean().item(), self._steps)
        self._writer.add_scalar('loss/mean_Q2', real_Q2.detach().mean().item(), self._steps)

        # See Eq 7 
        actor_loss = (self.alpha * actor_probs - min_Q).mean()
        self._writer.add_scalar('loss/actor', actor_loss.detach().item(), self._steps)

        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # Actor updated, now update the temperature (alpha), need:
        # * a'_t ~ policy(s_t) (NOT s_[t+1] like in Q updates)
        # * log(a'_t | s_t) from policy
        # * target entropy H
        # * current alpha

        # No need to backprop on the target or log probabilities since
        # we want to isolate the alpha update here
        alpha_loss = -self.log_alpha.exp() * (actor_probs + self.target_entropy).detach()
        alpha_loss = alpha_loss.mean()
        self._writer.add_scalar("loss/alpha", alpha_loss.detach().item(), self._steps)

        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # Make sure to update alpha now we've updated log_alpha
        self.alpha = self.log_alpha.exp()

        self._writer.add_scalar('stats/alpha', self.alpha.detach().item(), self._steps)

        # Update the frozen critics
        self.update_target_parameters()

    def update_target_parameters(self):
        # Update frozen targets, taken from https://github.com/sfujim/TD3/blob/385b33ac7de4767bab17eb02ade4a268d3e4e24f/TD3.py
        # theta' = tau * theta + (1 - tau) * theta'
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def run(self) -> None:
        while self._episodes < self._max_episodes:
            self._episodes += 1
            done = False
            state = self.env.reset()
            self._episode_steps = 0
            episode_reward = 0
            
            while not done: 
                if self._steps < self.warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.sample_regular_action(torch.tensor(state, device=self.device))
            
                for _ in range(self.gradient_steps):
                    self.train_policy()
                
                next_state, reward, done, _ = self.env.step(action)
                self._steps += 1
                self._episode_steps += 1
                episode_reward += reward

                self.replay_buffer.store_replay(state, action, reward, next_state, done)
                state = next_state 
            
            self._writer.add_scalar("stats/reward", episode_reward, self.episodes)
            self.log(f"Episode Reward: {episode_reward}")

            if self._episodes % 10 == 0:
                self.log("Evaluating...")
                eval_episodes = 10
                avg_reward = 0

                for _ in range(eval_episodes):
                    state = self.env.reset()
                    ep_reward = 0
                    done = False

                    while not done:
                        state = torch.tensor(state, device=self.device)
                        action = self.sample_evaluation_action(state)
                        next_state, reward, done, _ = self.env.step(action)
                        ep_reward += reward
                        state = next_state
                    
                    avg_reward += ep_reward
                
                avg_reward /= eval_episodes
                self._writer.add_scalar("stats/eval_reward", avg_reward, self.episodes)
                self.log(f"[EVAL] Average reward over {eval_episodes} was {avg_reward}")

class StandardMBPOAgent(RLAgent):
    def __init__(self, env_name: str, device: Union[str, torch.device], video_every: Optional[int],
                        buffer_size: int = int(1e6), lr: float = 0.0003, tau: float = 0.005,
                        replay_batch_size: int = 256, gamma: float = 0.99, gradient_steps: int = 20,
                        warmup_steps: int = 3000, target_update_interval: int = 1, ensemble_size: int = 7,
                        rollouts_per_step: int = 100000, dynamics_train_freq: int = 250, model_samples_per_env_sample = 20,
                        retained_step_rollouts: int = 1):
        """
        MBPO with SAC Implementation
        """
        super().__init__("MBPO", env_name, device, video_every)

        self.replay_buffer = buffers.StandardReplayBuffer(self._state_dim, self._action_dim, buffer_size, self.device)
        self.model_buffer = buffers.StandardReplayBuffer(self._state_dim, self._action_dim, retained_step_rollouts * rollouts_per_step, self.device)

        # Critics predicts the reward of taking action A from state S
        self.critic_1 = networks.Critic(input_size=self._state_dim + self._action_dim, output_size=1).to(self.device)
        self.critic_2 = networks.Critic(input_size=self._state_dim + self._action_dim, output_size=1).to(self.device)

        # Actor predicts the action to take based on the current state
        # Though in SAC it actually predicts the distribution of actions
        self.actor = networks.GaussianActor(self._state_dim, self._action_dim, self.device)

        # Establish the target networks
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)

        # Defined in the paper as -dim(A)
        # This line is taken from elsewhere
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item() # type: ignore

        # Alpha can get very close to 0 so for the same reason as in
        # the GaussianActor we optimise ln(alpha) instead for numerical stability
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp() # Or 0.2?

        # Optimisers
        self.opt_critic_1 = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.opt_critic_2 = torch.optim.Adam(self.critic_2.parameters(), lr=lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
        
        # Dynamics model
        self.dynamics = networks.EnsembleGaussianDynamics(self._state_dim, self._action_dim, ensemble_size, self.device, lr=lr)

        # Hyperparameters
        self.gamma = gamma 
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.replay_batch_size = replay_batch_size
        self.warmup_steps = warmup_steps
        self.gradient_steps = gradient_steps ######TODO:::NOT 1
        self.dynamics_train_freq = dynamics_train_freq
        self.rollouts_per_step = rollouts_per_step
        self.env_ratio = 1 / model_samples_per_env_sample
    
    def sample_evaluation_action(self, states: torch.Tensor) -> np.ndarray:
        states = states.unsqueeze(0)

        with torch.no_grad():
            action = self.actor.compute_actions(states, stochastic=False)

        return action.detach().cpu().numpy()[0]

    def sample_regular_action(self, states: torch.Tensor) -> np.ndarray:
        states = states.unsqueeze(0)

        with torch.no_grad():
            action = self.actor.compute_actions(states, stochastic=True)

        return action.detach().cpu().numpy()[0]

    def train_policy(self) -> None:
        # Need to fill the buffer before we can do anything
        if self.replay_buffer.count < self.replay_batch_size:
            return

        # Firstly update the Q functions (the critics)
        buffer_sample = self.replay_buffer.sample_buffer(self.replay_batch_size)

        # Q Target
        # For this we need:
        # * (s_t, a_t, r_t) ~ buffer
        # * min{1,2}(Q(s_t, a_t))
        # * a'_t ~ policy(s_[t+1])
        # This maps to Eq 3 and Eq 5 from Haarnoja et al. 2018

        # Parameters of the target critics are frozen so don't update them!
        with torch.no_grad():
            target_actions = self.actor.compute_actions(buffer_sample.next_states)
            target_input = torch.cat([buffer_sample.next_states, target_actions], dim=-1)

            # Calculate both critic Qs and then take the min
            target_Q1 = self.target_critic_1(target_input)
            target_Q2 = self.target_critic_2(target_input)
            min_Q_target = torch.min(target_Q1, target_Q2)

            # Compute r(s_t, a_t) + gamma * E_{s_[t+1] ~ p}(V_target(s_[t+1]))
            target_Q = buffer_sample.rewards.unsqueeze(1) + buffer_sample.terminals.unsqueeze(1) * self.gamma * min_Q_target 
            
            # Log for TB
            self._writer.add_scalar("stats/target_q", target_Q.detach().cpu().mean().item(), self._steps)

        # Now calculate the MSE for the critics and update parameters
        critic_input = torch.cat([buffer_sample.states, buffer_sample.actions], dim=-1)

        actual_Q1 = self.critic_1(critic_input)
        self._writer.add_scalar("stats/critic_1", actual_Q1.detach().cpu().mean().item(), self._steps)
        actual_Q2 = self.critic_2(critic_input)
        self._writer.add_scalar("stats/critic_2", actual_Q2.detach().cpu().mean().item(), self._steps)

        # Eq 5
        critic_1_loss = 0.5 * F.mse_loss(actual_Q1, target_Q)
        critic_2_loss = 0.5 * F.mse_loss(actual_Q2, target_Q)

        self._writer.add_scalar("stats/critic_loss_1", critic_1_loss.detach().cpu().item(), self._steps)
        self._writer.add_scalar("stats/critic_loss_2", critic_1_loss.detach().cpu().item(), self._steps)

        self.opt_critic_1.zero_grad()
        critic_1_loss.backward()
        self.opt_critic_1.step()
        
        self.opt_critic_2.zero_grad()
        critic_2_loss.backward()
        self.opt_critic_2.step()

        # Q functions updated, now update the policy, need:
        # * a'_t ~ policy(s_t) (NOT s_[t+1] like in Q updates)
        # * log(a'_t | s_t) from policy
        # * current alpha
        # Maps to Eq 7
                    
        real_sample_count = int(self.env_ratio * self.replay_batch_size)
        real_sample_indices = np.random.choice(self.replay_batch_size, real_sample_count)
        fake_sample_count = self.replay_batch_size - real_sample_count

        buffer_fake_samples = self.model_buffer.sample_buffer(fake_sample_count)

        mixed_states = torch.cat([buffer_sample.states[real_sample_indices], buffer_fake_samples.states], dim=0)

        actor_actions, actor_probs = self.actor.compute(mixed_states)
        real_input = torch.cat([mixed_states, actor_actions], dim=-1)

        # See what the critics think to the actor's prediction of the next action
        real_Q1 = self.critic_1(real_input)
        real_Q2 = self.critic_2(real_input)
        min_Q = torch.min(real_Q1, real_Q2)

        # See Eq 7 
        actor_loss = (self.alpha.detach() * actor_probs - min_Q).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # Actor updated, now update the temperature (alpha), need:
        # * a'_t ~ policy(s_t) (NOT s_[t+1] like in Q updates)
        # * log(a'_t | s_t) from policy
        # * target entropy H
        # * current alpha

        # No need to backprop on the target or log probabilities since
        # we want to isolate the alpha update here
        alpha_loss = -self.log_alpha.exp() * (actor_probs + self.target_entropy).detach()
        alpha_loss = alpha_loss.mean()
        self._writer.add_scalar("loss/alpha", alpha_loss.detach().item(), self._steps)

        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # Make sure to update alpha now we've updated log_alpha
        self.alpha = self.log_alpha.exp()

        self._writer.add_scalar('stats/alpha', self.alpha.detach().item(), self._steps)
        
        self._writer.add_scalar('loss/actor', actor_loss.detach().cpu().item(), self._steps)
        self._writer.add_scalar('loss/mean_Q1', real_Q1.detach().cpu().mean().item(), self._steps)
        self._writer.add_scalar('loss/mean_Q2', real_Q2.detach().cpu().mean().item(), self._steps)
        # Update the frozen critics
        self.update_target_parameters()

    def update_target_parameters(self):
        # Update frozen targets, taken from https://github.com/sfujim/TD3/blob/385b33ac7de4767bab17eb02ade4a268d3e4e24f/TD3.py
        # theta' = tau * theta + (1 - tau) * theta'
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def perform_rollouts(self):
        rollout_buffer = self.replay_buffer.sample_buffer(self.rollouts_per_step)
        predicted_actions = torch.tensor(self.sample_regular_action(rollout_buffer.states), device=self.device)
        rollout_next_states, rollout_next_rewards = self.dynamics.batch_predict(rollout_buffer.states, predicted_actions)
        self.model_buffer.store_replays(rollout_buffer.states.detach().cpu().numpy(), predicted_actions.detach().cpu().numpy(), rollout_next_rewards.detach().cpu().numpy(),
                                        rollout_next_states.detach().cpu().numpy(), np.zeros(shape=rollout_buffer.states.shape[0], dtype=bool))

    def run(self) -> None:
        while self._episodes < self._max_episodes:
            self._episodes += 1
            done = False
            state = self.env.reset()
            self._episode_steps = 0
            episode_reward = 0
            
            while not done:
                if self._steps < self.warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.sample_regular_action(torch.tensor(state, device=self.device))
            
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.store_replay(state, action, reward, next_state, done)
                
                self._steps += 1
                self._episode_steps += 1
                episode_reward += reward

                # Train the dynamics model
                if self._steps % self.dynamics_train_freq == 0:
                    buffer_sample = self.replay_buffer.get_all_replays()
                    self.log(f"Training dynamics on {buffer_sample.states.shape[0]} replays")
                    self.dynamics.train(buffer_sample.states, buffer_sample.actions, buffer_sample.next_states, buffer_sample.rewards.unsqueeze(1))
                    self.log(f"Trained dynamics, losses: {self.dynamics.losses}")

                    for i, loss in enumerate(self.dynamics.losses):
                        self._writer.add_scalar(f"dynamics/{i}", loss, self._steps)

                    # Perform rollouts on the new dynamics model
                    self.perform_rollouts()

                # Don't start training the policy until we have sampled the random steps
                # seems to break the dynamics model if this is on?? 
                # if self._steps < self.warmup_steps:
                #     continue

                # Train the policy
                for _ in range(self.gradient_steps):
                    self.train_policy()

                state = next_state 
            
            self._writer.add_scalar("stats/reward", episode_reward, self.episodes)
            self.log(f"Episode Reward: {episode_reward}")

            if self._episodes % 10 == 0:
                self.log("Evaluating...")
                eval_episodes = 10
                avg_reward = 0

                for _ in range(eval_episodes):
                    state = self.env.reset()
                    ep_reward = 0
                    done = False

                    while not done:
                        state = torch.tensor(state, device=self.device)
                        action = self.sample_evaluation_action(state)
                        next_state, reward, done, _ = self.env.step(action)
                        ep_reward += reward
                        state = next_state
                    
                    avg_reward += ep_reward
                
                avg_reward /= eval_episodes
                self._writer.add_scalar("stats/eval_reward", avg_reward, self.episodes)
                self.log(f"[EVAL] Average reward over {eval_episodes} was {avg_reward}")


AGENT_MAP = {
    "MBPO": StandardMBPOAgent,
    "SAC": StandardSACAgent,
    "TD3": StandardTD3Agent
}