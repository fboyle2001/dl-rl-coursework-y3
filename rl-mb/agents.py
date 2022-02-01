from typing import Any, Dict, Optional, Union, Callable, Tuple
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
import fbbuffer

class RLAgent(abc.ABC):
    def __init__(self, agent_name: str, env_name: str, device: Union[str, torch.device], video_every: Optional[int],
                    max_episodes: int = 50000, log_to_file: bool = True, log_to_screen: bool = True, seed: int = 42):
        self._agent_name = agent_name
        self.env = gym.make(env_name)
        self._action_dim = self.env.action_space.shape[0] # type: ignore
        self._state_dim = self.env.observation_space.shape[0] # type: ignore
        self._max_action = float(self.env.action_space.high[0]) # type: ignore

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
        
        self.log(f"Action dim: {self._action_dim}, State dim: {self._state_dim}, Max Action: {self._max_action}")

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
                    min_warmup_steps: int = 25000, warmup_function: Callable[[int], int] = lambda step: 0):
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

        while self._episodes <= self._max_episodes:
            if self._episodes % 10 == 0 and self._episodes != 0:
                avg_reward = 0

                for _ in range(10):
                    state = self.env.reset()
                    done = False

                    while not done:
                        action = self.sample_evaluation_action(torch.tensor(state, device=self.device))
                        state, reward, done, _ = self.env.step(action)
                        avg_reward += reward
                
                avg_reward = avg_reward / 10

                self.log(f"EVALUATION: Evaluated over 10 episodes, {avg_reward:.3f}")
                self._writer.add_scalar("stats/eval_reward", avg_reward, self.steps)

            self._episodes += 1

            state = self.env.reset()
            done = False
            episode_reward = 0
            self._episode_steps = 0

            while not done:
                if self._steps < self.min_warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.sample_regular_action(torch.tensor(state, device=self.device))
                
                next_state, reward, done, _ = self.env.step(action)
                done = done if self._episode_steps < self.env.max_episode_steps else False # type: ignore
                self.replay_buffer.store_replay(state, action, reward, next_state, done)

                self._steps += 1
                self._episode_steps += 1
                episode_reward += reward
                state = next_state

                if self.steps >= self.min_warmup_steps:
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
            target_actions = self.target_actor(buffer_sample.next_states).clamp(-self._max_action, self._max_action)
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
        self.log_alpha = torch.tensor(0.2, requires_grad=True, device=self.device)

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
    
    @property
    def alpha(self):
        return self.log_alpha.exp()

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
            target_actions, log_probs = self.actor.compute(buffer_sample.next_states)
            target_input = torch.cat([buffer_sample.next_states, target_actions], dim=-1)

            # Calculate both critic Qs and then take the min
            target_Q1 = self.target_critic_1(target_input)
            target_Q2 = self.target_critic_2(target_input)
            min_Q_target = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_probs

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
                        warmup_steps: int = 10000, target_update_interval: int = 1, ensemble_size: int = 7,
                        rollouts_per_step: int = 100000, dynamics_train_freq: int = 250, model_samples_per_env_sample = 20,
                        retained_step_rollouts: int = 1):
        """
        MBPO with SAC Implementation
        """
        super().__init__("MBPO", env_name, device, video_every)

        self.replay_buffer = buffers.StandardReplayBuffer(self._state_dim, self._action_dim, buffer_size, self.device)
        self.model_buffer = buffers.StandardReplayBuffer(self._state_dim, self._action_dim, retained_step_rollouts * rollouts_per_step * dynamics_train_freq, self.device)

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
        self._writer.add_scalar("stats/critic_loss_2", critic_2_loss.detach().cpu().item(), self._steps)

        self.opt_critic_1.zero_grad()
        critic_1_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1) # type: ignore
        self.opt_critic_1.step()
        
        self.opt_critic_2.zero_grad()
        critic_2_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1) # type: ignore
        self.opt_critic_2.step()

        # Q functions updated, now update the policy, need:
        # * a'_t ~ policy(s_t) (NOT s_[t+1] like in Q updates)
        # * log(a'_t | s_t) from policy
        # * current alpha
        # Maps to Eq 7
        
        # if self.dynamics.absolute_average_loss > 10:
        #     actor_actions, actor_probs = self.actor.compute(buffer_sample.states)
        #     real_input = torch.cat([buffer_sample.states, actor_actions], dim=-1)
        # else:
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
        actor_loss = (self.alpha * actor_probs - min_Q).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1) # type: ignore
        self.opt_actor.step()

        # Actor updated, now update the temperature (alpha), need:
        # * a'_t ~ policy(s_t) (NOT s_[t+1] like in Q updates)
        # * log(a'_t | s_t) from policy
        # * target entropy H
        # * current alpha

        # No need to backprop on the target or log probabilities since
        # we want to isolate the alpha update here
        alpha_loss = -self.log_alpha * (actor_probs + self.target_entropy).detach()
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
                    self.log(f"Trained dynamics, losses: {self.dynamics.losses}, |Avg Loss|: {self.dynamics.absolute_average_loss}")

                    for i, loss in enumerate(self.dynamics.losses):
                        self._writer.add_scalar(f"dynamics/{i}", loss, self._steps)

                    # Perform rollouts on the new dynamics model
                    self.perform_rollouts()

                # Don't start training the policy until we have sampled the random steps
                # seems to break the dynamics model if this is on?? 
                # if self._steps < self.warmup_steps:
                #     continue

                if self.steps >= self.warmup_steps:
                    # Train the policy
                    g_steps = self.gradient_steps# if self.dynamics.absolute_average_loss <= 0.25 else 1

                    for _ in range(g_steps):
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

class ModelBasedTD3Agent(RLAgent):
    def __init__(self, env_name: str, device: Union[str, torch.device], video_every: Optional[int],
                    buffer_size: int = int(1e6), lr: float = 3e-4, noise_sigma: float = 0.2, 
                    tau: float = 0.005, replay_batch_size: int = 256, noise_clip: float = 0.5,
                    gamma: float = 0.99, policy_update_frequency: int = 2, exploration_noise: float = 0.1,
                    min_warmup_steps: int = 25000, warmup_function: Callable[[int], int] = lambda step: 0):
        """
        TD3 Implementation
        """
        super().__init__("StandardTD3", env_name, device, video_every)

        # Replay buffer
        self.replay_buffer = buffers.StandardReplayBuffer(self._state_dim, self._action_dim, buffer_size, self.device)
        self.model_buffer = buffers.StandardReplayBuffer(self._state_dim, self._action_dim, 1 * 400, self.device)

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

        # Dynamics modelling
        self.dynamics = networks.EnsembleGaussianDynamics(self._state_dim, self._action_dim, 7, self.device)

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

        while self._episodes <= self._max_episodes:
            if self._episodes % 10 == 0 and self._episodes != 0:
                avg_reward = 0

                for _ in range(10):
                    state = self.env.reset()
                    done = False

                    while not done:
                        action = self.sample_evaluation_action(torch.tensor(state, device=self.device))
                        state, reward, done, _ = self.env.step(action)
                        avg_reward += reward
                
                avg_reward = avg_reward / 10

                self.log(f"EVALUATION: Evaluated over 10 episodes, {avg_reward:.3f}")
                self._writer.add_scalar("stats/eval_reward", avg_reward, self.steps)

            self._episodes += 1

            state = self.env.reset()
            done = False
            episode_reward = 0
            self._episode_steps = 0

            while not done:
                if self._steps < self.min_warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.sample_regular_action(torch.tensor(state, device=self.device))
                
                next_state, reward, done, _ = self.env.step(action)
                done = done if self._episode_steps < self.env.max_episode_steps else False # type: ignore
                self.replay_buffer.store_replay(state, action, reward, next_state, done)

                if self.steps % 250:
                    total_rollouts = 250 * 400

                self._steps += 1
                self._episode_steps += 1
                episode_reward += reward
                state = next_state

                if self.steps >= self.min_warmup_steps:
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
            target_actions = self.target_actor(buffer_sample.next_states).clamp(-self._max_action, self._max_action)
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

class SVGSACAgentOld(RLAgent):

    def __init__(self, env_name: str, device: Union[str, torch.device], video_every: Optional[int],
                    buffer_size: int = int(1e6), actor_critic_lr: float = 1e-4, alpha_lr: float = 5e-4,
                    alpha_init: float = 0.1, tau: float = 5e-3, critic_target_update_freq: int = 1,
                    actor_update_freq: int = 1, gamma: float = 0.99, single_step_updates: int = 1, 
                    single_step_batch_size: int = 512, actor_log_std_lower_bound: float = -5,
                    actor_log_std_upper_bound: float = 2, world_model_lr: float = 1e-3,
                    multi_step_updates_n_seq: int = 4, multi_step_batch_size: int = 512,
                    horizon: int = 3, clip_grad: float = 1.0, dynamics_update_freq: int = 1,
                    dynamics_update_repeat: int = 4, model_free_update_repeat: int = 4, 
                    warmup_steps: int = 0):
        """
        SVG(H) SAC Implementation
        Other:
            actor_detach_rho = False
            actor_dx_threshold = None
            actor_mve = True

            critic_target_mve = False
            full_target_mve = False


        """
        super().__init__("SVGSAC", env_name, device, video_every)

        # self.replay_buffer = buffers.StandardReplayBuffer(self._state_dim, self._action_dim, buffer_size, self.device)
        self.replay_buffer = fbbuffer.ReplayBuffer(self._state_dim, self._action_dim, buffer_size, self.device)

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
        self.log_alpha = torch.tensor([np.log(alpha_init)], requires_grad=True, device=self.device)

        # World model
        self.dynamics_model = networks.GRUDynamicsModel(self._state_dim + self._action_dim, self._state_dim, horizon,
                                                        self.device, clip_grad=clip_grad, lr=world_model_lr).to(self.device)
        self.termination_model = networks.TerminationModel(self._state_dim + self._action_dim, 1).to(self.device)
        self.reward_model = networks.RewardModel(self._state_dim + self._action_dim, 1).to(self.device)

        # Optimisers
        self.opt_critic_1 = torch.optim.Adam(self.critic_1.parameters(), lr=actor_critic_lr)
        self.opt_critic_2 = torch.optim.Adam(self.critic_2.parameters(), lr=actor_critic_lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_critic_lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.opt_term = torch.optim.Adam(self.termination_model.parameters(), lr=world_model_lr)
        self.opt_reward = torch.optim.Adam(self.reward_model.parameters(), lr=world_model_lr)

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.critic_target_update_freq = critic_target_update_freq
        self.actor_update_freq = actor_update_freq
        self.single_step_updates = single_step_updates
        self.single_step_batch_size = single_step_batch_size
        self.multi_step_updates_n_seq = multi_step_updates_n_seq
        self.multi_step_batch_size = multi_step_batch_size
        self.horizon = horizon
        self.clip_grad = clip_grad
        self.dynamics_update_freq = dynamics_update_freq
        self.dynamics_update_repeat = dynamics_update_repeat
        self.model_free_update_repeat = model_free_update_repeat
        self.warmup_steps = warmup_steps
        self.discount_horizon = torch.tensor([self.gamma**i for i in range(horizon)]).to(device)
    
    @property
    def alpha(self):
        return self.log_alpha.exp()

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

    def _train_reward_model(self, states, actions, expected_rewards) -> None:
        joined = torch.cat([states, actions], dim=-1)
        rm = self.reward_model(joined)
        # print(rm.shape, expected_rewards.shape)
        loss = F.mse_loss(rm, expected_rewards)
        self._writer.add_scalar("loss/reward", loss.detach().item(), self._steps)

        self.opt_reward.zero_grad()
        loss.backward()
        self.opt_reward.step()

    def _train_termination_model(self, states, actions, expected_termination) -> None:
        joined = torch.cat([states, actions], dim=-1)
        expected_termination = float(1) - expected_termination
        n_done = torch.sum(expected_termination)

        if n_done > 0:
            pos_weight = (states.shape[0] - n_done) / n_done
        else:
            pos_weight = torch.tensor(1.) 

        forward = self.termination_model(joined)
        loss = F.binary_cross_entropy_with_logits(forward, expected_termination, pos_weight=pos_weight)
        self._writer.add_scalar("loss/termination", loss.detach().item(), self._steps)

        self.opt_term.zero_grad()
        loss.backward()
        self.opt_term.step()

    def train_policy(self) -> None:
        # Need to fill the buffer before we can do anything
        if len(self.replay_buffer) < self.single_step_batch_size:
            return

        if self.steps % self.dynamics_update_freq == 0 and len(self.replay_buffer) > self.multi_step_batch_size:
            for _ in range(self.dynamics_update_repeat):
                states, actions, rewards = self.replay_buffer.sample_multistep(self.multi_step_batch_size, self.horizon)
                dynamics_loss = self.dynamics_model.train(states, actions, rewards)
                self._writer.add_scalar("loss/dynamics", dynamics_loss, self._steps)

        n_updates = 1 if self.steps < self.warmup_steps else self.model_free_update_repeat

        for _ in range(n_updates):
            # Firstly update the Q functions (the critics)
            states, actions, raw_rewards, next_states, not_dones, not_dones_no_max = self.replay_buffer.sample(self.single_step_batch_size)

            # Q Target
            # For this we need:
            # * (s_t, a_t, r_t) ~ buffer
            # * min{1,2}(Q(s_t, a_t))
            # * a'_t ~ policy(s_[t+1])
            # This maps to Eq 3 and Eq 5 from Haarnoja et al. 2018

            # Parameters of the target critics are frozen so don't update them!
            with torch.no_grad():
                target_actions = self.actor.compute_actions(next_states)
                target_input = torch.cat([next_states, target_actions], dim=-1)

                # Calculate both critic Qs and then take the min
                target_Q1 = self.target_critic_1(target_input)
                target_Q2 = self.target_critic_2(target_input)
                min_Q_target = torch.min(target_Q1, target_Q2) # should be minusing a term related to the entropy and log probabilities ?? 

                # Compute r(s_t, a_t) + gamma * E_{s_[t+1] ~ p}(V_target(s_[t+1]))
                target_Q = raw_rewards + not_dones_no_max * self.gamma * min_Q_target # may need unsqueezing
                target_Q = target_Q.detach() # New?
                
                # Log for TB
                self._writer.add_scalar("stats/target_q", target_Q.detach().cpu().mean().item(), self._steps)

            # Now calculate the MSE for the critics and update parameters
            critic_input = torch.cat([states, actions], dim=-1)

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

            if self.steps % self.actor_update_freq == 0:
                model_free_update = self.steps < self.warmup_steps

                if model_free_update:
                    # Q functions updated, now update the policy, need:
                    # * a'_t ~ policy(s_t) (NOT s_[t+1] like in Q updates)
                    # * log(a'_t | s_t) from policy
                    # * current alpha
                    # Maps to Eq 7

                    actor_actions, actor_probs = self.actor.compute(states)
                    real_input = torch.cat([states, actor_actions], dim=-1)

                    # See what the critics think to the actor's prediction of the next action
                    real_Q1 = self.critic_1(real_input)
                    real_Q2 = self.critic_2(real_input)
                    min_Q = torch.min(real_Q1, real_Q2)

                    self._writer.add_scalar('loss/mean_Q1', real_Q1.detach().mean().item(), self._steps)
                    self._writer.add_scalar('loss/mean_Q2', real_Q2.detach().mean().item(), self._steps)

                    # See Eq 7 
                    actor_loss = (self.alpha.detach() * actor_probs - min_Q).mean()
                else:
                    # print(states.shape)
                    predicted_states, policy_actions, actor_probs = self.dynamics_model.unroll_horizon(states, self.actor)
                    # print(predicted_states.shape)
                    all_states = torch.cat([predicted_states, states.unsqueeze(0)], dim=0)
                    joined = torch.cat([all_states, policy_actions], dim=2)
                    dones = self.termination_model(joined).sigmoid().squeeze(dim=2)
                    not_dones = 1. - dones
                    accum = [not_dones[0]]

                    for i in range(not_dones.size(0) - 1):
                        accum.append(accum[-1] * not_dones[i])
                    
                    accum = torch.stack(accum, dim=0)
                    last_not_dones = not_dones[-1]
                    rewards = not_dones * self.reward_model(joined).squeeze(2)

                    # q1 = self.critic_1(torch.cat([all_states[-1], policy_actions[-1]], dim=-1))
                    # q2 = self.critic_2(torch.cat([all_states[-1], policy_actions[-1]], dim=-1))
                    # q = torch.min(q1, q2).reshape(predicted_states.shape[0])
                    # rewards[-1] = last_not_dones * q
                    rewards -= self.alpha.detach() * actor_probs
                    rewards *= self.discount_horizon.unsqueeze(1)
                    total_rewards = rewards.sum()

                    actor_loss = -(total_rewards / self.horizon).mean()
                
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
                alpha_loss = (-self.alpha * (actor_probs + self.target_entropy).detach()).mean()
                self._writer.add_scalar("loss/alpha", alpha_loss.detach().item(), self._steps)

                self.opt_alpha.zero_grad()
                alpha_loss.backward() # Could enforce some requires on the grad or decay the alpha target entropy?
                self.opt_alpha.step()

                self._writer.add_scalar('stats/alpha', self.alpha.detach().item(), self._steps)

                self._train_reward_model(states, actions, raw_rewards)
                self._train_termination_model(states, actions, not_dones_no_max) # Does this need to be 1 - terminals?

                if self.steps % self.critic_target_update_freq == 0:
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
            
                self.train_policy()
                
                next_state, reward, done, _ = self.env.step(action)
                self._steps += 1
                self._episode_steps += 1
                episode_reward += reward

                store_done = float(done) if self._episode_steps + 1 < self.env.spec.max_episode_steps else 0. # type: ignore

                self.replay_buffer.add(state, action, reward, next_state, float(done), store_done)
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

class SVGSACSecond(RLAgent):
    def __init__(self, env_name: str, device: Union[str, torch.device], video_every: Optional[int],
                        buffer_size: int = int(1e6), lr: float = 0.0003, tau: float = 0.005,
                        replay_batch_size: int = 256, gamma: float = 0.99, gradient_steps: int = 1,
                        warmup_steps: int = 2000, target_update_interval: int = 1):
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
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=self.device)

        # Optimisers
        self.opt_critic_1 = torch.optim.Adam(self.critic_1.parameters(), lr=1e-4)
        self.opt_critic_2 = torch.optim.Adam(self.critic_2.parameters(), lr=1e-4)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=5e-4)
        
        # World Model
        self.dynamics_model = networks.GRUDynamicsModel(self._state_dim + self._action_dim, self._state_dim, 3, self.device, clip_grad=1., lr=1e-3)
        self.termination_model = networks.TerminationModel(self._state_dim + self._action_dim, 1)
        self.reward_model = networks.RewardModel(self._state_dim + self._action_dim, 1)
        self.opt_term = torch.optim.Adam(self.termination_model.parameters(), lr=1e-3)
        self.opt_reward = torch.optim.Adam(self.reward_model.parameters(), lr=1e-3)

        # Hyperparameters
        self.M_step = 1
        self.M_seq = 4
        self.tau = 0.005
        self.H = 3
        self.gamma = 0.99

        self.regular_replay_batch_size = 512
        self.multi_replay_batch_size = 1024
        self.warmup_steps = 0
    
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _update_rewards(self):
        pass

    def _update_terminations(self):
        pass

    def _update_dynamics(self):
        pass
    
    def _update_critics(self, states, actions, next_states, rewards, terminals):
        # Firstly update the Q functions (the critics)

        # Q Target
        # For this we need:
        # * (s_t, a_t, r_t) ~ buffer
        # * min{1,2}(Q(s_t, a_t))
        # * a'_t ~ policy(s_[t+1])
        # This maps to Eq 3 and Eq 5 from Haarnoja et al. 2018

        # Parameters of the target critics are frozen so don't update them!
        with torch.no_grad():
            target_actions = self.actor.compute_actions(next_states)
            target_input = torch.cat([next_states, target_actions], dim=-1)

            # Calculate both critic Qs and then take the min
            target_Q1 = self.target_critic_1(target_input)
            target_Q2 = self.target_critic_2(target_input)
            min_Q_target = torch.min(target_Q1, target_Q2)

            # Compute r(s_t, a_t) + gamma * E_{s_[t+1] ~ p}(V_target(s_[t+1]))
            target_Q = rewards.unsqueeze(1) + terminals.unsqueeze(1) * self.gamma * min_Q_target 
            
            # Log for TB
            self._writer.add_scalar("stats/target_q", target_Q.detach().cpu().mean().item(), self._steps)

        # Now calculate the MSE for the critics and update parameters
        critic_input = torch.cat([states, actions], dim=-1)

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

    def _model_free_actor_alpha(self, states):
        # Q functions updated, now update the policy, need:
        # * a'_t ~ policy(s_t) (NOT s_[t+1] like in Q updates)
        # * log(a'_t | s_t) from policy
        # * current alpha
        # Maps to Eq 7

        actor_actions, actor_probs = self.actor.compute(states)
        real_input = torch.cat([states, actor_actions], dim=-1)

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

        self._writer.add_scalar('stats/alpha', self.alpha.detach().item(), self._steps)

    def _update_actor_and_alpha(self, states):
        if self.steps < self.warmup_steps:
            self._model_free_actor_alpha(states)
            return
        
        """
        Otherwise update according to:

        """


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
        if self.replay_buffer.count < self.regular_replay_batch_size:
            return

        for _ in range(self.M_step):
            # Sample single steps
            b = self.replay_buffer.sample_buffer(self.regular_replay_batch_size)

            # 1. Fit the actor
            # 2. Fit alpha
            self._update_actor_and_alpha(b.states)

            # 3. Fit the critics
            self._update_critics(b.states, b.actions, b.next_states, b.rewards, b.terminals)

            # 4. Fit the rewards
            self._update_rewards()

            # 5. Fit the termination
            self._update_terminations()

            # 6. Update targets
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

AGENT_MAP = {
    "MBPO": StandardMBPOAgent,
    "SAC": StandardSACAgent,
    "TD3": StandardTD3Agent,
    "TD3Model": ModelBasedTD3Agent,
    "SVGSAC": SVGSACAgentOld
}