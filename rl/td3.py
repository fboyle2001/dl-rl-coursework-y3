import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

# Copied from https://github.com/sfujim/TD3/blob/385b33ac7de4767bab17eb02ade4a268d3e4e24f/utils.py#L5
class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size, device):
        self.max_size = max_size
        self.index_ptr = 0
        self.size = 0

        self.state_mem = np.zeros((max_size, state_dim))
        self.action_mem = np.zeros((max_size, action_dim))
        self.next_state_mem = np.zeros((max_size, state_dim))
        self.reward_mem = np.zeros((max_size, 1))
        self.not_done_mem = np.zeros((max_size, 1))
        
        self.device = device

    def store(self, state, action, reward, new_state, done):
        self.state_mem[self.ptr] = state
        self.action_mem[self.ptr] = action
        self.next_state_mem[self.ptr] = next_state
        self.reward_mem[self.ptr] = reward
        self.not_done_mem[self.ptr] = 1. - done

        self.index_ptr = (self.index_ptr + 1) % self.max_size
        self.size = self.size + 1 if self.size < self.max_size else self.size

    def sample(self, batch_size):
        sample_indices = np.random.choice(self.size, size=batch_size)

        return (
            F.Tensor(self.state_mem[sample_indices], dtype=float).to(self.device),
            F.Tensor(self.action_mem[sample_indices], dtype=float).to(self.device),
            F.Tensor(self.next_state_mem[sample_indices], dtype=float).to(self.device),
            F.Tensor(self.reward_mem[sample_indices], dtype=float).to(self.device),
            F.Tensor(self.not_done_mem[sample_indices], dtype=float).to(self.device)
        )

class Actor(nn.Module):
    def __init__(self, input_size, output_size, connected_size=256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(input_size, connected_size)
        self.fc2 = nn.Linear(connected_size, connected_size)
        self.output = nn.Linear(connected_size, output_size)

    def forward(self, state):
        step = F.relu(self.fc1(state))
        step = F.relu(self.fc2(step))
        out = F.tanh(self.output(step))

        return out

class Critic(nn.Module):
    def __init__(self, input_size, output_size=1, connected_size=256):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(input_size, connected_size)
        self.fc2 = nn.Linear(connected_size, connected_size)
        self.output = nn.Linear(connected_size, output_size)

    def forward(self, state):
        step = F.relu(self.fc1(state))
        step = F.relu(self.fc2(step))
        out = self.output(step)

        return out

class Agent():
    def __init__(self, input_size, num_env_comps, device):
        self.device = device

        self.critic_1 = Critic(input_size=input_size + num_env_comps).to(device)
        self.critic_2 = Critic(input_size=input_size + num_env_comps).to(device)
        self.actor = Actor(input_size=input_size, output_size=num_env_comps).to(device)

        self.target_critic_1 = copy.deepcopy(critic_1).to(device)
        self.target_critic_2 = copy.deepcopy(critic_2).to(device)
        self.target_actor = copy.deepcopy(actor).to(device)

        # size of the replay buffer, authors use 1e6
        self.buffer_size = int(1e6) 
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # Hyperparameters
        # self.alpha = alpha
        # self.beta = beta

        # tau is used to update the target network parameters
        self.tau = 0.005
        # warmup_iters period for exploring at the start, 1k for unstable, 10k for stable
        self.warmup_iters = 1000
        # update the policy every policy_update_freq iterations
        self.policy_update_freq = 2
        # sample from N(0, noise_sigma) and clip it for SARSA-style updates
        self.noise_sigma = 0.2
        # clip the N(0, noise_sigma) into this range
        self.noise_clip = 0.5
        # mini batch size for sampling from replay buffer
        self.mini_batch_size = 256
        # discount factor for temporal difference learning
        self.gamma = 0.99
        # learning rate for actor and critic optimisers, 3e-4 is the same as authors
        self.learning_rate = 3e-4
        # exploration noise
        self.exploration_noise = 0.1

        self.iters = 0

        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=self.learning_rate)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=self.learning_rate)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

    def update_target_parameters(self):
        # Update frozen targets, taken from https://github.com/sfujim/TD3/blob/385b33ac7de4767bab17eb02ade4a268d3e4e24f/TD3.py
        # theta' = tau * theta + (1 - tau) * theta'
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_memory(self, state, action, reward, new_state, done):
        self.replay_buffer.store(state, action, reward, new_state, done)

    def sample_action(self, s):
        s = s.reshape(1, -1)
        state = F.Tensor(s, dtype=float).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self):
        self.iters += 1

        # Do I need to account for the warmup period here?

        # Need to fill the replay buffer first
        if self.replay_buffer.size < self.mini_batch_size:
            return

        state, action, next_state, reward, not_done = self.replay_buffer.sample(self.mini_batch_size)

        with torch.no_grad():
            clipped_noise = (torch.randn_like(action) * self.noise_sigma).clamp(-self.noise_clip, self.noise_clip)
            frozen_action = (self.target_actor(next_state) + clipped_noise).clamp(-1, 1)

            target_Q1 = self.target_critic_1(next_state, frozen_action)
            target_Q2 = self.target_critic_2(next_state, frozen_action)

            # TD learning
            target_Q = reward + not_done * self.gamma * torch.min(target_Q1, target_Q2)

        actual_Q1 = self.critic_1(state, action)
        actual_Q2 = self.critic_2(state, action)

        critic_loss = F.mse_loss(actual_Q1, target_Q) + F.mse_loss(actual_Q2, target_Q)

        self.critic_1_opt.zero_grad()
        self.critic_2_opt.zero_grad()

        critic_loss.backward()

        self.critic_1_opt.step()
        self.critic_2_opt.step()

        if self.iters % self.policy_update_freq == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.update_target_parameters()

