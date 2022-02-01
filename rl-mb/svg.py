from cmath import log
from ntpath import join
from operator import pos
import re
from turtle import done
from typing import Union, Optional

from agents import RLAgent
import buffers
import networks

import torch
import copy
import numpy as np
import torch.nn.functional as F

## STOLEN FUNCTIONS
def accum_prod(x):
    assert x.dim() == 2
    x_accum = [x[0]]
    for i in range(x.size(0)-1):
        x_accum.append(x_accum[-1]*x[i])
    x_accum = torch.stack(x_accum, dim=0)
    return x_accum



class SACSVGAgent(RLAgent):
    def __init__(self, env_name: str, device: Union[str, torch.device], video_every: Optional[int]):
        super().__init__("SACSVG", env_name, device, video_every)

        self.replay_buffer = buffers.MultiStepReplayBuffer(self._state_dim, self._action_dim, int(1e6), self.device)

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

        # World Model
        self.dynamics = networks.GRUDynamicsModel(self._state_dim + self._action_dim, self._state_dim, horizon=3,
                                     device=self.device, clip_grad=1.0, lr=1e-3).to(self.device)
        self.termination_model = networks.TerminationModel(self._state_dim + self._action_dim, 1).to(self.device)
        self.reward_model = networks.RewardModel(self._state_dim + self._action_dim, 1).to(self.device)

        # Optimisers
        self.opt_critic_1 = torch.optim.Adam(self.critic_1.parameters(), lr=1e-4)
        self.opt_critic_2 = torch.optim.Adam(self.critic_2.parameters(), lr=1e-4)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=1e-4)

        self.opt_rewards = torch.optim.Adam(self.reward_model.parameters(), lr=1e-3)
        self.opt_terminations = torch.optim.Adam(self.termination_model.parameters(), lr=1e-3)
        
        # Hyperparameters
        self.tau = 0.005
        self.M_steps = 1
        self.M_seqs = 4
        self.single_step_batch_size = 512
        self.horizon = 3
        self.gamma = 0.99
        self.gamma_horizon = torch.tensor([self.gamma ** i for i in range(self.horizon)]).to(device)
        self.multi_step_batch_size = 1024

        self.warmup_steps = 5000

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

    def update_target_parameters(self):
        # Update frozen targets, taken from https://github.com/sfujim/TD3/blob/385b33ac7de4767bab17eb02ade4a268d3e4e24f/TD3.py
        # theta' = tau * theta + (1 - tau) * theta'
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_actor_and_alpha(self, states):
        if self._steps < self.warmup_steps or True:
            actor_actions, log_probs = self.actor.compute(states)
            real_input = torch.cat([states, actor_actions], dim=-1)

            # See what the critics think to the actor's prediction of the next action
            real_Q1 = self.critic_1(real_input)
            real_Q2 = self.critic_2(real_input)
            min_Q = torch.min(real_Q1, real_Q2)

            actor_loss = (self.alpha.detach() * log_probs - min_Q).mean()
        else:
            # Unroll from the dynamics using the initial states
            predicted_states, policy_actions, log_probs = self.dynamics.unroll_horizon(states, self.actor)
            
            # Now we need the rewards at each state, quickest way is to join and pass through reward model
            # Note need to unsqueeze states since predicted_states has format [timestep, batch, ...]
            # whereas states is [batch, ...] so transform to [1, batch, ...] for concat
            all_states = torch.cat([states.unsqueeze(0), predicted_states], dim=0)

            #print(all_states.shape, predicted_states.shape, policy_actions.shape)
            assert all_states.shape[0] == policy_actions.shape[0]

            # r(x, u) so join the actions too on the final dimension
            joined_state_actions = torch.cat([all_states, policy_actions], dim=-1)

            # compute r(x, u) 
            rewards = self.reward_model(joined_state_actions)
            assert rewards.shape[0] == joined_state_actions.shape[0] and rewards.shape[1] == joined_state_actions.shape[1]

            # need the terminal indicators too, they need to be in the range [0, 1]
            done = self.termination_model(joined_state_actions).sigmoid()
            assert done.shape == rewards.shape

            # both terminal and rewards are [timesteps, batch, 1] dimensions so squeeze them
            # makes them [timesteps, batch] dimensions instead
            done = done.squeeze(dim=2)
            rewards = rewards.squeeze(dim=2)

            # TODO: What is the accumulation for?
            not_dones = accum_prod(1. - done)

            # Scale the rewards by the probability of accumlated termination
            rewards = not_dones * rewards

            # V~(x_H) takes the form of a simple terminal reward function
            # Recall that Q^[pi](x, u) - r(x, u) = V^[pi](x, u) (simplified notation here)
            # See Eq 1 from the paper
            final_obs = torch.cat([all_states[-1], policy_actions[-1]], dim=-1)
            q1 = self.critic_1(final_obs)
            q2 = self.critic_2(final_obs)
            q = torch.min(q1, q2).reshape(states.shape[0])
            rewards[-1] = not_dones[-1] * q
            
            assert rewards.shape == log_probs.shape

            # See J_(pi, alpha)^[SVG](D_s) this is the internal expectation expression
            # actually its the first expectation fully as states ~ D_s as input
            rewards -= self.alpha.detach() * log_probs
            
            # Discount rewards the further in time we go
            rewards *= self.gamma_horizon.unsqueeze(1)

            # Take the average reward over each timestep
            # [timesteps, batch] -> [batch] dimensions
            rewards = rewards.sum(dim=0)

            # Calculate the loss
            # TODO: Where does this actually come from?
            actor_loss = -(rewards / self.horizon).mean()

        self._writer.add_scalar("loss/actor", actor_loss.detach().cpu(), self._steps)
        self._writer.add_scalar("stats/entropy", -log_probs[0].detach().mean(), self._steps)

        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # Now update alpha (do it here since we have everything we need so it's efficient)
        # Optimise over the first timestep i.e. those sampled directly from D only
        # Remember ~= expectation so take the mean
        alpha_loss = (-self.alpha * (log_probs[0] * self.target_entropy).detach()).mean()

        self._writer.add_scalar("loss/alpha", alpha_loss.detach().cpu(), self._steps)

        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        self._writer.add_scalar("stats/alpha", self.alpha.detach().cpu(), self._steps)

    def update_critics(self, states, actions, rewards, next_states, not_dones):
        # Compute Q^(target)_theta(x_t, u_t)
        with torch.no_grad():
            # V_theta_frozen(x_[t+1])
            target_actions, log_probs = self.actor.compute(next_states)
            target_input = torch.cat([next_states, target_actions], dim=-1)

            # Calculate both critic Qs and then take the min
            target_Q1 = self.target_critic_1(target_input)
            target_Q2 = self.target_critic_2(target_input)
            min_Q_target = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_probs

            # Compute r(s_t, a_t) + gamma * E_{s_[t+1] ~ p}(V_target(s_[t+1]))
            # TODO: Maybe see what happens when we remove the unsqueeze?
            target_Q = rewards.unsqueeze(1) + not_dones.unsqueeze(1) * self.gamma * min_Q_target 
            target_Q = target_Q.detach()
            self._writer.add_scalar("stats/target_q", target_Q.detach().cpu().mean().item(), self._steps)

        # Now compute Q_theta(x_t, u_t)
        critic_input = torch.cat([states, actions], dim=-1)

        actual_Q1 = self.critic_1(critic_input)
        self._writer.add_scalar("stats/critic_1", actual_Q1.detach().cpu().mean().item(), self._steps)
        
        actual_Q2 = self.critic_2(critic_input)
        self._writer.add_scalar("stats/critic_2", actual_Q2.detach().cpu().mean().item(), self._steps)

        # Take the losses as MSE loss since we are taking the expectation
        critic_1_loss = F.mse_loss(actual_Q1, target_Q)
        critic_2_loss = F.mse_loss(actual_Q2, target_Q)

        self._writer.add_scalar("loss/q1", critic_1_loss.detach().cpu().item(), self._steps)
        self._writer.add_scalar("loss/q2", critic_2_loss.detach().cpu().item(), self._steps)

        # Optimise
        self.opt_critic_1.zero_grad()
        critic_1_loss.backward()
        self.opt_critic_1.step()
        
        self.opt_critic_2.zero_grad()
        critic_2_loss.backward()
        self.opt_critic_2.step()

    def update_rewards(self, states, actions, rewards):
        # Simple to train, just MSE between predicted and real
        states_actions = torch.cat([states, actions], dim=-1)
        predicted_rewards = self.reward_model(states_actions)
        rewards = rewards.unsqueeze(1)

        assert predicted_rewards.shape == rewards.shape

        loss = F.mse_loss(predicted_rewards, rewards)
        self._writer.add_scalar("loss/rewards_model", loss.detach().item(), self._steps)

        self.opt_rewards.zero_grad()
        loss.backward()
        self.opt_rewards.step()

    def update_terminations(self, states, actions, not_dones):
        dones = (1. - not_dones).unsqueeze(1)
        states_actions = torch.cat([states, actions], dim=-1)
        predicted = self.termination_model(states_actions)

        # Paper said that reweighting was unnecessary but they implement it anyway
        done_count = torch.sum(dones)

        if done_count > 0:
            pos_weight = (states.shape[0] - done_count) / done_count
        else:
            pos_weight = torch.tensor(1.)

        assert predicted.shape == dones.shape

        # TODO: Understand how this matches up with the equation given
        loss = F.binary_cross_entropy_with_logits(predicted, dones, pos_weight=pos_weight)

        self.opt_terminations.zero_grad()
        loss.backward()
        self.opt_terminations.step()

    def train_policy(self) -> None:
        # Add some checks to make sure we have a large enough buffer
        # Probably should also add some stuff to handle model-free updates to start

        # Single-step updates 
        for _ in range(self.M_steps):
            # Sample N samples from the replay buffer
            states, actions, rewards, next_states, true_not_dones, not_dones = self.replay_buffer.sample(self.single_step_batch_size)
            
            # Update the actor first followed by alpha
            self.update_actor_and_alpha(states)

            # Now update the critics
            self.update_critics(states, actions, rewards, next_states, true_not_dones)

            # Next update the rewards model
            self.update_rewards(states, actions, rewards)

            # Then update the termination model
            self.update_terminations(states, actions, true_not_dones)

            # Finally in the single steps soft update the target's parameters
            self.update_target_parameters()

        if self._steps <= self.warmup_steps:
            return

        # Multi-step updates
        for _ in range(self.M_seqs):
            # Now we move on to updating the dynamics model
            # Start by sampling the multi-step trajectories
            states, actions, _ = self.replay_buffer.sample_sequences(self.multi_step_batch_size, self.horizon)
            dynamics_loss = self.dynamics.train(states, actions)
            self._writer.add_scalar("loss/dynamics", dynamics_loss, self._steps)

    def run(self) -> None:
        while self.episodes < self._max_episodes:
            # Start a new episode
            self._episodes += 1
            done = False
            state = self.env.reset()
            self._episode_steps = 0
            episode_reward = 0

            # Loop until we are finished with the episode
            while not done:
                if self._steps < self.warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.sample_regular_action(torch.tensor(state, device=self.device))
                
                next_state, reward, done, _ = self.env.step(action)
                true_done = done if self._episode_steps + 1 < self.env.spec.max_episode_steps else False # type: ignore
                self.replay_buffer.store(state, action, reward, next_state, true_done, done)

                # Train the agent
                self.train_policy()

                self._steps += 1
                self._episode_steps += 1
                episode_reward += reward
                state = next_state

            self._writer.add_scalar("stats/training_reward", episode_reward, self.episodes)
            self.log(f"Episode reward: {episode_reward}")

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

            