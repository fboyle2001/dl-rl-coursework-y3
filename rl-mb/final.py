from typing import Union, Optional
from agents import RLAgent
import torch
import buffers
import networks
import copy
import numpy as np
import torch.nn.functional as F

# Weighted Importance Sampling Mean Squared Error 
# Cite https://proceedings.neurips.cc/paper/2014/file/be53ee61104935234b174e62a07e53cf-Paper.pdf
def wis_mse(input, target, weights):
    return (weights * (target - input).square()).mean()

class FinalSACAgent(RLAgent):
    def __init__(self, env_name: str, device: Union[str, torch.device], video_every: Optional[int],
                        buffer_size: int = int(1e6), lr: float = 0.0003, tau: float = 0.005,
                        replay_batch_size: int = 256, gamma: float = 0.99, gradient_steps: int = 2,
                        warmup_steps: int = 10000, target_update_interval: int = 1):
        """
        SAC Implementation
        """
        super().__init__("SACRewrite", env_name, device, video_every)

        self.replay_buffer = buffers.PriorityReplayBuffer(self._state_dim, self._action_dim, 20, self.device)
        # self.replay_buffer = buffers.StandardReplayBuffer(self._state_dim, self._action_dim, buffer_size, self.device)

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

        # Dynamics
        self.horizon = 2
        self.real_ratio = 0.9
        self.rollouts_per_step = 256
        self.rollout_lifetime_steps = 100 
        self.dynamics = networks.GRUDynamicsModel(self._state_dim + self._action_dim, self._state_dim, self.horizon, self.device, 1.0).to(self.device)
        self.termination_model = networks.TerminationModel(self._state_dim + self._action_dim, 1).to(self.device)
        self.reward_model = networks.RewardModel(self._state_dim + self._action_dim, 1).to(self.device)
        self.fake_buffer = buffers.MultiStepReplayBuffer(self._state_dim, self._action_dim, self.horizon * self.rollout_lifetime_steps * self.rollouts_per_step, self.device)

        self.opt_rewards = torch.optim.Adam(self.reward_model.parameters(), lr=1e-3)
        self.opt_terminations = torch.optim.Adam(self.termination_model.parameters(), lr=1e-3)
        
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

    def rollout_model(self):
        if self.real_ratio == 1:
            return

        (states, _, nr, ns, nd), _, _ = self.replay_buffer.sample_buffer(self.rollouts_per_step * self.rollout_lifetime_steps)
        nd = 1 - nd # We want to know if it is done rather than not done

        print("Rolling out...")

        next_states, policy_actions, log_probs = self.dynamics.unroll_horizon(states, self.actor)
        start_states = states

        for i in range(self.horizon - 1):
            joined = torch.cat([start_states, policy_actions[i, :]], dim=-1)
            reward = self.reward_model(joined)
            done = self.termination_model(joined).sigmoid()

            done[torch.where(done >= 0.5)] = 1
            done[torch.where(done < 0.5)] = 0

            self.fake_buffer.store_replays(
                states.detach().cpu().numpy(),
                policy_actions[i, :].detach().cpu().numpy(),
                reward.detach().cpu().numpy(),
                next_states[i, :].detach().cpu().numpy(),
                done.detach().cpu().numpy()
            )

            # Write losses to TB
            if i == 0:
                # print(done.shape, nd.shape)
                # print(reward.shape, nr.shape)
                # print(next_states[0, :].shape, ns.shape)
                self._writer.add_scalar("rollout/dones", F.mse_loss(done.squeeze(1), nd), self.steps)
                self._writer.add_scalar("rollout/rewards", F.mse_loss(reward.squeeze(1), nr), self.steps)
                self._writer.add_scalar("rollout/next_states", F.mse_loss(next_states[0, :], ns), self.steps)

            start_states = next_states[i, :]
        
        # print("Rollout:", self.fake_buffer.count)

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
        (states, actions, rewards, next_states, not_terminals), buffer_indexes, buffer_weights  = self.replay_buffer.sample_buffer(self.replay_batch_size)

        self.update_rewards(states, actions, rewards)
        self.update_terminations(states, actions, not_terminals)

        # Mix in some fake samples
        if self.fake_buffer.count != 0:
            real_count = int(self.replay_batch_size * self.real_ratio)
            fake_count = self.replay_batch_size - real_count

            if self.steps % 1000 == 0:
                self.log(f"Sampling {fake_count} fakes")

            #print(f"Sampling {fake_count} fakes")

            #rstates, ractions, rrewards, rnext_states, rnot_dones, rnot_dones_no_max = self.replay_buffer.sample(real_count)
            fstates, factions, frewards, fnext_states, fnot_terminals, _ = self.fake_buffer.sample(fake_count)

            # print(states[0], actions[0], rewards[0], next_states[0], not_terminals[0])
            # print(fstates[0], factions[0], frewards[0], fnext_states[0], fnot_terminals[0])

            shuffle = torch.randperm(real_count + fake_count)

            states = torch.cat([states[:real_count], fstates], dim=0)[shuffle]
            actions = torch.cat([actions[:real_count], factions], dim=0)[shuffle]
            rewards = torch.cat([rewards[:real_count], frewards], dim=0)[shuffle]
            next_states = torch.cat([next_states[:real_count], fnext_states], dim=0)[shuffle]
            not_terminals = torch.cat([not_terminals[:real_count], fnot_terminals], dim=0)[shuffle]

        # Q Target
        # For this we need:
        # * (s_t, a_t, r_t) ~ buffer
        # * min{1,2}(Q(s_t, a_t))
        # * a'_t ~ policy(s_[t+1])
        # This maps to Eq 3 and Eq 5 from Haarnoja et al. 2018

        # Parameters of the target critics are frozen so don't update them!
        with torch.no_grad():
            target_actions, log_probs = self.actor.compute(next_states)
            target_input = torch.cat([next_states, target_actions], dim=-1)

            # Calculate both critic Qs and then take the min
            target_Q1 = self.target_critic_1(target_input)
            target_Q2 = self.target_critic_2(target_input)
            min_Q_target = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_probs

            # Compute r(s_t, a_t) + gamma * E_{s_[t+1] ~ p}(V_target(s_[t+1]))
            target_Q = rewards.unsqueeze(1) + not_terminals.unsqueeze(1) * self.gamma * min_Q_target 
            
            # Log for TB
            self._writer.add_scalar("stats/target_q", target_Q.detach().cpu().mean().item(), self._steps)

        # Now calculate the MSE for the critics and update parameters
        critic_input = torch.cat([states, actions], dim=-1)

        actual_Q1 = self.critic_1(critic_input)
        self._writer.add_scalar("stats/critic_1", actual_Q1.detach().cpu().mean().item(), self._steps)
        actual_Q2 = self.critic_2(critic_input)
        self._writer.add_scalar("stats/critic_2", actual_Q2.detach().cpu().mean().item(), self._steps)

        # Eq 5
        critic_1_loss = 0.5 * wis_mse(actual_Q1, target_Q, buffer_weights) #F.mse_loss(actual_Q1, target_Q)
        critic_2_loss = 0.5 * wis_mse(actual_Q2, target_Q, buffer_weights) #F.mse_loss(actual_Q2, target_Q)

        self._writer.add_scalar("stats/critic_loss_1", critic_1_loss.detach().cpu().item(), self._steps)
        self._writer.add_scalar("stats/critic_loss_2", critic_1_loss.detach().cpu().item(), self._steps)

        self.opt_critic_1.zero_grad()
        critic_1_loss.backward()
        self.opt_critic_1.step()
        
        self.opt_critic_2.zero_grad()
        critic_2_loss.backward()
        self.opt_critic_2.step()

        td_errors = torch.abs(target_Q - actual_Q1).detach().cpu().numpy()
        self.replay_buffer.update_priorities(buffer_indexes, td_errors)

        # Q functions updated, now update the policy, need:
        # * a'_t ~ policy(s_t) (NOT s_[t+1] like in Q updates)
        # * log(a'_t | s_t) from policy
        # * current alpha
        # Maps to Eq 7

        g_steps = 1 if 8 * self.warmup_steps > self.steps else self.gradient_steps

        for s in range(g_steps):
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

            if s == 0:
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

        # Update the frozen critics, delayed
        # if self.steps % 2 == 0:
        self.update_target_parameters()

        if self.steps < 2000:
            return

        for _ in range(4):
            traj_states, traj_actions, _ = self.replay_buffer.sample_trajectories(1024, self.horizon)
            # print(traj_states.shape, traj_actions.shape)
            dyn_loss = self.dynamics.train_model(traj_states, traj_actions)
            self._writer.add_scalar("train/dynamics_2", dyn_loss, self.steps)

        if self.steps > self.warmup_steps and self.steps % self.rollout_lifetime_steps == 0:
            self.rollout_model()

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
            
                g_steps = 1 # if self.warmup_steps > self.steps else self.gradient_steps

                for _ in range(g_steps):
                    self.train_policy()
                
                next_state, reward, done, _ = self.env.step(action)
                self._steps += 1
                self._episode_steps += 1
                episode_reward += reward

                self.replay_buffer.store_replay(state, action, reward, next_state, done, None)
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