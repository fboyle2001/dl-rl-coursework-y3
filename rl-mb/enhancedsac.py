from agents import RLAgent
from typing import Union, Optional
import networks
import torch 
import copy
import numpy as np
import fbbuffer
import buffers
import torch.nn.functional as F
import svgsacmodels as ssm
import svg
import gym

class EnhancedSACAgent(RLAgent):
    def __init__(self, env_name: str, device: Union[str, torch.device], video_every: Optional[int],
                        buffer_size: int = int(1e6), lr: float = 0.0003, tau: float = 0.005,
                        replay_batch_size: int = 256, gamma: float = 0.99, gradient_steps: int = 1,
                        warmup_steps: int = 10000, target_update_interval: int = 1, normalize_obs: bool = False):
        """
        SAC Implementation
        """
        super().__init__("EnhancedSAC", env_name, device, video_every)
        # self.env = gym.wrappers.NormalizeReward(self.env)

        # self.replay_buffer = buffers.StandardReplayBuffer(self._state_dim, self._action_dim, buffer_size, self.device)
        self.replay_buffer = fbbuffer.ReplayBuffer(self._state_dim, self._action_dim, buffer_size, self.device, normalize_obs=normalize_obs)

        # Critics predicts the reward of taking action A from state S
        self.critic_1 = networks.Critic(input_size=self._state_dim + self._action_dim, output_size=1).to(self.device)
        self.critic_2 = networks.Critic(input_size=self._state_dim + self._action_dim, output_size=1).to(self.device)

        # Actor predicts the action to take based on the current state
        # Though in SAC it actually predicts the distribution of actions
        # self.actor = networks.GaussianActor(self._state_dim, self._action_dim, self.device)
        self.actor = svg.Actor(self._state_dim, self._action_dim, 512, 4, log_std_bounds=[-5, 2]).to(self.device)

        # Establish the target networks
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)

        # Defined in the paper as -dim(A)
        # This line is taken from elsewhere
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item() # type: ignore

        # Alpha can get very close to 0 so for the same reason as in
        # the GaussianActor we optimise ln(alpha) instead for numerical stability
        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=self.device)

        # Optimisers
        self.opt_critic_1 = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.opt_critic_2 = torch.optim.Adam(self.critic_2.parameters(), lr=lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)

        self.action_range = [float(self.env.action_space.low.min()), float(self.env.action_space.high.max())] # type: ignore
        self.horizon = 3

        # Dynamics
        self.dynamics = ssm.SeqDx(self.env.spec.id, self._state_dim, self._action_dim, self.action_range, self.horizon, self.device, True, 1.0, 512, 2, 512, 0, "GRU", 512, 2, 1e-3).to(self.device) # type: ignore
        self.termination_model = networks.TerminationModel(self._state_dim + self._action_dim, 1).to(self.device)
        self.reward_model = networks.RewardModel(self._state_dim + self._action_dim, 1).to(self.device)
        self.rollout_lifetime_steps = 100
        self.rollouts_per_step = 256
        self.rollout_length = 3
        self.model_buffer = fbbuffer.ReplayBuffer(self._state_dim, self._action_dim, self.rollouts_per_step * self.rollout_lifetime_steps * self.rollout_length, self.device, normalize_obs=normalize_obs)

        self.opt_rewards = torch.optim.Adam(self.reward_model.parameters(), lr=1e-3)
        self.opt_terminations = torch.optim.Adam(self.termination_model.parameters(), lr=1e-3)

        # Hyperparameters
        self.gamma = gamma 
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.replay_batch_size = replay_batch_size
        self.real_ratio = 0.5
        self.warmup_steps = warmup_steps
        self.gradient_steps = 2
        self.multi_step_batch_size = 1024
        self.M_seqs = 4
        self.M_steps = 1

        self._train_steps = 0
    
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update_rewards(self, states, actions, rewards):
        # Simple to train, just MSE between predicted and real
        states_actions = torch.cat([states, actions], dim=-1)
        predicted_rewards = self.reward_model(states_actions)
        # rewards = rewards.unsqueeze(1)

        assert predicted_rewards.shape == rewards.shape

        loss = F.mse_loss(predicted_rewards, rewards)
        self._writer.add_scalar("loss/rewards_model", loss.detach().item(), self._steps)

        self.opt_rewards.zero_grad()
        loss.backward()
        self.opt_rewards.step()

    def update_terminations(self, states, actions, not_dones):
        dones = (1. - not_dones)

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

    def sample_evaluation_action(self, states: np.ndarray) -> np.ndarray:
        state = torch.FloatTensor(states).to(self.device).unsqueeze(0)

        with torch.no_grad():
            # action = self.actor.compute_actions(state, stochastic=False)
            action, _, _ = self.actor(state, compute_pi=False, compute_log_pi=False)

        return action.detach().cpu().numpy()[0]

    def sample_regular_action(self, states: np.ndarray) -> np.ndarray:
        state = torch.FloatTensor(states).to(self.device).unsqueeze(0)

        with torch.no_grad():
            # action = self.actor.compute_actions(state, stochastic=True)
            _, action, _ = self.actor(state, compute_log_pi=False)

        return action.detach().cpu().numpy()[0]

    def rollout_model(self):
        if self.real_ratio == 1:
            return

        states, _, nr, ns, _, nd = self.replay_buffer.sample(self.rollouts_per_step * self.rollout_lifetime_steps)
        nd = 1 - nd # We want to know if it is done rather than not done

        for i in range(self.rollout_length):
            actions = torch.tensor(self.sample_regular_action(states.detach().cpu().numpy()), device=self.device)
            next_states = self.dynamics.unroll(states, actions.unsqueeze(0), detach_xt=True).squeeze(0)

            joined = torch.cat([states, actions], dim=-1)
            reward = self.reward_model(joined)
            done = self.termination_model(joined).sigmoid()

            done[torch.where(done >= 0.5)] = 1
            done[torch.where(done < 0.5)] = 0

            self.model_buffer.bulk_add(
                states.detach().cpu().numpy(),
                actions.detach().cpu().numpy(),
                reward.detach().cpu().numpy(),
                next_states.detach().cpu().numpy(),
                done.detach().cpu().numpy(),
                done.detach().cpu().numpy()
            )

            # Write losses to TB
            if i == 0:
                self._writer.add_scalar("rollout/dones", F.mse_loss(done, nd), self.steps)
                self._writer.add_scalar("rollout/rewards", F.mse_loss(reward, nr), self.steps)
                self._writer.add_scalar("rollout/next_states", F.mse_loss(next_states, ns), self.steps)

            # print(reward, nr)
            # print(done, nd)

            states = next_states


        # predicted_actions, _, predicted_states = self.dynamics.unroll_policy(states, self.actor, sample=True, last_u=False, detach_xt=True)

        # for t in range(predicted_states.shape[0] - 1):
        #     ts_actions = predicted_actions[t]
        #     ts_states = predicted_states[t]
        #     ts_joined = torch.cat([ts_states, ts_actions], dim=-1)
        #     ts_rewards = self.reward_model(ts_joined)
        #     ts_dones = self.termination_model(ts_joined).sigmoid()
        #     ts_dones[torch.where(ts_dones > 0.5)] = 1
        #     ts_dones[torch.where(ts_dones <= 0.5)] = 0

        #     for i in range(ts_states.shape[0]):
        #         self.model_buffer.add(
        #             ts_states[i].detach().cpu().numpy(),
        #             ts_actions[i].detach().cpu().numpy(),
        #             ts_rewards[i].detach().cpu().numpy(),
        #             predicted_states[t+1][i].detach().cpu().numpy(),
        #             ts_dones[i].detach().cpu().numpy(),
        #             ts_dones[i].detach().cpu().numpy()
        #         )

        # want states, predicted_actions[0], predicted_states[0]
        # predicted_states[0], predicted_actions[1], predicted_states[1]
        # need to predict reward and terminals too

        # print(predicted_states.shape, predicted_actions.shape)

    def train_policy(self) -> None:
        # Need to fill the buffer before we can do anything
        if len(self.replay_buffer) < self.replay_batch_size:
            return

        self._train_steps += 1

        for s in range(self.M_steps):
            # Firstly update the Q functions (the critics)
            states, actions, rewards, next_states, not_dones, not_dones_no_max = self.replay_buffer.sample(self.replay_batch_size)

            # Update reward and termination models
            self.update_terminations(states, actions, not_dones_no_max)
            self.update_rewards(states, actions, rewards)

            # if self.steps > self.warmup_steps:
            #    print("Taking step", len(self.model_buffer))

            # Mix in some fake samples
            if len(self.model_buffer) != 0:
                real_count = int(self.replay_batch_size * self.real_ratio)
                fake_count = self.replay_batch_size - real_count

                if self.steps % 1000 == 0 and s == 0:
                    self.log(f"Sampling {fake_count} fakes")

                #print(f"Sampling {fake_count} fakes")

                #rstates, ractions, rrewards, rnext_states, rnot_dones, rnot_dones_no_max = self.replay_buffer.sample(real_count)
                fstates, factions, frewards, fnext_states, fnot_dones, fnot_dones_no_max = self.model_buffer.sample(fake_count)

                shuffle = torch.randperm(real_count + fake_count)

                states = torch.cat([states[:real_count], fstates], dim=0)[shuffle]
                actions = torch.cat([actions[:real_count], factions], dim=0)[shuffle]
                rewards = torch.cat([rewards[:real_count], frewards], dim=0)[shuffle]
                next_states = torch.cat([next_states[:real_count], fnext_states], dim=0)[shuffle]
                not_dones = torch.cat([not_dones[:real_count], fnot_dones], dim=0)[shuffle]
                not_dones_no_max = torch.cat([not_dones_no_max[:real_count], fnot_dones_no_max], dim=0)[shuffle]

            # Q Target
            # For this we need:
            # * (s_t, a_t, r_t) ~ buffer
            # * min{1,2}(Q(s_t, a_t))
            # * a'_t ~ policy(s_[t+1])
            # This maps to Eq 3 and Eq 5 from Haarnoja et al. 2018

            # Parameters of the target critics are frozen so don't update them!
            with torch.no_grad():
                # target_actions = self.actor(next_states, compute_pi=False)
                mu, target_actions, log_pi = self.actor(next_states, compute_pi=True, compute_log_pi=True)
                target_input = torch.cat([next_states, target_actions], dim=-1)

                # Calculate both critic Qs and then take the min
                target_Q1 = self.target_critic_1(target_input)
                target_Q2 = self.target_critic_2(target_input)
                min_Q_target = torch.min(target_Q1, target_Q2) # - self.alpha.detach() * log_pi

                # Compute r(s_t, a_t) + gamma * E_{s_[t+1] ~ p}(V_target(s_[t+1]))
                # target_Q = rewards.unsqueeze(1) + not_dones_no_max.unsqueeze(1) * self.gamma * min_Q_target 
                target_Q = rewards + not_dones_no_max * self.gamma * min_Q_target 
                
                # Log for TB
                self._writer.add_scalar("stats/target_q", target_Q.detach().cpu().mean().item(), self._train_steps)

            # Now calculate the MSE for the critics and update parameters
            critic_input = torch.cat([states, actions], dim=-1)

            actual_Q1 = self.critic_1(critic_input)
            self._writer.add_scalar("stats/critic_1", actual_Q1.detach().cpu().mean().item(), self._train_steps)
            actual_Q2 = self.critic_2(critic_input)
            self._writer.add_scalar("stats/critic_2", actual_Q2.detach().cpu().mean().item(), self._train_steps)

            # Eq 5
            critic_1_loss = 0.5 * F.mse_loss(actual_Q1, target_Q)
            critic_2_loss = 0.5 * F.mse_loss(actual_Q2, target_Q)

            self._writer.add_scalar("stats/critic_loss_1", critic_1_loss.detach().cpu().item(), self._train_steps)
            self._writer.add_scalar("stats/critic_loss_2", critic_1_loss.detach().cpu().item(), self._train_steps)

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

            _, actor_actions, actor_probs = self.actor(states, compute_pi=True, compute_log_pi=True)
            real_input = torch.cat([states, actor_actions], dim=-1)

            # See what the critics think to the actor's prediction of the next action
            real_Q1 = self.critic_1(real_input)
            real_Q2 = self.critic_2(real_input)
            min_Q = torch.min(real_Q1, real_Q2)

            self._writer.add_scalar('loss/mean_Q1', real_Q1.detach().mean().item(), self._train_steps)
            self._writer.add_scalar('loss/mean_Q2', real_Q2.detach().mean().item(), self._train_steps)

            # See Eq 7 
            actor_loss = (self.alpha * actor_probs - min_Q).mean()
            self._writer.add_scalar('loss/actor', actor_loss.detach().item(), self._train_steps)

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
            alpha_loss = -self.alpha * (actor_probs + self.target_entropy).detach()
            alpha_loss = alpha_loss.mean()
            self._writer.add_scalar("loss/alpha", alpha_loss.detach().item(), self._train_steps)

            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()

            # Make sure to update alpha now we've updated log_alpha
            # self.alpha = self.log_alpha.exp() # no need as handled by property

            self._writer.add_scalar('stats/alpha', self.alpha.detach().item(), self._train_steps)

            # Update the frozen critics
            self.update_target_parameters()

        if self.steps < self.warmup_steps:
            return

        # Multi-step updates
        for _ in range(self.M_seqs):
            # Now we move on to updating the dynamics model
            # Start by sampling the multi-step trajectories
            states, actions, rewards = self.replay_buffer.sample_multistep(self.multi_step_batch_size, self.horizon)
            dynamics_loss = self.dynamics.update_step(states, actions, rewards, self._writer, self.steps)
            self._writer.add_scalar("loss/dynamics", dynamics_loss, self._train_steps)

        # Give the dynamics model a chance to warmup prior to using samples from it
        if self._train_steps % self.rollout_lifetime_steps == 0 and self.steps >= 2 * self.warmup_steps:
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
                    action = self.sample_regular_action(state)

                policy_steps = self.gradient_steps if self.steps > 2 * self.warmup_steps else 1
            
                for _ in range(policy_steps):
                    self.train_policy()
                
                next_state, reward, done, _ = self.env.step(action)
                done_no_max = float(done) if self._episode_steps + 1 < self._max_episodes else 0.

                self._steps += 1
                self._episode_steps += 1
                episode_reward += reward

                self.replay_buffer.add(state, action, reward, next_state, float(done), done_no_max)
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
                        action = self.sample_evaluation_action(state)
                        next_state, reward, done, _ = self.env.step(action)
                        ep_reward += reward
                        state = next_state
                    
                    avg_reward += ep_reward
                
                avg_reward /= eval_episodes
                self._writer.add_scalar("stats/eval_reward", avg_reward, self.episodes)
                self.log(f"[EVAL] Average reward over {eval_episodes} was {avg_reward}")

# s = EnhancedSACAgent("Pendulum-v1", "cuda", None)
# s.run()