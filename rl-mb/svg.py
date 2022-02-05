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

from torch import nn
import copy
import numpy as np
import torch.nn.functional as F
import math
from torch import distributions as pyd
import gym
import wrappers

## STOLEN FUNCTIONS
def get_params(models):
    for m in models:
        for p in m.parameters():
            yield p

def accum_prod(x):
    assert x.dim() == 2
    x_accum = [x[0]]
    for i in range(x.size(0)-1):
        x_accum.append(x_accum[-1]*x[i])
    x_accum = torch.stack(x_accum, dim=0)
    return x_accum

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        # print(m.weight.data.shape)
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0) # type: ignore
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

class TanhTransform(pyd.transforms.Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = pyd.constraints.real # type: ignore
    codomain = pyd.constraints.interval(-1.0, 1.0) # type: ignore
    bijective = True
    sign = +1 # type: ignore

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self):
        return self.base_dist.entropy()

class Actor(nn.Module):
    """An isotropic Gaussian policy."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, compute_pi=True, compute_log_pi=True):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        policy = SquashedNormal(mu, std)
        pi = policy.rsample() if compute_pi else None
        log_pi = policy.log_prob(pi).sum(
            -1, keepdim=True) if compute_log_pi else None

        return policy.mean, pi, log_pi

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)

# Copyright (c) Facebook, Inc. and its affiliates.

class SeqDx(nn.Module):
    def __init__(self,
                 env_name,
                 obs_dim, action_dim, action_range,
                 horizon, device,
                 detach_xt,
                 clip_grad_norm,
                 xu_enc_hidden_dim, xu_enc_hidden_depth,
                 x_dec_hidden_dim, x_dec_hidden_depth,
                 rec_type, rec_latent_dim, rec_num_layers,
                 lr):
        super().__init__()

        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = device
        self.detach_xt = detach_xt
        self.clip_grad_norm = clip_grad_norm

        # Manually freeze the goal locations
        if env_name == 'gym_petsReacher':
            self.freeze_dims = torch.LongTensor([7,8,9])
        elif env_name == 'gym_petsPusher':
            self.freeze_dims = torch.LongTensor([20,21,22])
        else:
            self.freeze_dims = None

        self.rec_type = rec_type
        self.rec_num_layers = rec_num_layers
        self.rec_latent_dim = rec_latent_dim

        self.xu_enc = mlp(
            obs_dim+action_dim, xu_enc_hidden_dim, rec_latent_dim, xu_enc_hidden_depth)
        self.x_dec = mlp(
            rec_latent_dim, x_dec_hidden_dim, obs_dim, x_dec_hidden_depth)

        self.apply(weight_init) # Don't apply this to the recurrent unit.


        mods = [self.xu_enc, self.x_dec]

        if rec_num_layers > 0:
            if rec_type == 'LSTM':
                self.rec = nn.LSTM(
                    rec_latent_dim, rec_latent_dim, num_layers=rec_num_layers)
            elif rec_type == 'GRU':
                self.rec = nn.GRU(
                    rec_latent_dim, rec_latent_dim, num_layers=rec_num_layers)
            else:
                assert False
            mods.append(self.rec) #type: ignore

        params = get_params(mods)
        self.opt = torch.optim.Adam(params, lr=lr)

    def __getstate__(self):
        d = self.__dict__
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.rec.flatten_parameters()

    def init_hidden_state(self, init_x):
        assert init_x.dim() == 2
        n_batch = init_x.size(0)

        if self.rec_type == 'LSTM':
            h = torch.zeros(
                self.rec_num_layers, n_batch, self.rec_latent_dim, device=init_x.device)
            c = torch.zeros_like(h)
            h = (h, c)
        elif self.rec_type == 'GRU':
            h = torch.zeros(
                self.rec_num_layers, n_batch, self.rec_latent_dim, device=init_x.device)
        else:
            assert False

        return h

    def unroll_policy(self, init_x, policy, sample=True,
                      last_u=True, detach_xt=True):
        assert init_x.dim() == 2
        n_batch = init_x.size(0)

        if self.freeze_dims is not None:
            obs_frozen = init_x[:, self.freeze_dims]

        if self.rec_num_layers > 0:
            h = self.init_hidden_state(init_x)

        pred_xs = []
        us = []
        log_p_us = []
        xt = init_x
        for t in range(self.horizon-1):
            policy_kwargs = {}
            if sample:
                _, ut, log_p_ut = policy(xt, **policy_kwargs)
            else:
                ut, _, log_p_ut = policy(xt, **policy_kwargs)
            us.append(ut)
            log_p_us.append(log_p_ut)

            if detach_xt:
                xt = xt.detach()

            xut = torch.cat((xt, ut), dim=1)
            xu_emb = self.xu_enc(xut).unsqueeze(0)
            if self.rec_num_layers > 0:
                xtp1_emb, h = self.rec(xu_emb, h) # type: ignore
            else:
                xtp1_emb = xu_emb
            xtp1 = xt + self.x_dec(xtp1_emb.squeeze(0))
            if self.freeze_dims is not None:
                xtp1[:,self.freeze_dims] = obs_frozen  # type: ignore

            pred_xs.append(xtp1)
            xt = xtp1

        if last_u:
            policy_kwargs = {}
            if sample:
                _, ut, log_p_ut = policy(xt, **policy_kwargs)
            else:
                ut, _, log_p_ut = policy(xt, **policy_kwargs)
            us.append(ut)
            log_p_us.append(log_p_ut)

        us = torch.stack(us)
        log_p_us = torch.stack(log_p_us).squeeze(2)
        if self.horizon <= 1:
            pred_xs = torch.empty(0, n_batch, self.obs_dim).to(init_x.device)
        else:
            pred_xs = torch.stack(pred_xs)

        return pred_xs, us, log_p_us


    def unroll(self, x, us, detach_xt=False):
        assert x.dim() == 2
        assert us.dim() == 3
        n_batch = x.size(0)
        assert us.size(1) == n_batch

        if self.freeze_dims is not None:
            obs_frozen = x[:, self.freeze_dims]

        if self.rec_num_layers > 0:
            h = self.init_hidden_state(x)

        pred_xs = []
        xt = x
        for t in range(us.size(0)):
            ut = us[t]

            if detach_xt:
                xt = xt.detach()

            xut = torch.cat((xt, ut), dim=1)
            xu_emb = self.xu_enc(xut).unsqueeze(0)
            if self.rec_num_layers > 0:
                xtp1_emb, h = self.rec(xu_emb, h) # type: ignore
            else:
                xtp1_emb = xu_emb
            xtp1 = xt + self.x_dec(xtp1_emb.squeeze(0))
            if self.freeze_dims is not None:
                xtp1[:,self.freeze_dims] = obs_frozen # type: ignore
            pred_xs.append(xtp1)
            xt = xtp1

        pred_xs = torch.stack(pred_xs)

        return pred_xs


    def forward(self, x, us):
        return self.unroll(x, us)


    def update_step(self, obs, action, reward, logger, step):
        assert obs.dim() == 3
        T, batch_size, _ = obs.shape

        pred_obs = self.unroll(obs[0], action[:-1], detach_xt=self.detach_xt)
        target_obs = obs[1:]
        assert pred_obs.size() == target_obs.size()

        obs_loss = F.mse_loss(pred_obs, target_obs, reduction='mean')

        self.opt.zero_grad()
        obs_loss.backward()
        if self.clip_grad_norm is not None:
            assert len(self.opt.param_groups) == 1
            params = self.opt.param_groups[0]['params']
            torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm) # type: ignore
        self.opt.step()

        logger.add_scalar('train_model/obs_loss', obs_loss.detach().cpu(), step)

        return obs_loss.item()

class SACSVGAgent(RLAgent):
    def __init__(self, env_name: str, device: Union[str, torch.device], video_every: Optional[int], normalise: bool = True):
        super().__init__("SVGSafety", env_name, device, video_every)

        # Try normalisation:
        self.env = gym.wrappers.RescaleAction(self.env, -1., 1.)
        self.env = gym.wrappers.ClipAction(self.env) # type: ignore
        # self.env = gym.wrappers.NormalizeObservation(self.env)
        self.env = gym.wrappers.NormalizeReward(self.env)


        print(f"Normalisation: {normalise}")
        self.replay_buffer = buffers.MultiStepReplayBuffer(self._state_dim, self._action_dim, int(1e6), self.device, normalise=normalise)

        # Critics predicts the reward of taking action A from state S
        self.critic_1 = networks.Critic(input_size=self._state_dim + self._action_dim, output_size=1).to(self.device)
        self.critic_2 = networks.Critic(input_size=self._state_dim + self._action_dim, output_size=1).to(self.device)

        # Actor predicts the action to take based on the current state
        # Though in SAC it actually predicts the distribution of actions
        # self.actor = networks.GaussianActor(self._state_dim, self._action_dim, self.device)
        self.replaced_actor = Actor(self._state_dim, self._action_dim, 512, 4, log_std_bounds=[-5, 2]).to(self.device)
        # Establish the target networks
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)

        # Defined in the paper as -dim(A)
        # This line is taken from elsewhere
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item() # type: ignore

        # Alpha can get very close to 0 so for the same reason as in
        # the GaussianActor we optimise ln(alpha) instead for numerical stability
        self.log_alpha = torch.tensor(np.log(1.), requires_grad=True, device=self.device)
        # self.alpha = 0.2

        # World Model
        # self.dynamics = networks.GRUDynamicsModel(self._state_dim + self._action_dim, self._state_dim, horizon=3,
        #                             device=self.device, clip_grad=1.0, lr=1e-3).to(self.device)
        print(self.env.spec.id) # type: ignore
        self.dynamics = SeqDx(self.env.spec.id, self._state_dim, self._action_dim, 1, 3, self.device, True, 1.0, 512, 2, 512, 0, "GRU", 512, 2, 1e-3).to(self.device) # type: ignore
        self.termination_model = networks.TerminationModel(self._state_dim + self._action_dim, 1).to(self.device)
        self.reward_model = networks.RewardModel(self._state_dim + self._action_dim, 1).to(self.device)

        # Optimisers
        self.opt_critic_1 = torch.optim.Adam(self.critic_1.parameters(), lr=1e-4)
        self.opt_critic_2 = torch.optim.Adam(self.critic_2.parameters(), lr=1e-4)
        self.opt_actor = torch.optim.Adam(self.replaced_actor.parameters(), lr=1e-4)
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

        self.warmup_steps = 10000

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def sample_evaluation_action(self, states: torch.Tensor) -> np.ndarray:
        states = states.unsqueeze(0)

        with torch.no_grad():
            # action = self.actor.compute_actions(states, stochastic=False)
            action, _, _ = self.replaced_actor(states, compute_pi=False, compute_log_pi=False)
            
        return action.detach().cpu().numpy()[0]

    def sample_regular_action(self, states: torch.Tensor) -> np.ndarray:
        states = states.unsqueeze(0)

        with torch.no_grad():
            #action = self.actor.compute_actions(states, stochastic=True)
            _, action, _ = self.replaced_actor(states, compute_log_pi=False)

        return action.detach().cpu().numpy()[0]

    def update_target_parameters(self):
        # Update frozen targets, taken from https://github.com/sfujim/TD3/blob/385b33ac7de4767bab17eb02ade4a268d3e4e24f/TD3.py
        # theta' = tau * theta + (1 - tau) * theta'
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_actor_and_alpha(self, states):
        if self._steps < self.warmup_steps:
            #actor_actions, log_probs = self.actor.compute(states)
            _, actor_actions, log_probs = self.replaced_actor(states)
            real_input = torch.cat([states, actor_actions], dim=-1)

            # See what the critics think to the actor's prediction of the next action
            real_Q1 = self.critic_1(real_input)
            real_Q2 = self.critic_2(real_input)
            min_Q = torch.min(real_Q1, real_Q2)

            actor_loss = (self.alpha * log_probs - min_Q).mean()
        elif self._steps > self.warmup_steps:
            # Unroll from the dynamics using the initial states
            predicted_states, policy_actions, log_probs = self.dynamics.unroll_policy(states, self.replaced_actor)
            
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
            # rewards -= self.alpha * log_probs

            # Discount rewards the further in time we go
            rewards *= self.gamma_horizon.unsqueeze(1)

            # Take the average reward over each timestep
            # [timesteps, batch] -> [batch] dimensions
            rewards = rewards.sum(dim=0)

            # Calculate the loss
            # TODO: Where does this actually come from?
            actor_loss = -(rewards / self.horizon).mean()
        else:
            return

        self._writer.add_scalar("loss/actor", actor_loss.detach().cpu(), self._steps)
        self._writer.add_scalar("stats/entropy", -log_probs[0].detach().mean(), self._steps)

        self.opt_actor.zero_grad()
        actor_loss.backward()
        # nn.utils.clip_grad_norm_(self.replaced_actor.parameters(), 1.0) #type: ignore
        self.opt_actor.step()

        # Now update alpha (do it here since we have everything we need so it's efficient)
        # Optimise over the first timestep i.e. those sampled directly from D only
        # Remember ~= expectation so take the mean
        alpha_loss = (-self.alpha * (log_probs[0] + self.target_entropy).detach()).mean()

        self._writer.add_scalar("loss/alpha", alpha_loss.detach().cpu(), self._steps)

        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        # nn.utils.clip_grad_norm_([self.log_alpha], 1.0) # type: ignore
        self.opt_alpha.step()

        self._writer.add_scalar("stats/alpha", self.alpha.detach().cpu(), self._steps)
        # self._writer.add_scalar("stats/alpha", self.alpha, self._steps)

    def update_critics(self, states, actions, rewards, next_states, not_dones):
        # Compute Q^(target)_theta(x_t, u_t)
        with torch.no_grad():
            # V_theta_frozen(x_[t+1])
            #target_actions, log_probs = self.actor.compute(next_states)
            _, target_actions, log_probs = self.replaced_actor(next_states, compute_pi=True, compute_log_pi=True)
            target_input = torch.cat([next_states, target_actions], dim=-1)

            # Calculate both critic Qs and then take the min
            target_Q1 = self.target_critic_1(target_input)
            target_Q2 = self.target_critic_2(target_input)
            min_Q_target = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_probs
            # min_Q_target = torch.min(target_Q1, target_Q2) - self.alpha * log_probs

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
        critic_1_loss = 0.5 * F.mse_loss(actual_Q1, target_Q)
        critic_2_loss = 0.5 * F.mse_loss(actual_Q2, target_Q)

        self._writer.add_scalar("loss/q1", critic_1_loss.detach().cpu().item(), self._steps)
        self._writer.add_scalar("loss/q2", critic_2_loss.detach().cpu().item(), self._steps)

        # Optimise
        self.opt_critic_1.zero_grad()
        critic_1_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0) #type: ignore
        self.opt_critic_1.step()
        
        self.opt_critic_2.zero_grad()
        critic_2_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0) #type: ignore
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
            states, actions, rewards = self.replay_buffer.sample_sequences(self.multi_step_batch_size, self.horizon)
            dynamics_loss = self.dynamics.update_step(states, actions, rewards, self._writer, self.steps)
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