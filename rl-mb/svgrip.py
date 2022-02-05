from tracemalloc import start
from typing import Union, Optional
from agents import RLAgent

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
import numpy as np
import math
from sortedcontainers import SortedSet
import numpy as np
import numpy.random as npr
import copy
import time
from gym import spaces
import gym

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
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

def get_params(models):
    for m in models:
        for p in m.parameters():
            yield p

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
            mods.append(self.rec) # type: ignore

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
                      last_u=True, detach_xt=False):
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
                xtp1_emb, h = self.rec(xu_emb, h)  # type: ignore
            else:
                xtp1_emb = xu_emb
            xtp1 = xt + self.x_dec(xtp1_emb.squeeze(0))
            if self.freeze_dims is not None:
                xtp1[:,self.freeze_dims] = obs_frozen # type: ignore

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

        return us, log_p_us, pred_xs


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


    def update_step(self, obs, action, reward, writer, step):
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

        writer.add_scalar('train_model/obs_loss', obs_loss.detach().cpu(), step)

        return obs_loss.item()

def eval_float_maybe(x):
    if isinstance(x, int):
        return float(x)
    elif isinstance(x, float):
        return x
    else:
        return float(eval(x))

class LearnTemp:
    def __init__(
        self,
        init_temp, max_steps,
        init_targ_entr, final_targ_entr,
        entr_decay_factor,
        only_decrease_alpha,
        lr, device
    ):
        self.device = device
        self.init_temp = init_temp
        self.max_steps = max_steps
        self.log_alpha = torch.tensor(np.log(init_temp)).to(self.device)
        self.log_alpha.requires_grad = True
        self.init_targ_entr = eval_float_maybe(init_targ_entr)
        self.final_targ_entr = eval_float_maybe(final_targ_entr)
        assert self.final_targ_entr <= self.init_targ_entr
        self.entr_decay_factor = entr_decay_factor
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.targ_entr = self.init_targ_entr
        self.only_decrease_alpha = only_decrease_alpha

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, first_log_p, writer, step):
        if step < self.max_steps:
            t = (1.-step/self.max_steps)**self.entr_decay_factor
            self.targ_entr = (self.init_targ_entr - self.final_targ_entr)*t + self.final_targ_entr
        else:
            self.targ_entr = self.final_targ_entr
            
        self.log_alpha_opt.zero_grad()
        alpha_loss = (self.alpha * (-first_log_p - self.targ_entr).detach()).mean()
        alpha_loss.backward()
        if not self.only_decrease_alpha or self.log_alpha.grad.item() > 0.:  # type: ignore
            self.log_alpha_opt.step()
        writer.add_scalar('train_actor/target_entropy', self.targ_entr, step)
        writer.add_scalar('train_alpha/loss', alpha_loss.detach().cpu(), step)

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

    def log(self, writer, step):
        for k, v in self.outputs.items():
            writer.add_histogram(f'train_actor/{k}_hist', v, step)

        # for i, m in enumerate(self.trunk):
        #     if type(m) == nn.Linear:
        #         to do.log_param(f'train_actor/fc{i}', m, step)

class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, bounds=None):
        super().__init__()

        self.Q1 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.bounds = bounds

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        if hasattr(self, 'bounds') and self.bounds is not None:
            lb, ub = self.bounds
            r = ub-lb
            q1 = r*torch.sigmoid(q1) + lb
            q2 = r*torch.sigmoid(q2) + lb

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, writer, step):
        for k, v in self.outputs.items():
            writer.add_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        # for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
        #     assert type(m1) == type(m2)
        #     if type(m1) is nn.Linear:
        #         to do.log_param(f'train_critic/q1_fc{i}', m1, step)
        #         to do.log_param(f'train_critic/q2_fc{i}', m2, step)

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

def accum_prod(x):
    assert x.dim() == 2
    x_accum = [x[0]]
    for i in range(x.size(0)-1):
        x_accum.append(x_accum[-1]*x[i])
    x_accum = torch.stack(x_accum, dim=0)
    return x_accum

# https://pswww.slac.stanford.edu/svn-readonly/psdmrepo/RunSummary/trunk/src/welford.py
class Welford(object):
    """Knuth implementation of Welford algorithm.
    """

    def __init__(self, x=None):
        self._K = np.float64(0.)
        self.n = np.float64(0.)
        self._Ex = np.float64(0.)
        self._Ex2 = np.float64(0.)
        self.shape = None
        self._min = None
        self._max = None
        self._init = False
        self.__call__(x)

    def add_data(self, x):
        """Add data.
        """
        if x is None:
            return

        x = np.array(x)
        self.n += 1.
        if not self._init:
            self._init = True
            self._K = x
            self._min = x
            self._max = x
            self.shape = x.shape
        else:
            self._min = np.minimum(self._min, x) # type: ignore
            self._max = np.maximum(self._max, x) # type: ignore

        self._Ex += (x - self._K) / self.n
        self._Ex2 += (x - self._K) * (x - self._Ex)
        self._K = self._Ex

    def __call__(self, x):
        self.add_data(x)

    def max(self):
        """Max value for each element in array.
        """
        return self._max

    def min(self):
        """Min value for each element in array.
        """
        return self._min

    def mean(self, axis=None):
        """Compute the mean of accumulated data.

           Parameters
           ----------
           axis: None or int or tuple of ints, optional
                Axis or axes along which the means are computed. The default is to
                compute the mean of the flattened array.
        """
        if self.n < 1:
            return None

        val = np.array(self._K + self._Ex / np.float64(self.n))
        if axis:
            return val.mean(axis=axis)
        else:
            return val

    def sum(self, axis=None):
        """Compute the sum of accumulated data.
        """
        return self.mean(axis=axis)*self.n # type: ignore

    def var(self):
        """Compute the variance of accumulated data.
        """
        if self.n <= 1:
            return  np.zeros(self.shape) # type: ignore

        val = np.array((self._Ex2 - (self._Ex*self._Ex)/np.float64(self.n)) / np.float64(self.n-1.))

        return val

    def std(self):
        """Compute the standard deviation of accumulated data.
        """
        return np.sqrt(self.var())

#    def __add__(self, val):
#        """Add two Welford objects.
#        """
#

    def __str__(self):
        if self._init:
            return "{} +- {}".format(self.mean(), self.std())
        else:
            return "{}".format(self.shape)

    def __repr__(self):
        return "< Welford: {:} >".format(str(self))

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, normalize_obs):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.capacity = capacity
        self.device = device

        self.pixels = False
        self.empty_data()

        self.done_idxs = SortedSet()
        self.global_idx = 0
        self.global_last_save = 0

        self.normalize_obs = normalize_obs

        if normalize_obs:
            assert not self.pixels
            self.welford = Welford()

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d['obses'], d['next_obses'], d['actions'], d['rewards'], \
          d['not_dones'], d['not_dones_no_max']
        return d

    def __setstate__(self, d):
        self.__dict__ = d

        # Manually need to re-load the transitions with load()
        self.empty_data()


    def empty_data(self):
        obs_dtype = np.float32 if not self.pixels else np.uint8
        obs_shape = self.obs_shape
        action_shape = self.action_shape
        capacity = self.capacity

        self.obses = np.empty((capacity, obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.payload = []
        self.done_idxs = None


    def __len__(self):
        return self.capacity if self.full else self.idx

    def get_obs_stats(self):
        assert not self.pixels
        MIN_STD = 1e-1
        MAX_STD = 10
        mean = self.welford.mean()
        std = self.welford.std()
        std[std < MIN_STD] = MIN_STD
        std[std > MAX_STD] = MAX_STD
        return mean, std

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        # For saving
        self.payload.append((
            obs.copy(), next_obs.copy(),
            action.copy(), reward,
            not done, not done_no_max
        ))

        if self.normalize_obs:
            self.welford.add_data(obs)

        # if self.full and not self.not_dones[self.idx]:
        if done:
            self.done_idxs.add(self.idx) # type: ignore
        elif self.full:
            self.done_idxs.discard(self.idx) # type: ignore

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.global_idx += 1
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx,
            size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        if self.normalize_obs:
            mu, sigma = self.get_obs_stats()
            obses = (obses-mu)/sigma
            next_obses = (next_obses-mu)/sigma

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)



        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max


    def sample_multistep(self, batch_size, T):
        assert batch_size < self.idx or self.full

        last_idx = self.capacity if self.full else self.idx
        last_idx -= T

        # raw here means the "coalesced" indices that map to valid
        # indicies that are more than T steps away from a done
        done_idxs_sorted = np.array(list(self.done_idxs) + [last_idx]) # type: ignore
        n_done = len(done_idxs_sorted)
        done_idxs_raw = done_idxs_sorted - np.arange(1, n_done+1)*T

        samples_raw = npr.choice(
            last_idx-(T+1)*n_done, size=batch_size,
            replace=True # for speed
        )
        samples_raw = sorted(samples_raw)  # type: ignore
        js = np.searchsorted(done_idxs_raw, samples_raw)  # type: ignore
        offsets = done_idxs_raw[js] - samples_raw + T
        start_idxs = done_idxs_sorted[js] - offsets

        obses, actions, rewards = [], [], []

        for t in range(T):
            obses.append(self.obses[start_idxs + t])
            actions.append(self.actions[start_idxs + t])
            rewards.append(self.rewards[start_idxs + t])
            assert np.all(self.not_dones[start_idxs + t])

        obses = np.stack(obses)
        actions = np.stack(actions)
        rewards = np.stack(rewards).squeeze(2)

        if self.normalize_obs:
            mu, sigma = self.get_obs_stats()
            obses = (obses-mu)/sigma

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)

        return obses, actions, rewards

# https://github.com/openai/gym/blob/master/gym/wrappers/rescale_action.py
class RescaleAction(gym.ActionWrapper):
    def __init__(self, env, a, b):
        assert isinstance(env.action_space, spaces.Box), (
            "expected Box action space, got {}".format(type(env.action_space)))
        assert np.less_equal(a, b).all(), (a, b)
        super(RescaleAction, self).__init__(env)
        dtype = env.action_space.sample().dtype
        self.a = np.zeros(env.action_space.shape, dtype=dtype) + a # type: ignore
        self.b = np.zeros(env.action_space.shape, dtype=dtype) + b # type: ignore
        self.action_space = spaces.Box(
            low=a, high=b, shape=env.action_space.shape) # type: ignore

    def action(self, action):
        assert np.all(np.greater_equal(action, self.a)), (action, self.a)
        assert np.all(np.less_equal(action, self.b)), (action, self.b)
        low = self.env.action_space.low # type: ignore
        high = self.env.action_space.high # type: ignore
        action = low + (high - low)*((action - self.a)/(self.b - self.a)) # type: ignore
        action = np.clip(action, low, high)
        return action

class SVGDirectRipAgent(RLAgent):
    def __init__(self, env_name: str, device: Union[str, torch.device], video_every: Optional[int], normalise: bool = True):
        super().__init__("SVGDirectRip", env_name, device, video_every, 50000, True, True, 42)

        print("Normalised", normalise)

        if normalise:
            self.env = RescaleAction(self.env, -1., 1.)

        self.normalise = normalise

        self.env_name = env_name
        self.obs_dim = self._state_dim
        self.action_dim = self._action_dim
        self.action_range = [float(self.env.action_space.low.min()), float(self.env.action_space.high.max())] # type: ignore
        self.device = torch.device(device)
        self.num_train_steps = int(2e6)
        # self.det_suffix = det_suffix

        horizon = 2

        self.discount = 0.99
        self.discount_horizon = torch.tensor(
            [self.discount**i for i in range(horizon)]).to(device)
        self.seq_batch_size = 512

        # self.seq_train_length = eval(seq_train_length)
        self.seq_train_length = 3

        self.step_batch_size = 1024
        self.update_freq = 1
        self.model_update_repeat = 4
        self.model_update_freq = 1
        self.model_free_update_repeat = 1

        self.horizon = horizon

        self.warmup_steps = 10000

        print(-torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()) # type: ignore

        self.temp = LearnTemp(2, int(2e6), -3, -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item(), 64, False, 1e-4, self.device) # type: ignore
        self.dx = SeqDx(self.env.spec.id, self._state_dim, self._action_dim, self.action_range, self.horizon, self.device, True, 1.0, 512, 2, 512, 0, "GRU", 512, 2, 1e-3).to(self.device) # type: ignore

        self.rew = mlp(
            self._state_dim+self._action_dim, 512, 1, 2
        ).to(self.device)
        self.rew_opt = torch.optim.Adam(self.rew.parameters(), lr=1e-3)

        self.done = mlp(
            self._state_dim+self._action_dim, 512, 1, 2
        ).to(self.device)
        self.done_ctrl_accum = True
        self.done_opt = torch.optim.Adam(self.done.parameters(), lr=1e-3)

        self.actor = Actor(self._state_dim, self._action_dim, 512, 4, [-5, 2]).to(self.device)
        mods = [self.actor]
        params = get_params(mods)
        self.actor_opt = torch.optim.Adam(
            params, lr=1e-4, betas=(0.9, 0.999))
        self.actor_update_freq = 1
        self.actor_mve = True
        self.actor_detach_rho = False
        self.actor_dx_threshold = None

        self.critic = None

        self.critic = DoubleQCritic(self._state_dim, self._action_dim, 512, 4).to(self.device)
        self.critic_target = DoubleQCritic(self._state_dim, self._action_dim, 512, 4).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.train()
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=1e-4)
        self.critic_tau = 0.005
        self.critic_target_update_freq = 1

        self.critic_target_mve = False
        self.full_target_mve = False

        self.train()
        self.last_step = 0
        self.rolling_dx_loss = None
    
    def train(self, training=True):
        self.training = training
        self.dx.train(training)
        self.rew.train(training)
        self.done.train(training) # type: ignore
        self.actor.train(training)
        if self.critic is not None:
            self.critic.train(training)
    
    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(dim=0)

        if not sample:
            action, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
        else:
            with torch.no_grad():
                _, action, _ = self.actor(obs, compute_log_pi=False)

        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return to_np(action[0])


    def expand_Q(self, xs, critic, sample=True, discount=False):
        assert xs.dim() == 2
        n_batch = xs.size(0)
        us, log_p_us, pred_obs = self.dx.unroll_policy(
            xs, self.actor, sample=sample, detach_xt=self.actor_detach_rho)

        all_obs = torch.cat((xs.unsqueeze(0), pred_obs), dim=0)
        xu = torch.cat((all_obs, us), dim=2)
        dones = self.done(xu).sigmoid().squeeze(dim=2) # type: ignore
        not_dones = 1. - dones
        not_dones = accum_prod(not_dones)
        last_not_dones = not_dones[-1]

        rewards = not_dones * self.rew(xu).squeeze(2)
        if critic is not None:
            with eval_mode(critic):
                q1, q2 = critic(all_obs[-1], us[-1])
            q = torch.min(q1, q2).reshape(n_batch)
            rewards[-1] = last_not_dones * q

        assert rewards.size() == (self.horizon, n_batch)
        assert log_p_us.size() == (self.horizon, n_batch)
        rewards -= self.temp.alpha.detach() * log_p_us

        if discount:
            rewards *= self.discount_horizon.unsqueeze(1)

        total_rewards = rewards.sum(dim=0)

        first_log_p = log_p_us[0]
        total_log_p_us = log_p_us.sum(dim=0).squeeze()
        return total_rewards, first_log_p, total_log_p_us

    def update_actor_and_alpha(self, xs, step):
        assert xs.ndimension() == 2
        n_batch, _ = xs.size()

        do_model_free_update = step < self.warmup_steps or \
          self.horizon == 0 or not self.actor_mve or \
          (self.actor_dx_threshold is not None and \
           self.rolling_dx_loss is not None and
           self.rolling_dx_loss > self.actor_dx_threshold)

        if do_model_free_update:
            # Do vanilla SAC updates while the model warms up.
            # i.e., fit to just the Q function
            _, pi, first_log_p = self.actor(xs)
            actor_Q1, actor_Q2 = self.critic(xs, pi) # type: ignore
            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.temp.alpha.detach() * first_log_p - actor_Q).mean()
        else:
            # Switch to the model-based updates.
            # i.e., fit to the controller's sequence cost
            rewards, first_log_p, total_log_p_us = self.expand_Q(
                xs, self.critic, sample=True, discount=True)
            assert total_log_p_us.size() == rewards.size()
            actor_loss = -(rewards/self.horizon).mean()

        self._writer.add_scalar('train_actor/loss', actor_loss.detach().cpu(), step)
        self._writer.add_scalar('train_actor/entropy', -first_log_p.detach().cpu().mean(), step)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.actor.log(self._writer, step)
        self.temp.update(first_log_p, self._writer, step)

        self._writer.add_scalar('train_alpha/value', self.temp.alpha.detach().cpu(), step)


    def update_critic(self, xs, xps, us, rs, not_done, step):
        assert xs.ndimension() == 2
        n_batch, _ = xs.size()
        rs = rs.squeeze()
        not_done = not_done.squeeze()

        with torch.no_grad():
            if not self.critic_target_mve or step < self.warmup_steps:
                mu, target_us, log_pi = self.actor.forward(
                    xps, compute_pi=True, compute_log_pi=True)
                log_pi = log_pi.squeeze(1) # type: ignore

                target_Q1, target_Q2 = [
                    Q.squeeze(1) for Q in self.critic_target(xps, target_us)]
                target_Q = torch.min(target_Q1, target_Q2) - self.temp.alpha.detach() * log_pi
                assert target_Q.size() == rs.size()
                assert target_Q.ndimension() == 1
                target_Q = rs + not_done * self.discount * target_Q
                target_Q = target_Q.detach()
            else:
                target_Q, first_log_p, total_log_p_us = self.expand_Q(
                    xps, self.critic_target, sample=True, discount=True)
                target_Q = target_Q - self.temp.alpha.detach() * first_log_p
                target_Q = rs + not_done * self.discount * target_Q
                target_Q = target_Q.detach()

        current_Q1, current_Q2 = [Q.squeeze(1) for Q in self.critic(xs, us)] #type: ignore
        assert current_Q1.size() == target_Q.size()
        assert current_Q2.size() == target_Q.size()
        Q_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        self._writer.add_scalar('train_critic/Q_loss', Q_loss.detach().cpu(), step)
        current_Q = torch.min(current_Q1, current_Q2)
        self._writer.add_scalar('train_critic/value', current_Q.detach().cpu().mean(), step)

        self.critic_opt.zero_grad()
        Q_loss.backward()
        self._writer.add_scalar('train_critic/Q_loss', Q_loss.detach().cpu(), step)
        self.critic_opt.step()

        self.critic.log(self._writer, step) # type: ignore

    def update_critic_mve(self, first_xs, first_us, first_rs, next_xs, first_not_dones, step):
        """ MVE critic loss from Feinberg et al (2015) """
        assert first_xs.dim() == 2
        assert first_us.dim() == 2
        assert first_rs.dim() == 2
        assert next_xs.dim() == 2
        assert first_not_dones.dim() == 2
        n_batch = next_xs.size(0)

        # unroll policy, concatenate obs and actions
        pred_us, log_p_us, pred_xs = self.dx.unroll_policy(
            next_xs, self.actor, sample=True, detach_xt=self.actor_detach_rho)
        all_obs = torch.cat((first_xs.unsqueeze(0), next_xs.unsqueeze(0), pred_xs))
        all_us = torch.cat([first_us.unsqueeze(0), pred_us])
        xu = torch.cat([all_obs, all_us], dim=2)
        horizon_len = all_obs.size(0) - 1  # H

        # get immediate rewards
        pred_rs = self.rew(xu[1:-1])  # t from 0 to H - 1
        rewards = torch.cat([first_rs.unsqueeze(0), pred_rs]).squeeze(2)
        rewards = rewards.unsqueeze(1).expand(-1, horizon_len, -1)
        log_p_us = log_p_us.unsqueeze(1).expand(-1, horizon_len, -1)

        # get not dones factor matrix, rows --> t, cols --> k
        first_not_dones = first_not_dones.unsqueeze(0)
        init_not_dones = torch.ones_like(first_not_dones)  # we know the first states are not terminal
        pred_not_dones = 1. - self.done(xu[2:]).sigmoid()  # type: ignore # t from 1 to H 
        not_dones = torch.cat([init_not_dones, first_not_dones, pred_not_dones]).squeeze(2)
        not_dones = not_dones.unsqueeze(1).repeat(1, horizon_len, 1)
        triu_rows, triu_cols = torch.triu_indices(row=horizon_len + 1, col=horizon_len, offset=1, device=not_dones.device)
        not_dones[triu_rows, triu_cols, :] = 1.
        not_dones = not_dones.cumprod(dim=0).detach()

        # get lower-triangular reward discount factor matrix
        discount = torch.tensor(self.discount, device=rewards.device)
        discount_exps = torch.stack([torch.arange(-i, -i + horizon_len) for i in range(horizon_len)], dim=1)
        r_discounts = discount ** discount_exps.to(rewards.device)
        r_discounts = r_discounts.tril().unsqueeze(-1)

        # get discounted sums of soft rewards (t from -1 to H - 1 (k from t to H - 1))
        alpha = self.temp.alpha.detach()
        soft_rewards = (not_dones[:-1] * rewards) - (discount * alpha * not_dones[1:] * log_p_us)
        soft_rewards = (r_discounts * soft_rewards).sum(0)

        # get target q-values, final critic targets
        target_q1, target_q2 = self.critic_target(all_obs[-1], all_us[-1])
        target_qs = torch.min(target_q1, target_q2).squeeze(-1).expand(horizon_len, -1)
        q_discounts = discount ** torch.arange(horizon_len, 0, step=-1).to(target_qs.device)
        target_qs = target_qs * (not_dones[-1] * q_discounts.unsqueeze(-1))
        critic_targets = (soft_rewards + target_qs).detach()

        # get predicted q-values
        with eval_mode(self.critic):
            q1, q2 = self.critic(all_obs[:-1].flatten(end_dim=-2), all_us[:-1].flatten(end_dim=-2)) # type: ignore
            q1, q2 = q1.reshape(horizon_len, n_batch), q2.reshape(horizon_len, n_batch)
        assert q1.size() == critic_targets.size()
        assert q2.size() == critic_targets.size()

        # update critics
        q1_loss = (not_dones[:-1, 0] * (q1 - critic_targets).pow(2)).mean()
        q2_loss = (not_dones[:-1, 0] * (q2 - critic_targets).pow(2)).mean()
        Q_loss = q1_loss + q2_loss

        self._writer.add_scalar('train_critic/Q_loss', Q_loss.detach().cpu(), step)
        current_Q = torch.min(q1, q2)
        self._writer.add_scalar.log('train_critic/value', current_Q.detach().cpu().mean(), step)

        self.critic_opt.zero_grad()
        Q_loss.backward()
        self._writer.add_scalar.log('train_critic/Q_loss', Q_loss.detach().cpu(), step)
        self.critic_opt.step()

        self.critic.log(self._writer, step) # type: ignore

    def update(self, replay_buffer, step):
        self.last_step = step
        if step % self.update_freq != 0:
            return

        if (self.horizon > 1 or not self.critic) and \
              (step % self.model_update_freq == 0) and \
              (self.actor_mve or self.critic_target_mve):
            for i in range(self.model_update_repeat):
                obses, actions, rewards = replay_buffer.sample_multistep(
                    self.seq_batch_size, self.seq_train_length)
                assert obses.ndimension() == 3
                dx_loss = self.dx.update_step(obses, actions, rewards, self._writer, step)
                if self.actor_dx_threshold is not None:
                    if self.rolling_dx_loss is None:
                        self.rolling_dx_loss = dx_loss
                    else:
                        factor = 0.9
                        self.rolling_dx_loss = factor*self.rolling_dx_loss + \
                          (1.-factor)*dx_loss

        n_updates = 1 if step < self.warmup_steps else self.model_free_update_repeat
        for i in range(n_updates):
            obs, action, reward, next_obs, not_done, not_done_no_max = \
              replay_buffer.sample(self.step_batch_size)

            if self.critic is not None:
                if self.full_target_mve:
                    self.update_critic_mve(obs, action, reward, next_obs, not_done_no_max, step)
                else:
                    self.update_critic(
                        obs, next_obs,
                        action, reward, not_done_no_max, step
                    )

            if step % self.actor_update_freq == 0:
                self.update_actor_and_alpha(obs, step)

            if self.rew_opt is not None:
                self.update_rew_step(obs, action, reward, step)

            self.update_done_step(obs, action, not_done_no_max, step)

            if self.critic is not None and step % self.critic_target_update_freq == 0:
                soft_update_params(self.critic, self.critic_target, self.critic_tau)


    def update_rew_step(self, obs, action, reward, step):
        assert obs.dim() == 2
        batch_size, _ = obs.shape

        xu = torch.cat((obs, action), dim=1)
        pred_reward = self.rew(xu)
        assert pred_reward.size() == reward.size()
        reward_loss = F.mse_loss(pred_reward, reward, reduction='mean')

        self.rew_opt.zero_grad()
        reward_loss.backward()
        self.rew_opt.step()

        self._writer.add_scalar('train_model/reward_loss', reward_loss.detach().cpu(), step)

    def update_done_step(self, obs, action, not_done, step):
        assert obs.dim() == 2
        batch_size, _ = obs.shape

        done = 1.-not_done
        xu = torch.cat((obs, action), dim=1)

        pred_logits = self.done(xu) # type: ignore
        n_done = torch.sum(done)
        if n_done > 0.:
            pos_weight = (batch_size - n_done) / n_done
        else:
            pos_weight = torch.tensor(1.)
        done_loss = F.binary_cross_entropy_with_logits(
            pred_logits, done, pos_weight=pos_weight,
            reduction='mean')

        self.done_opt.zero_grad()
        done_loss.backward()
        self.done_opt.step()

        self._writer.add_scalar('train_model/done_loss', done_loss.detach().cpu(), step)
    
    def sample_evaluation_action(self, states: torch.Tensor) -> np.ndarray:
        assert False
        return super().sample_evaluation_action(states)
    
    def sample_regular_action(self, states: torch.Tensor) -> np.ndarray:
        assert False
        return super().sample_regular_action(states)

    def train_policy(self) -> None:
        assert False
        return super().train_policy()

    def evaluate(self, normalise_obs, replay_buffer, step):
        episode_rewards = []
        for episode in range(10):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                with eval_mode(self):
                    if normalise_obs:
                        mu, sigma = replay_buffer.get_obs_stats()
                        obs_norm = (obs - mu) / sigma
                        action = self.act(obs_norm, sample=False)
                    else:
                        action = self.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
            episode_rewards.append(episode_reward)

            self._writer.add_scalar('eval/episode_reward', episode_reward, step)
        return np.mean(episode_rewards)

    def run(self) -> None:
        obs = self.env.reset()
        done = True
        episode_reward = 0
        start_time = time.time()
        steps_since_eval = 0
        EVAL_FREQ = 2000
        SEED_STEPS = 1000
        normalise_obs = self.normalise
        replay_buffer = ReplayBuffer(self._state_dim, self._action_dim, int(1e6), self.device, normalise_obs)

        while self.steps < self.num_train_steps:
            if done:
                self.log(f"Reward: {episode_reward}, time: {time.time() - start_time:.3f}")
                if self.steps > 0:
                    self._writer.add_scalar(
                            'train/episode_reward', episode_reward, self.episodes)
                    self._writer.add_scalar('train/duration',
                                    time.time() - start_time, self.steps)
                    self._writer.add_scalar('train/episode', self._episodes, self.steps)
                    start_time = time.time()
                
                if steps_since_eval > EVAL_FREQ:
                    self._writer.add_scalar('eval/episode', self._episodes, self.steps)
                    eval_mean = self.evaluate(normalise_obs, replay_buffer, self.steps)
                    self.log(f"Evaluated mean: {eval_mean}")
                    steps_since_eval = 0

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                self._episodes += 1
            
            if self.steps < SEED_STEPS:
                action = self.env.action_space.sample()
            else:
                with eval_mode(self):
                    if normalise_obs:
                        mu, sigma = replay_buffer.get_obs_stats()
                        obs_norm = (obs - mu) / sigma
                        action = self.act(obs_norm, sample=True)
                    else:
                        action = self.act(obs, sample=True) 
            
            if self.steps > SEED_STEPS: 
                self.update(replay_buffer, self.steps)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done_float = float(done)
            done_no_max = done_float if episode_step + 1 < self.env.spec.max_episode_steps else 0. # type: ignore
            episode_reward += reward

            replay_buffer.add(obs, action, reward, next_obs, done_float, done_no_max)

            obs = next_obs
            self._episode_steps += 1
            self._steps += 1
            steps_since_eval += 1

# agent = SVGDirectRipAgent("BipedalWalker-v3", "cuda", None, normalise=True)
# agent.run()