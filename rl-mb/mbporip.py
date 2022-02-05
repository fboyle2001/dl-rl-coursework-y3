from agents import RLAgent
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import itertools
from operator import itemgetter
import random

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
device = "cuda:0"
BATCH_SIZE = 100

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

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class StandardScaler(object):
    def __init__(self):
        pass

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.
        Arguments:
        data (np.ndarray): A numpy array containing the input
        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.
        Arguments:
        data (np.array): A numpy array containing the points to be transformed.
        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.
        Arguments:
        data (np.array): A numpy array containing the points to be transformed.
        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu

def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)

class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.parameter.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class EnsembleModel(nn.Module):
    def __init__(self, state_size, action_size, reward_size, ensemble_size, hidden_size=200, learning_rate=1e-3, use_decay=False):
        super(EnsembleModel, self).__init__()
        self.hidden_size = hidden_size
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.use_decay = use_decay

        self.output_dim = state_size + reward_size
        # Add variance output
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)

        self.max_logvar = nn.parameter.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.parameter.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self.swish = Swish()

    def forward(self, x, ret_log_var=False):
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)

        mean = nn5_output[:, :, :self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
                # print(m.weight.shape)
                # print(m, decay_loss, m.weight_decay)
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        # print('loss:', loss.item())
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.shape, torch.mean(param.grad), param.grad.flatten()[:5])
        self.optimizer.step()

class EnsembleDynamicsModel():
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200, use_decay=False):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.ensemble_model = EnsembleModel(state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay).to(device)
        self.scaler = StandardScaler()

    def train(self, inputs, labels, batch_size=256, holdout_ratio=0., max_epochs_since_update=5):
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])
        last_hml = None

        for epoch in itertools.count():

            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)])
            # train_idx = np.vstack([np.arange(train_inputs.shape[0])] for _ in range(self.network_size))
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos: start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)
                losses = []
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.train(loss)
                losses.append(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, ret_log_var=True)
                _, holdout_mse_losses = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False)
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break
            
            last_hml = holdout_mse_losses
            # print('epoch: {}, holdout mse losses: {}'.format(epoch, holdout_mse_losses))
        
        print(f"HML {last_hml}")

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, inputs, batch_size=1024, factored=True):
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(device)
            b_mean, b_var = self.ensemble_model(input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False)
            ensemble_mean.append(b_mean.detach().cpu().numpy())
            ensemble_var.append(b_var.detach().cpu().numpy())
        ensemble_mean = np.hstack(ensemble_mean)
        ensemble_var = np.hstack(ensemble_var)

        if factored:
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
            return mean, var

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[:len(self.buffer) - self.position]
            self.buffer[:len(batch) - len(self.buffer) + self.position] = batch[len(self.buffer) - self.position:]
            self.position = len(batch) - len(self.buffer) + self.position

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, int(batch_size))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_all_batch(self, batch_size):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def return_all(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)

class SAC(object):
    def __init__(self, num_inputs, action_space):
        # torch.autograd.set_detect_anomaly(True)
        self.gamma = 0.99 
        self.tau = 0.005
        self.alpha = 0.2
        self.lr = 0.0003

        self.target_update_interval = 1 
        self.automatic_entropy_tuning = True
        self.q_hidden_size = 256

        self.device = torch.device("cuda")

        self.critic = QNetwork(num_inputs, action_space.shape[0], self.q_hidden_size).to(device=self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], self.q_hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

            # if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], self.q_hidden_size, action_space).to(self.device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # else:
        #     self.alpha = 0
        #     self.automatic_entropy_tuning = False
        #     # self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        #     assert False
        #     self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, updates):
        # Sample a batch from memory
        # state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.critic_optim.zero_grad()
        (qf1_loss+qf2_loss).backward()
        self.critic_optim.step()

        # self.critic_optim.zero_grad()
        # qf2_loss.backward()
        # self.critic_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

class RippedMBPOAgent(RLAgent):
    def __init__(self, env_name, d, video_every):
        super().__init__("RippedMBPO", env_name, d, video_every)
        self.num_networks = 7
        self.num_elites = 5
        self.reward_size = 1
        self.pred_hidden_size = 200
        self.use_decay = True

        self.buffer_size = 1000000
        self.model_train_freq = 250
        self.rollout_length = 1
        self.real_ratio = 0.05
        self.min_env_pool = 1000
        self.train_every_n_steps = 1
        self.max_train_repeat_per_step = 5
        self.num_train_repeat = 20
        self.policy_train_batch_size = 256
        self.rollout_batch_size = 100000
        self.model_retain_epochs = 1
        self.init_exploration_steps = 5000

        self.replay_buffer = ReplayMemory(self.buffer_size)
        self.rollouts_per_epoch = self.rollout_batch_size * self.env.spec.max_episode_steps / self.model_train_freq # type: ignore
        self.model_steps_per_epoch = int(self.rollout_length * self.rollouts_per_epoch)
        self.model_buffer = ReplayMemory(self.model_steps_per_epoch * self.model_retain_epochs)

        self.env_model = EnsembleDynamicsModel(self.num_networks, self.num_elites, self._state_dim, self._action_dim, self.reward_size, self.pred_hidden_size, use_decay=self.use_decay)
        self.done = mlp(self._state_dim + self._action_dim, 512, 1, 2).to(self.device)
        self.done_opt = torch.optim.Adam(self.done.parameters(), lr=1e-3)

        self.agent = SAC(self._state_dim, self.env.action_space)

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

    def _model_get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    def _model_step(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.env_model.predict(inputs)
        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        model_idxes = np.random.choice(self.env_model.elite_model_idxes, size=batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        model_means = ensemble_model_means[model_idxes, batch_idxes]
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        log_prob, dev = self._model_get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:, :1], samples[:, 1:]
        joined_terms = torch.tensor(np.concatenate([obs, act], axis=1), device=self.device)
        terminals = self.done(joined_terms).detach().cpu().numpy()

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:, :1], terminals, model_means[:, 1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:, :1], np.zeros((batch_size, 1)), model_stds[:, 1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, info

    def train_policy(self, episode_train_steps: int) -> int:
        if self.steps % self.train_every_n_steps == 0:
            return 0

        if episode_train_steps > self.max_train_repeat_per_step * self.steps:
            return 0
    
        for i in range(self.num_train_repeat):
            env_batch_size = int(self.policy_train_batch_size * self.real_ratio)
            model_batch_size = self.policy_train_batch_size - env_batch_size

            env_state, env_action, env_reward, env_next_state, env_done = self.replay_buffer.sample(int(env_batch_size))

            if model_batch_size > 0 and len(self.model_buffer) > 0:
                model_state, model_action, model_reward, model_next_state, model_done = self.model_buffer.sample_all_batch(int(model_batch_size))
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                                                                                        np.concatenate((env_action, model_action),
                                                                                                    axis=0), np.concatenate(
                    (np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                                                                                        np.concatenate((env_next_state, model_next_state),
                                                                                                    axis=0), np.concatenate(
                    (np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
            else:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

            batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
            batch_done = (~batch_done).astype(int)

            ## Now 
            self.agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), i)

        return 0

    def sample_evaluation_action(self, states: torch.Tensor) -> np.ndarray:
        return self.agent.select_action(states, eval=True)

    def sample_regular_action(self, states: torch.Tensor) -> np.ndarray:
        return self.agent.select_action(states, eval=False)

    def explore_before_start(self):
        cur_state = self.env.reset()
        done = False

        for i in range(self.init_exploration_steps):
            action = self.sample_regular_action(cur_state)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(cur_state, action, reward, next_state, done)
            cur_state = next_state

            if done:
                done = False
                cur_state = self.env.reset()
        
        while not done:
            action = self.sample_regular_action(cur_state)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(cur_state, action, reward, next_state, done)
            cur_state = next_state

    def train_predict_model(self):
        # Get all samples from environment
        state, action, reward, next_state, done = self.replay_buffer.sample(len(self.replay_buffer))
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

        self.env_model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)

    def rollout_model(self):
        state, action, reward, next_state, done = self.replay_buffer.sample_all_batch(self.rollout_batch_size)
        for i in range(self.rollout_length):
            # TODO: Get a batch of actions
            action = self.sample_regular_action(torch.tensor(state, device="cpu"))

            if torch.is_tensor(action):
                action = action.detach().cpu().numpy()
            
            if torch.is_tensor(state):
                state = action.detach().cpu().nump()
            # action = torch.tensor(action, device=self.device)
            next_states, rewards, terminals, info = self._model_step(state, action)
            # TODO: Push a batch of samples
            self.model_buffer.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
            terminals = (terminals > .5).astype(int)
            nonterm_mask = ~terminals.squeeze(-1)
            if nonterm_mask.sum() == 0:
                break
            state = next_states[nonterm_mask]

    def run(self) -> None:
        self.explore_before_start()
        steps_since_eval = 0

        while self.episodes < 1000:
            obs = self.env.reset()
            self._episode_steps = 0
            done = False
            episode_reward = 0
            episode_train_steps = 0

            while not done:
                if self._episode_steps % self.model_train_freq == 0:
                    self.train_predict_model()
                    # self.set_rollout_length()
                    self.rollout_model()

                action = self.sample_regular_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                episode_reward += reward

                stored_done = done if self._episode_steps + 1 < self.env.spec.max_episode_steps else False # type: ignore

                self.replay_buffer.push(obs, action, reward, next_obs, stored_done)

                if len(self.replay_buffer) > self.min_env_pool:
                    episode_train_steps += self.train_policy(episode_train_steps)
                    self._writer.add_scalar("train/policy_train_steps", episode_train_steps, self.steps)

                self._episode_steps += 1
                self._steps += 1
                steps_since_eval += 1

                obs = next_obs

            self.log(f"Episode reward {episode_reward}")
            self._writer.add_scalar("reward/reg", episode_reward, self.episodes)
            
            if steps_since_eval % 8000 == 0:
                avg_reward = 0

                for _ in range(10):
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
                
                avg_reward /= 10
                self.log(f"Average Reward for evaluation: {avg_reward}")
                self._writer.add_scalar("reward/eval", avg_reward, self.episodes)

            self._episodes += 1
            
            