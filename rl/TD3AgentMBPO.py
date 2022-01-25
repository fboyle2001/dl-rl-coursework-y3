#@title Reinforcement learning agent (TD3) with PRB and MBPO

class SingleDynamicsModel(nn.Module):
    def __init__(self, input_size, output_size, connected_size=256):
        super(SingleDynamicsModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, connected_size),
            nn.ReLU(),
            nn.Linear(connected_size, connected_size),
            nn.ReLU(),
            nn.Linear(connected_size, connected_size),
            nn.ReLU(),
            nn.Linear(connected_size, output_size)
        )
    
    def forward(self, states_actions):
        output = self.layers(states_actions)
        # Split into delta states and rewards
        return output[:, :-1], output[:, -1]

class DynamicsModel(nn.Module):
    """
    Each member of the ensemble is a probabilistic neural network whose outputs parametrise
    a Gaussian distribution with diagonal covariance ...
    To generate a prediction from the ensemble we simply select a model uniformly at random
    allowing for different transitions along a single model rollout to be sampled from different
    dynamics models
    [Cite https://arxiv.org/pdf/1906.08253.pdf]
    """
    def __init__(self, input_size, output_size, connected_size=256):
        super(DynamicsModel, self).__init__()

    def train(self, inputs, labels, batch_size, holdout_ratio, max_epochs_since_update):
        """
        num_holdouts = holdout_ratio * inputs.shape[0]
        randomly shuffle inputs (and match their labels)
        partition into training inputs[:num_holdouts], eval inputs[:num_holdouts]
        fit standard scaler to training data
        scale both training data and eval data

        """

"""
The dynamics model needs to take a state and action in and output the next state and reward (also output done but not sure I really care?)
To train the dynamics model need to provide 
"""

# Weighted Importance Sampling Mean Squared Error 
# Cite https://proceedings.neurips.cc/paper/2014/file/be53ee61104935234b174e62a07e53cf-Paper.pdf
def wis_mse(input, target, weights):
    return (weights * (target - input).square()).mean()

# MBPO Hyperparameters - TODO: Temp will externalise on Colab
dynamics_ensemble_size = 7 # B
policy_updates_per_env_step = 20 # G
model_horizon = lambda epoch: int(min(max(1 + (25 - 1) * (epoch - 20) / (100 - 20), 1), 25)) # k
real_ratio = 0.05 # r
dynamics_train_frequency = 5000 # T
rollout_size = 400 # M

class TD3AgentMBPO:
    def __init__(self, state_dim, action_dim, device):
        self.device = device

        # Prioritised Replay Buffer
        self.env_replay_buffer = PrioritisedReplayBuffer(state_dim, action_dim, 20, self.device)
        # Think this now right? TODO: What size does this need to be?? Should this be a priority buffer??
        self.model_replay_buffer = ReplayBuffer(state_dim, action_dim, rollout_size * model_horizon, self.device)

        # Output is next state and the reward
        self.dynamics_model = SingleDynamicsModel(input_size=state_dim + action_dim, output_size=state_dim + 1)
        self.dynamics_opt = torch.optim.Adam(self.dynamics_model.parameters(), lr=learning_rate)

        # Critics predicts the reward of taking action A from state S
        self.critic_1 = Critic(input_size=state_dim + action_dim, output_size=1).to(self.device)
        self.critic_2 = Critic(input_size=state_dim + action_dim, output_size=1).to(self.device)

        # Actor predicts the action to take based on the current state
        self.actor = Actor(input_size=state_dim, output_size=action_dim).to(self.device)

        # Establish the target networks
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_1.eval()
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)
        self.target_critic_2.eval()
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_actor.eval()

        # Establish optimisers
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=learning_rate)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=learning_rate)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

        # Establish sampling distributions
        self.noise_distribution = torch.distributions.normal.Normal(0, noise_sigma)

        self.steps = 0

    def store_memory(self, state, action, reward, next_state, is_terminal):
        self.env_replay_buffer.store_replay(state, action, reward, next_state, is_terminal, None)

    def _train_dynamics_model(self, states_actions, delta_states, rewards):
        # TODO: May need to unsqueeze?
        labels = torch.cat([delta_states, rewards], dim=-1)
        predicted_delta_states, predicted_rewards = self.dynamics_model(states_actions)
        loss = F.mse_loss(torch.cat([predicted_delta_states, predicted_rewards.unsqueeze(-1)], dim=-1), labels)
        dynamics_opt.zero_grad()
        loss.backward()
        dynamics_opt.step()

    def predict_dynamics(self, states, actions):
        states_actions = torch.cat([states, actions], dim=-1)

        with torch.no_grad():
            predicted_delta_states, predicted_rewards = self.dynamics_model(states_actions)

        next_states = states + predicted_delta_states
        return next_states, predicted_rewards

    def update_target_parameters(self):
        # Update frozen targets, taken from https://github.com/sfujim/TD3/blob/385b33ac7de4767bab17eb02ade4a268d3e4e24f/TD3.py
        # theta' = tau * theta + (1 - tau) * theta'
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def sample_action(self, s):
        # Given state S what is the action A to take?
            s = s.reshape(1, -1)
            state = torch.FloatTensor(s).to(self.device)
            # state = F.Tensor(s, dtype=float).to(self.device)
            return self.actor(state).cpu().data.numpy().flatten()

    def train(self, t):
        self.steps += 1

        # Let the replay buffer fill up enough to be sampled from
        if self.env_replay_buffer.current_buffer_size < replay_batch_size:
            return

        # Sample the replay buffer
        buffer_sample, buffer_indexes, buffer_weights = self.env_replay_buffer.sample_buffer(replay_batch_size)

        # t or self.steps?? Surely self.steps since otherwise we will never reach t?
        if self.steps % dynamics_train_frequency == 0:
            # Train the dynamics model on ALL the environment replays
            dynamics_sample = self.env_replay_buffer.get_all_replays()
            dynamics_delta_states = dynamics_sample.next_states - dynamics_sample.states
            dynamics_input = torch.cat([dynamics_sample.states, dynamics_sample.actions], dim=-1)
            self._train_dynamics_model(dynamics_input, dynamics_delta_states, dynamics_sample.rewards)

        # TD3 
        with torch.no_grad():
            noise = self.noise_distribution.sample(buffer_sample.actions.shape).clamp(-noise_clip, noise_clip)
            target_actions = self.target_actor(buffer_sample.next_states)
            target_actions = (target_actions + noise).clamp(-1, 1)

            target_input = torch.cat([buffer_sample.next_states, target_actions], dim=-1)

            target_Q1 = self.target_critic_1(target_input)
            target_Q2 = self.target_critic_2(target_input)
            min_Q_target = torch.min(target_Q1, target_Q2)

            # Temporal difference learning
            target_Q = buffer_sample.rewards.unsqueeze(1) + buffer_sample.terminals.unsqueeze(1) * gamma * min_Q_target 

        actual_input = torch.cat([buffer_sample.states, buffer_sample.actions], dim=-1)

        actual_Q1 = self.critic_1(actual_input)
        actual_Q2 = self.critic_2(actual_input)

        # print("Buffer weights:", buffer_weights)

        critic_loss = wis_mse(actual_Q1, target_Q, buffer_weights) + wis_mse(actual_Q2, target_Q, buffer_weights)
        # print("Critic Loss:", critic_loss.detach().cpu())

        self.critic_1_opt.zero_grad()
        self.critic_2_opt.zero_grad()

        # Maybe add some gradient clipping?
        critic_loss.backward()

        self.critic_1_opt.step()
        self.critic_2_opt.step()

        # Use the TD errors to update the priorities in the replay buffer
        td_errors = torch.abs(target_Q - actual_Q1).detach().cpu().numpy()
        self.env_replay_buffer.update_priorities(buffer_indexes, td_errors)

        # Model rollouts
        with torch.no_grad():
            rollout_states, _, _ = self.env_replay_buffer.sample_buffer(rollout_size)

            for rollout_step in range(model_horizon):
                rollout_actions = self.actor(rollout_states)
                r_next_states, r_rewards = self.predict_dynamics(rollout_states, rollout_actions) # TODO: Define the dynamics model, what about terminal?
                is_terminals = torch.zeros(rollout_states.shape[0], dtype=torch.bool).to(self.device)
                self.model_replay_buffer.store_replays(rollout_states, rollout_actions, r_rewards, r_next_states, is_terminals)

        # Delayed policy updates
        if self.steps % policy_update_frequency == 0:
            for policy_update in range(policy_updates_per_env_step):
                real_sample_count = int(real_ratio * replay_batch_size)
                fake_sample_count = replay_batch_size - real_samples

                # Update using a mix of fake and real samples
                real_samples, _, _ = self.env_replay_buffer.sample_buffer(real_sample_count)
                fake_samples, _, _ = self.model_replay_buffer.sample_buffer(fake_sample_count)

                print(f"RS shape, {real_samples.shape}, FS shape, {fake_samples.shape}") # Check

                joined_states = torch.cat([real_samples.states, fake_samples.states], dim=0)
                joined_states = joined_states[torch.randperm(joined_states.shape[0])] # TODO: Is this on the correct dimension??
                policy_actions = self.actor(joined_states)

                actor_input = torch.cat([joined_states, policy_actions], dim=-1)
                actor_loss = -self.critic_1(actor_input).mean()

                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()

                self.update_target_parameters()

