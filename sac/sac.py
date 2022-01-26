class StandardSACAgent:
    def __init__(self, state_dim, action_dim, device: Union[str, torch.device], parameters: DotMap, env):
        """
        Required parameters:
        buffer_size => defines max number of experiences able to hold in memory
        gradient_steps
        lr
        tau
        replay_batch_size
        start_steps
        gradient_update_steps
        """
        self.parameters = parameters
        self.device = device
        self.env = env

        # Replay buffer
        self.replay_buffer = buffers.StandardReplayBuffer(state_dim, action_dim, self.parameters.buffer_size, self.device)

        # Critics predicts the reward of taking action A from state S
        self.critic_1 = networks.Critic(input_size=state_dim + action_dim, output_size=1).to(self.device)
        self.critic_2 = networks.Critic(input_size=state_dim + action_dim, output_size=1).to(self.device)

        # Actor predicts the action to take based on the current state
        self.actor = networks.SACGaussianPolicy(input_size=state_dim, output_size=action_dim).to(self.device)

        # Establish the target networks
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)

        # Disable gradient calculations for targets?
        for param in self.target_critic_1.parameters():
            param.requires_grad = False

        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)

        for param in self.target_critic_2.parameters():
            param.requires_grad = False

        # Establish optimisers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.parameters.lr)
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=self.parameters.lr)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=self.parameters.lr)

        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.parameters.lr)

        self.steps = 0
        self.episodes = 0
        self.learning_steps = 0

        self.log_dir = f"./TBLogs/{time.time()}/tensorboard"
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def run(self):
        while True:
            self.train_episode()
    
    def act(self, state):
        if self.parameters.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state)

        return action

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()

        while not done:
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            self.store_memory(state, action, reward, next_state, done)

            if self.replay_buffer.count > self.parameters.replay_batch_size and self.steps > self.parameters.start_steps:
                for _ in range(self.parameters.gradient_update_steps):
                    self.learn()
            
            state = next_state
        
        print(f"Episode {self.episodes}, steps {episode_steps}, episode_reward {episode_reward}, alpha {self.alpha}")
    
    def learn(self):
        self.learning_steps += 1

        if self.learning_steps % self.parameters.target_update_interval == 0:
            self.update_target_parameters()

        buffer_sample = self.replay_buffer.sample_buffer(self.parameters.replay_batch_size)

        actual_input = torch.cat([buffer_sample.states, buffer_sample.actions], dim=-1)
        actual_Q1 = self.critic_1(actual_input)
        actual_Q2 = self.critic_2(actual_input)

        with torch.no_grad():
            target_actions, target_entropies, _ = self.actor.sample(buffer_sample.next_states)

            target_input = torch.cat([buffer_sample.next_states, target_actions], dim=-1)
            target_Q1 = self.target_critic_1(target_input)
            target_Q2 = self.target_critic_2(target_input)

            min_Q_target = torch.min(target_Q1, target_Q2)
            next_q = min_Q_target + self.alpha * target_entropies

        # Temporal difference learning ish (tab back?)
        target_Q = buffer_sample.rewards.unsqueeze(1) + buffer_sample.terminals.unsqueeze(1) * self.parameters.gamma * next_q

        #print(target_Q.detach().mean())

        mean_q1 = actual_Q1.detach().mean().item()
        q1_critic_loss = torch.mean((actual_Q1 - target_Q).pow(2)) # F.mse_loss(actual_Q1, target_Q)
        #print("Q1CL", q1_critic_loss.detach())
        mean_q2 = actual_Q2.detach().mean().item()
        q2_critic_loss = torch.mean((actual_Q2 - target_Q).pow(2)) # F.mse_loss(actual_Q2, target_Q)
        #print("Q2CL", q2_critic_loss.detach())

        self.critic_1_opt.zero_grad()
        q1_critic_loss.backward()
        self.critic_1_opt.step()

        self.critic_2_opt.zero_grad()
        q2_critic_loss.backward()
        self.critic_2_opt.step()

        sampled_action, entropy, _ = self.actor.sample(buffer_sample.states)
        real_input = torch.cat([buffer_sample.states, sampled_action], dim=-1)
        q1 = self.critic_1(real_input)
        q2 = self.critic_2(real_input)
        q = torch.min(q1, q2)

        actor_loss = torch.mean(-q - self.alpha * entropy)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        entropy_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropy).detach())
        #print("EL", entropy_loss.detach())

        self.alpha_optim.zero_grad()
        entropy_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        self.writer.add_scalar("loss/alpha", entropy_loss.detach().item(), self.steps)

        if self.learning_steps % self.parameters.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_critic_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_critic_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', actor_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps),
            self.writer.add_scalar(
                'stats/entropy', entropy.detach().mean().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)

    def explore(self, state):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.actor.sample(state)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state):
        # act without randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, action = self.actor.sample(state)
        return action.cpu().numpy().reshape(-1)
    
    def store_memory(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, is_terminal: bool) -> None:
        self.replay_buffer.store_replay(state, action, reward, next_state, is_terminal)
    
    def update_target_parameters(self):
        # Update frozen targets, taken from https://github.com/sfujim/TD3/blob/385b33ac7de4767bab17eb02ade4a268d3e4e24f/TD3.py
        # theta' = tau * theta + (1 - tau) * theta'
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.parameters.tau * param.data + (1 - self.parameters.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.parameters.tau * param.data + (1 - self.parameters.tau) * target_param.data)

    """def _pseudo_train(self) -> None:
        for episode in range(episodes):
            while not done:
                sample action from the policy
                take step in environment using the action
                store the transition in the buffer
            
            for gradient steps
                update the q functions
                update the policy weights
                update alpha temperature
                soft update target networks"""

def train(self) -> None:
        super().train()
        
        if self.replay_buffer.count < self.replay_batch_size:
            return

        # Sample the replay buffer
        buffer_sample = self.replay_buffer.sample_buffer(self.replay_batch_size)

        # TD3 
        with torch.no_grad():
            noise = self.noise_distribution.sample(buffer_sample.actions.shape).clamp(-self.noise_clip, self.noise_clip).to(self.device)
            target_actions = self.target_actor(buffer_sample.next_states)
            target_actions = (target_actions + noise).clamp(-1, 1)

            target_input = torch.cat([buffer_sample.next_states, target_actions], dim=-1)

            target_Q1 = self.target_critic_1(target_input)
            target_Q2 = self.target_critic_2(target_input)
            min_Q_target = torch.min(target_Q1, target_Q2)

            # Temporal difference learning
            target_Q = buffer_sample.rewards.unsqueeze(1) + buffer_sample.terminals.unsqueeze(1) * self.gamma * min_Q_target 

        actual_input = torch.cat([buffer_sample.states, buffer_sample.actions], dim=-1)

        actual_Q1 = self.critic_1(actual_input)
        actual_Q2 = self.critic_2(actual_input)

        critic_loss = F.mse_loss(actual_Q1, target_Q) + F.mse_loss(actual_Q2, target_Q)

        self.critic_1_opt.zero_grad()
        self.critic_2_opt.zero_grad()

        critic_loss.backward()

        self.critic_1_opt.step()
        self.critic_2_opt.step()

        # Delayed policy updates
        if self._steps % self.policy_update_frequency == 0:
            actor_input = torch.cat([buffer_sample.states, self.actor(buffer_sample.states)], dim=-1)
            actor_loss = -self.critic_1(actor_input).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.update_target_parameters()