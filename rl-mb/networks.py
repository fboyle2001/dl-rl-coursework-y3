import torch.nn as nn

class Critic(nn.Sequential):
    def __init__(self, input_size, output_size, connected_size=256):
        super(Critic, self).__init__(
            nn.Linear(input_size, connected_size),
            nn.ReLU(),
            nn.Linear(connected_size, connected_size),
            nn.ReLU(),
            nn.Linear(connected_size, connected_size),
            nn.ReLU(),
            nn.Linear(connected_size, output_size)
        )

class Actor(nn.Sequential):
    def __init__(self, input_size, output_size, connected_size=256):
        super(Actor, self).__init__(
            nn.Linear(input_size, connected_size),
            nn.ReLU(),
            nn.Linear(connected_size, connected_size),
            nn.ReLU(),
            nn.Linear(connected_size, connected_size),
            nn.ReLU(),
            nn.Linear(connected_size, output_size),
            nn.Tanh()
        )