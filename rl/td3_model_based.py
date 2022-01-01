import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

# Will need to take in no of vars in state which is the obs_dim and the action which is act_dim
# It will output the next state and the reward of that action from the original state
class EnvModel(nn.Module):
    def __init__(self, input_size, output_size, connected_size=256):
        super(EnvModel, self).__init__()

        # Reused the actor model
        self.fc1 = nn.Linear(input_size, connected_size)
        self.fc2 = nn.Linear(connected_size, connected_size)
        self.output = nn.Linear(connected_size, output_size)

    def forward(self, state):
        step = F.relu(self.fc1(state))
        step = F.relu(self.fc2(step))
        out = self.output(step)

        return out
        
