import config
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(config.N_STATE_FEATURES, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, config.N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)
