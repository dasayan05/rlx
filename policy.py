import abc
import torch.nn as nn
from torch.distributions import Categorical
from models import PolicyNetwork, ValueNetwork

class Policy(nn.Module):
    def __init__(self, observation_space, action_space, device=None):
        super().__init__()

        self.n_states = observation_space.shape[0]
        self.n_actions = action_space.n
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu') if device is None else device

    @abc.abstractmethod
    def forward(self, state):
        pass

class DiscreteMLPPolicyValue(Policy):
    def __init__(self, observation_space, action_space, device=None):
        super().__init__(observation_space, action_space, device=device)

        self.policynet = PolicyNetwork(self.n_states, self.n_actions).to(self.device)
        self.valuenet = ValueNetwork(self.n_states).to(self.device)

    def forward(self, state):
        return self.valuenet(state), Categorical(self.policynet(state))

class DiscreteMLPPolicy(Policy):
    def __init__(self, observation_space, action_space, device=None):
        super().__init__(observation_space, action_space, device=device)

        self.policynet = PolicyNetwork(self.n_states, self.n_actions).to(self.device)

    def forward(self, state):
        return Categorical(self.policynet(state))