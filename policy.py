import torch.nn as nn
from torch.distributions import Categorical
from models import PolicyNetwork, ValueNetwork

class DiscreteMLPPolicyValue(nn.Module):
    def __init__(self, observation_space, action_space, device=None):
        super().__init__()

        self.n_states = observation_space.shape[0]
        self.n_actions = action_space.n
        if device is None:
            self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.policynet = PolicyNetwork(self.n_states, self.n_actions).to(self.device)
        self.valuenet = ValueNetwork(self.n_states).to(self.device)

    def __call__(self, state):
        return self.valuenet(state), Categorical(self.policynet(state))

class DiscreteMLPPolicy(nn.Module):
    def __init__(self, observation_space, action_space, device=None):
        super().__init__()

        self.n_states = observation_space.shape[0]
        self.n_actions = action_space.n
        if device is None:
            self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.policynet = PolicyNetwork(self.n_states, self.n_actions).to(self.device)

    def __call__(self, state):
        return Categorical(self.policynet(state))