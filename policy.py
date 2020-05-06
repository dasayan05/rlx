import abc
import torch.nn as nn
from torch.distributions import Categorical
from models import FFPolicyNetwork, FFValueNetwork, RNNPolicyNetwork, RNNPolicyValueNetwork

class Policy(nn.Module):
    def __init__(self, observation_space, action_space, device=None):
        super().__init__()

        self.n_states = observation_space.shape[0]
        self.n_actions = action_space.n
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu') if device is None else device

    @abc.abstractmethod
    def forward(self, state):
        pass

    def reset(self):
        pass

class DiscreteMLPPolicyValue(Policy):
    def __init__(self, observation_space, action_space, device=None):
        super().__init__(observation_space, action_space, device=device)

        self.policynet = FFPolicyNetwork(self.n_states, self.n_actions).to(self.device)
        self.valuenet = FFValueNetwork(self.n_states).to(self.device)

    def forward(self, state):
        return self.valuenet(state), Categorical(self.policynet(state))

    def parameters(self):
        return [*self.policynet.parameters(), *self.valuenet.parameters()]

class DiscreteMLPPolicy(Policy):
    def __init__(self, observation_space, action_space, device=None):
        super().__init__(observation_space, action_space, device=device)

        self.policynet = FFPolicyNetwork(self.n_states, self.n_actions).to(self.device)

    def forward(self, state):
        return Categorical(self.policynet(state))

    def parameters(self):
        return self.policynet.parameters()

class DiscreteRNNPolicy(Policy):
    def __init__(self, observation_space, action_space, device=None):
        super().__init__(observation_space, action_space, device=device)

        self.policynet = RNNPolicyNetwork(self.n_states, self.n_actions).to(self.device)
    
    def forward(self, state):
        return Categorical(self.policynet(state))

    def reset(self):
        self.policynet.h = None

class DiscreteRNNPolicyValue(Policy):
    def __init__(self, observation_space, action_space, device=None):
        super().__init__(observation_space, action_space, device=device)

        self.policynet = RNNPolicyValueNetwork(self.n_states, self.n_actions).to(self.device)
    
    def forward(self, state):
        action_probs, val = self.policynet(state)
        return val, Categorical(action_probs)

    def reset(self):
        self.policynet.h = None
