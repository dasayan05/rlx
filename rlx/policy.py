import abc
import torch, gym
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions import Categorical

from .utils import ActionDistribution

class Parametric(nn.Module):
    """
    Base class of the learnable component of an agent. It should contain Policy, Value etc.

    Required API:
        forward(states) -> ActionDistribution, others (Given states, returns ActionDistribution and other stuff)
        reset() -> None (Resets the internals of the learnable. Specifically required for RNNs)
    """

    def __init__(self, observation_space, action_spaces):
        ''' Constructs every learnable item from environment specifications. '''

        super().__init__()

        assert isinstance(action_spaces, tuple), "There must be a list (potentially singleton) of action spaces"

        self.observation_space = observation_space
        self.action_spaces = action_spaces

    @abc.abstractmethod
    def forward(self, state):
        pass

    def reset(self):
        pass

class DiscreteMLPPolicy(Parametric):
    """ Feed forward policy for discrete action space """

    def __init__(self, observation_space, action_spaces, *, n_hidden=128):
        ''' Constructs every learnable item from environment specifications. '''

        super().__init__(observation_space, action_spaces)
        assert len(self.action_spaces) == 1, "Only one action component"
        assert isinstance(self.action_spaces[0], gym.spaces.Discrete), "Only discrete action allowed"

        # Track arguments for further use
        self.n_state = self.observation_space.shape[0]
        self.n_action = self.action_spaces[0].n
        self.n_hidden = n_hidden

        # Layer definitions
        self.affine = torch.nn.Linear(self.n_state, self.n_hidden)
        self.pi = torch.nn.Linear(self.n_hidden, self.n_action)

    def forward(self, state):
        h = F.relu(self.affine(state.unsqueeze(0)))
        act = Categorical(F.softmax(self.pi(h), dim=-1))
        return ActionDistribution(act), None

class DiscreteMLPPolicyValue(Parametric):
    """ Feed forward (policy + value) for discrete action space """

    def __init__(self, observation_space, action_spaces, *, n_hidden=128):
        ''' Constructs every learnable item from environment specifications. '''

        super().__init__(observation_space, action_spaces)
        assert len(self.action_spaces) == 1, "Only one action component"
        assert isinstance(self.action_spaces[0], gym.spaces.Discrete), "Only discrete action allowed"

        # Track arguments for further use
        self.n_state = self.observation_space.shape[0]
        self.n_action = self.action_spaces[0].n
        self.n_hidden = n_hidden

        # Layer definitions
        self.affine = torch.nn.Linear(self.n_state, self.n_hidden)
        self.pi = torch.nn.Linear(self.n_hidden, self.n_action)
        self.value = torch.nn.Linear(self.n_hidden, 1)

    def forward(self, state):
        h = F.relu(self.affine(state.unsqueeze(0)))
        act = Categorical(F.softmax(self.pi(h), dim=-1))
        v = self.value(h)
        return ActionDistribution(act), v

class DiscreteRNNPolicy(Parametric):
    """ Recurrent policy for discrete action space """

    def __init__(self, observation_space, action_spaces, *, n_hidden=128):
        ''' Constructs every learnable item from environment specifications. '''

        super().__init__(observation_space, action_spaces)
        assert len(self.action_spaces) == 1, "Only one action component"
        assert isinstance(self.action_spaces[0], gym.spaces.Discrete), "Only discrete action allowed"

        # Track arguments for further use
        self.n_state = self.observation_space.shape[0]
        self.n_action = self.action_spaces[0].n
        self.n_hidden = n_hidden

        # Layer definitions
        self.cell, self.h = torch.nn.GRUCell(self.n_state, self.n_hidden), None
        self.pi = torch.nn.Linear(self.n_hidden, self.n_action)
    
    def forward(self, state):
        self.h = self.cell(state.unsqueeze(0), self.h)
        act = Categorical(F.softmax(self.pi(self.h), dim=-1))
        return ActionDistribution(act), None

    def reset(self):
        self.h = None

class DiscreteRNNPolicyValue(Parametric):
    """ Recurrent (policy + value) for discrete action space """

    def __init__(self, observation_space, action_spaces, *, n_hidden=128):
        ''' Constructs every learnable item from environment specifications. '''

        super().__init__(observation_space, action_spaces)
        assert len(self.action_spaces) == 1, "Only one action component"
        assert isinstance(self.action_spaces[0], gym.spaces.Discrete), "Only discrete action allowed"

        # Track arguments for further use
        self.n_state = self.observation_space.shape[0]
        self.n_action = self.action_spaces[0].n
        self.n_hidden = n_hidden

        # Layer definitions
        self.cell, self.h = torch.nn.GRUCell(self.n_state, self.n_hidden), None
        self.pi = torch.nn.Linear(self.n_hidden, self.n_action)
        self.V = torch.nn.Linear(self.n_hidden, 1)
    
    def forward(self, state):
        self.h = self.cell(state.unsqueeze(0), self.h)
        act = Categorical(F.softmax(self.pi(self.h), dim=-1))
        v = self.V(self.h)
        return ActionDistribution(act), v

    def reset(self):
        self.h = None