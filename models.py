import torch
import torch.nn.functional as F

class FFPolicyNetwork(torch.nn.Module):
    def __init__(self, n_states, n_actions, n_affine = 256):
        super().__init__()

        # Track arguments for further use
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_affine = n_affine

        # Layer definitions
        self.affine = torch.nn.Linear(self.n_states, self.n_affine)
        self.pi = torch.nn.Linear(self.n_affine, self.n_actions)
    
    def forward(self, x):
        x = F.relu(self.affine(x))
        return F.softmax(self.pi(x), dim=-1)

class FFValueNetwork(torch.nn.Module):
    def __init__(self, n_states, n_affine = 256):
        super().__init__()

        # Track arguments for further use
        self.n_states = n_states
        self.n_affine = n_affine

        # Layer definitions
        self.affine = torch.nn.Linear(self.n_states, self.n_affine)
        self.value = torch.nn.Linear(self.n_affine, 1)
    
    def forward(self, x):
        h = F.relu(self.affine(x))
        v = self.value(h)
        return v