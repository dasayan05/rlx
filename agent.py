import abc
import torch

from utils import Rollout

class PGAgent(object):
    def __init__(self, env, policy, device=None, lr=1e-4):
        super().__init__()

        # Track arguments
        self.environment = env
        self.policy = policy
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu') if device is None else device

        # Internal objects
        self.network = self.policy((self.environment.observation_space,), (self.environment.action_space,), n_hidden=128)
        if torch.cuda.is_available():
            self.network = self.network.to(device)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    def reset(self):
        return torch.from_numpy(self.environment.reset()).float().to(self.device)

    def timestep(self, *states):
        action_dist, *extra = self.network(*states) # invoke the policy
        action = action_dist.sample() # sample an action
        
        # Transition to new state and retrieve a reward
        st, reward, done, _ = self.environment.step(*[a.item() for a in action])
        # TODO: The below state can have multiple components
        next_state = torch.from_numpy(st).float().to(self.device) # update current state
        logprob = action_dist.log_prob(*action)

        return (action, logprob, reward, next_state, done, *extra)

    def episode(self, horizon, global_state=None, detach=False):
        ep_reward = 0 # total reward for full episode

        state = self.reset() # prepares for a new episode
        self.network.reset()

        rollout = Rollout(device=self.device)

        # loop for many time-steps
        for t in range(horizon):
            state_tuple = (state,) if global_state is None else (state, global_state)
            action, logprob, reward, next_state, done, *extra = self.timestep(*state_tuple)
            rollout.step(state, action, reward, logprob, *extra)
            state = next_state
            
            if done: break

        return rollout

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()