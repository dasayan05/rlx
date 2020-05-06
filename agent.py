import abc
import torch

class PGAgent(object):
    def __init__(self, env, policytype, storages=[], device=None):
        super().__init__()

        # Track arguments
        self.environment = env
        self.PolicyType = policytype
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.storages = storages # names of containers to be used during episodes

        # Internal objects
        self.policy = self.PolicyType(self.environment.observation_space, self.environment.action_space, self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters())

    def reset(self):
        initial_state = torch.from_numpy(self.environment.reset()).float().to(self.device)
        for storage in self.storages:
            setattr(self, storage, []) # one list for every storage
        return initial_state

    @abc.abstractmethod
    def timestep(self, state):
        pass

    @abc.abstractmethod
    def compute_loss(self):
        pass

    def episode(self, max_length, **kwargs):
        ep_reward = 0 # total reward for full episode

        state = self.reset() # prepares for a new episode
        # loop for many time-steps
        for t in range(max_length):            
            next_state, r, done = self.timestep(state)
            state = next_state
            ep_reward += r
            
            if done:
                break

        return ep_reward, state

    def train(self):
        self.optimizer.zero_grad()
        loss = self.compute_loss()
        loss.backward()
        self.optimizer.step()