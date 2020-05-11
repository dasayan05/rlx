import torch
import torch.distributions as dist

def compute_returns(rewards, gamma = 0.99):
    # compute return (sum of future rewards)
    returns = [rewards[-1],]
    for r in reversed(rewards[:-1]):
        returns.insert(0, r + gamma * returns[0])
    returns = torch.tensor(returns)
    return (returns - returns.mean()) / returns.std()

class Rollout(object):
    ''' Contains one single rollout/episode '''

    def __init__(self, device=None):
        super().__init__()

        # The data container
        self.__states, self.__actions, self.__rewards = [], [], []
        self.__logprobs = []
        self.__others = []

        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu') if device is None else device

    def __len__(self):
        return len(self.__states) # length of any one attribute

    def __iter__(self):
        self.t = 0
        return self

    @property
    def rewards(self):
        return torch.tensor(self.__rewards, device=self.device)

    @property
    def logprobs(self):
        return torch.cat(self.__logprobs, dim=-1)

    @property
    def others(self):
        n_others = len(self.__others[0])
        if n_others != 0:
            return tuple(torch.tensor([q[index] for q in self.__others], device=self.device)
                        for index in range(n_others))

    def __next__(self):
        if self.t < self.__len__():
            s, a, r, logprob, *extra = self.__data[self.t]
            self.t += 1
            return (s, a, r), logprob, extra
        else:
            raise StopIteration

    def step(self, state, action, reward, logprob, *other):
        self.__states.append( state )
        self.__actions.append( action )
        self.__rewards.append( reward )
        self.__logprobs.append( logprob )
        self.__others.append( other )

class ActionDistribution(object):
    ''' Encapsulates a multi-part action distribution '''

    def __init__(self, *distribs: dist.Distribution):
        super().__init__()
        # Track the arguments
        self.distribs = distribs

    def sample(self):
        return tuple(d.sample() for d in self.distribs)

    def log_prob(self, *samples):
        # breakpoint()
        return sum([d.log_prob(s) for d, s in zip(self.distribs, samples)])