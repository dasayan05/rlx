import torch
import torch.distributions as dist

def compute_returns(rewards, gamma = 1.0):
    '''
    Computes 'return' (sum of future rewards, optionally discounted)
    Arguments:
        rewards: A sequence of rewards
        gamma: (optional) Discount factor
    '''
    returns = [rewards[-1].view(1,),]
    for r in reversed(rewards[:-1]):
        returns.insert(0, (r + gamma * returns[0]).view(1,))
    returns = torch.cat(returns, dim=-1)
    return (returns - returns.mean()) / returns.std()

class ActionDistribution(object):
    """ Encapsulates a multi-part action distribution """

    def __init__(self, *distribs: dist.Distribution):
        '''
        Constructs an object from a variable number of 'dist.Distribution's
        Arguments:
            *distribs: A set of 'dist.Distribution' instances
        '''
        super().__init__()

        self.distribs = distribs
        self.n_dist = len(self.distribs)

    def sample(self):
        ''' Sample an action (set of several constituent actions) '''
        return tuple(d.sample() for d in self.distribs)

    def log_prob(self, *samples):
        '''
        Computes the log-probability of the action-tuple under this ActionDistribution.
        Note: Assumed to be independent, i.e., probs factorize.
        Arguments:
            samples: A tuple of actions
        '''
        assert len(samples) == self.n_dist, "Number of constituent distributions is different than number of samples"
        return sum([d.log_prob(s) for d, s in zip(self.distribs, samples)])