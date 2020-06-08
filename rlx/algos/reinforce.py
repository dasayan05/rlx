import torch

from .pgalgo import PGAlgorithm

class REINFORCE(PGAlgorithm):
    ''' REINFORCE algorithm. '''

    def train(self, global_network_state, global_env_state, *, horizon, gamma=0.99, batch_size=4, render=False, **kwargs):
        standardize = False if 'standardize_return' not in kwargs.keys() else kwargs['standardize_return']
        grad_clip = kwargs['grad_clip']

        avg_length, avg_reward = 0., 0.
        self.zero_grad()
        for b in range(batch_size):
            rollout = self.agent(self.network).episode(horizon, global_network_state, global_env_state, render=render)[:-1]
            rewards, logprobs = rollout.rewards, rollout.logprobs
            returns = rollout.mc_returns(gamma, standardize=standardize)
            
            # compute some metrics to track
            avg_length = ((avg_length * b) + len(rollout)) / (b + 1)
            avg_reward = ((avg_reward * b) + rewards.sum()) / (b + 1)

            policyloss = - returns.detach() * logprobs
            loss = policyloss.sum()
            loss /= batch_size
            loss.backward()
        
        self.step(grad_clip)

        return avg_reward, avg_length