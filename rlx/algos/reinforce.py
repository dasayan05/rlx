import torch

from .pgalgo import PGAlgorithm

class REINFORCE(PGAlgorithm):
    ''' REINFORCE algorithm. '''

    def train(self, global_network_state, global_env_state, *, horizon, gamma=0.99, batch_size=4, render=False, **kwargs):
        standardize = False if 'standardize_return' not in kwargs.keys() else kwargs['standardize_return']
        grad_clip = None if 'grad_clip' not in kwargs.keys() else kwargs['grad_clip']

        avg_length, avg_reward = 0., 0.

        self.zero_grad()
        for b in range(batch_size):
            with torch.set_grad_enabled(self.reccurrent):
                rollout = self.agent(self.network).episode(horizon, global_network_state, global_env_state,
                    dry=not self.reccurrent, render=render)[:-1]
                rollout.mc_returns(gamma)

            # compute some metrics to track
            avg_length = ((avg_length * b) + len(rollout)) / (b + 1)
            avg_reward = ((avg_reward * b) + rollout.rewards.sum()) / (b + 1)

            if not self.reccurrent:
                rollout = rollout.vectorize()
                rollout = self.agent(self.network).evaluate(rollout)
            returns, logprobs = rollout.returns, rollout.logprobs
            
            if standardize and returns.numel() != 1:
                returns = (returns - returns.mean()) / returns.std()

            policyloss = - returns * logprobs
            loss = policyloss.mean()
            loss /= batch_size
            loss.backward()
        
        self.step(grad_clip)

        return avg_reward, avg_length