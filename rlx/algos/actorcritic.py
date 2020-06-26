import torch

from .pgalgo import PGAlgorithm

class ActorCritic(PGAlgorithm):
    ''' REINFORCE with Value-baseline. '''

    def train(self, global_network_state, global_env_state, *, horizon, batch_size=4, gamma=0.99, entropy_reg=1e-2,
            value_reg=0.5, render=False, **kwargs):
        standardize = False if 'standardize_return' not in kwargs.keys() else kwargs['standardize_return']
        grad_clip = None if 'grad_clip' not in kwargs.keys() else kwargs['grad_clip']
        
        avg_length, avg_reward = 0., 0.

        self.zero_grad()
        for b in range(batch_size):
            with torch.set_grad_enabled(self.reccurrent):
                rollout = self.agent(self.network).episode(horizon, global_network_state, global_env_state,
                    dry=not self.reccurrent, render=render)[:-1]
                rollout.mc_returns()

            # compute some metrics to track
            avg_length = ((avg_length * b) + len(rollout)) / (b + 1)
            avg_reward = ((avg_reward * b) + rollout.rewards.sum()) / (b + 1)

            if not self.reccurrent:
                rollout = rollout.vectorize()
                rollout = self.agent(self.network).evaluate(rollout)

            returns, logprobs = rollout.returns, rollout.logprobs
            values, = rollout.others
            entropyloss = rollout.entropy

            advantage = returns - values
            if standardize and advantage.numel() != 1:
                advantage = (advantage - advantage.mean()) / advantage.std()

            policyloss = - advantage.detach() * logprobs
            valueloss = advantage.pow(2)
            loss = policyloss.mean() + value_reg * valueloss.mean() - entropy_reg * entropyloss.mean()
            loss /= batch_size
            loss.backward()
        
        self.step(grad_clip)

        return avg_reward, avg_length

class A2C(PGAlgorithm):
    ''' Advantage Actor Critic (A2C). '''

    def compute_bootstrapped_returns(rewards, end_v, gamma = 1.0):
        returns = []
        v = end_v
        for t in reversed(range(rewards.shape[0])):
            returns.insert(0, rewards[t,:] + gamma * v)
            v = returns[0]
        return torch.stack(returns, dim=0)

    def train(self, global_network_state, global_env_state, *, horizon, batch_size=4, gamma=0.99, entropy_reg=1e-2,
            value_reg=0.5, render=False, **kwargs):
        standardize = False if 'standardize_return' not in kwargs.keys() else kwargs['standardize_return']
        grad_clip = None if 'grad_clip' not in kwargs.keys() else kwargs['grad_clip']

        avg_length, avg_reward = 0., 0.
        self.zero_grad()
        for b in range(batch_size):
            with torch.set_grad_enabled(self.reccurrent):
                rollout = self.agent(self.network).episode(horizon, global_network_state, global_env_state,
                    dry=False, render=render)
            end_v, = rollout[-1].others
            if not self.reccurrent:
                # TODO: This hack is really ugly. Need to fix the interface
                rollout = rollout.make_dry()[:-1]
            else:
                rollout = rollout[:-1]
            rewards = rollout.rewards
            rollout.returns = A2C.compute_bootstrapped_returns(rewards, end_v, gamma)

            # compute some metrics to track
            avg_length = ((avg_length * b) + len(rollout)) / (b + 1)
            avg_reward = ((avg_reward * b) + rewards.sum()) / (b + 1)

            if not self.reccurrent:
                rollout = rollout.vectorize()
                rollout = self.agent(self.network).evaluate(rollout)
            
            returns, logprobs = rollout.returns, rollout.logprobs
            values, = rollout.others
            entropyloss = rollout.entropy

            advantage = returns - values
            if standardize and advantage.numel() != 1:
                advantage = (advantage - advantage.mean()) / advantage.std()

            policyloss = - advantage.detach() * logprobs
            valueloss = advantage.pow(2)
            loss = policyloss.mean() + value_reg * valueloss.mean() - entropy_reg * entropyloss.mean()
            loss /= batch_size
            loss.backward()
        
        self.step(grad_clip)

        return avg_reward, avg_length