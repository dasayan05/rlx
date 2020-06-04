from .pgalgo import PGAlgorithm
from ..utils import compute_returns, compute_bootstrapped_returns

class ActorCritic(PGAlgorithm):
    ''' REINFORCE with Value-baseline. '''

    def train(self, global_network_state, global_env_state, *, horizon, batch_size=4, gamma=0.99, entropy_reg=1e-2, render=False, **kwargs):
        standardize = False if 'standardize_return' not in kwargs.keys() else kwargs['standardize_return']
        grad_clip = kwargs['grad_clip']
        
        avg_length, avg_reward = 0., 0.
        self.zero_grad()
        for b in range(batch_size):
            rollout = self.agent(self.network).episode(horizon, global_network_state, global_env_state, render=render)[:-1]
            rewards, logprobs = rollout.rewards, rollout.logprobs
            returns = compute_returns(rewards, gamma)
            values, = rollout.others
            entropyloss = rollout.entropy

            # compute some metrics to track
            avg_length = ((avg_length * b) + len(rollout)) / (b + 1)
            avg_reward = ((avg_reward * b) + rewards.sum()) / (b + 1)

            advantage = returns - values.squeeze()
            if standardize and advantage.numel() != 1:
                advantage = (advantage - advantage.mean()) / advantage.std()

            policyloss = - advantage.detach() * logprobs
            valueloss = advantage.pow(2)
            loss = policyloss.sum() + valueloss.sum() - entropy_reg * entropyloss.sum()
            loss /= batch_size
            loss.backward()
        
        self.step(grad_clip)

        return avg_reward, avg_length

class A2C(PGAlgorithm):
    ''' Advantage Actor Critic (A2C). '''

    def train(self, global_network_state, global_env_state, *, horizon, batch_size=4, gamma=0.99, entropy_reg=1e-2, render=False, **kwargs):
        standardize = False if 'standardize_return' not in kwargs.keys() else kwargs['standardize_return']
        grad_clip = kwargs['grad_clip']

        avg_length, avg_reward = 0., 0.
        self.zero_grad()
        for b in range(batch_size):
            rollout = self.agent(self.network).episode(horizon, global_network_state, global_env_state, render=render)
            end_v, = rollout[-1].others
            rollout = rollout[:-1]
            rewards, logprobs = rollout.rewards, rollout.logprobs
            returns = compute_bootstrapped_returns(rewards, end_v, gamma)
            values, = rollout.others
            entropyloss = rollout.entropy

            # compute some metrics to track
            avg_length = ((avg_length * b) + len(rollout)) / (b + 1)
            avg_reward = ((avg_reward * b) + rewards.sum()) / (b + 1)

            advantage = returns - values.squeeze()
            if standardize and advantage.numel() != 1:
                advantage = (advantage - advantage.mean()) / advantage.std()

            policyloss = - advantage.detach() * logprobs
            valueloss = advantage.pow(2)
            loss = policyloss.sum() + valueloss.sum() - entropy_reg * entropyloss.sum()
            loss /= batch_size
            loss.backward()
        
        self.step(grad_clip)

        return avg_reward, avg_length