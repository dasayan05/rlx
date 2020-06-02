from .utils import compute_returns

class REINFORCE(object):
    ''' REINFORCE algorithm. '''

    def __init__(self, agent):
        super().__init__()
        self.agent = agent # Track the agent

    def train(self, global_network_state, global_env_state, *, horizon, gamma=0.99, batch_size=4, render=False, **kwargs):
        standardize = False if 'standardize_return' not in kwargs.keys() else kwargs['standardize_return']
        grad_clip = kwargs['grad_clip']

        avg_length, avg_reward = 0., 0.
        self.agent.zero_grad()
        for b in range(batch_size):
            rollout = self.agent.episode(horizon, global_network_state, global_env_state, render=render)[:-1]
            rewards, logprobs = rollout.rewards, rollout.logprobs
            returns = compute_returns(rewards, gamma, standardize=standardize)
            
            # compute some metrics to track
            avg_length = ((avg_length * b) + len(rollout)) / (b + 1)
            avg_reward = ((avg_reward * b) + rewards.sum()) / (b + 1)

            policyloss = - returns.detach() * logprobs
            loss = policyloss.sum()
            loss /= batch_size
            loss.backward()
        
        self.agent.step(grad_clip)

        return avg_reward, avg_length