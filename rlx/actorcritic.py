from .utils import compute_returns, compute_bootstrapped_returns

class ActorCritic(object):
    ''' REINFORCE with Value-baseline. '''

    def __init__(self, agent):
        super().__init__()
        self.agent = agent # Track the agent

    def train(self, **kwargs):
        horizon, batch_size, gamma, entropy_reg, render = kwargs['horizon'], kwargs['batch_size'], \
                    kwargs['gamma'], kwargs['entropy_reg'], kwargs['render']
        
        avg_length = 0
        avg_reward = 0.
            
        self.agent.zero_grad()
        for b in range(batch_size):
            rollout = self.agent.episode(horizon, render=(render, 0.01))[:-1]
            rewards, logprobs = rollout.rewards, rollout.logprobs
            returns = compute_returns(rewards, gamma)
            values, = rollout.others
            entropyloss = rollout.entropy

            # compute some metrics to track
            avg_length = ((avg_length * b) + len(rollout)) // (b + 1)
            avg_reward = ((avg_reward * b) + rewards.sum()) // (b + 1)

            advantage = returns - values.squeeze()
            advantage = (advantage - advantage.mean()) / advantage.std()
            policyloss = - advantage.detach() * logprobs
            valueloss = advantage.pow(2)
            loss = policyloss.sum() + valueloss.sum() - entropy_reg * entropyloss.sum()
            loss /= batch_size
            loss.backward()
        
        self.agent.step()

        return avg_reward, avg_length

class A2C(object):
    ''' Advantage Actor Critic (A2C). '''

    def __init__(self, agent):
        super().__init__()
        self.agent = agent # Track the agent

    def train(self, **kwargs):
        horizon, batch_size, gamma, entropy_reg, render = kwargs['horizon'], kwargs['batch_size'], \
                    kwargs['gamma'], kwargs['entropy_reg'], kwargs['render']

        avg_length = 0
        avg_reward = 0.
        
        self.agent.zero_grad()
        for b in range(batch_size):
            rollout = self.agent.episode(horizon, render=(render, 0.01))
            end_v, = rollout[-1].others
            rollout = rollout[:-1]
            rewards, logprobs = rollout.rewards, rollout.logprobs
            returns = compute_bootstrapped_returns(rewards, end_v, gamma)
            values, = rollout.others
            entropyloss = rollout.entropy

            # compute some metrics to track
            avg_length = ((avg_length * b) + len(rollout)) // (b + 1)
            avg_reward = ((avg_reward * b) + rewards.sum()) // (b + 1)

            advantage = returns - values.squeeze()
            advantage = (advantage - advantage.mean()) / advantage.std()
            policyloss = - advantage.detach() * logprobs
            valueloss = advantage.pow(2)
            loss = policyloss.sum() + valueloss.sum() - entropy_reg * entropyloss.sum()
            loss /= batch_size
            loss.backward()
        
        self.agent.step()

        return avg_reward, avg_length