import abc
import torch

from utils import compute_returns, compute_bootstrapped_returns

class REINFORCE(object):
    ''' REINFORCE algorithm. '''

    def __init__(self, agent):
        super().__init__()
        self.agent = agent # Track the agent

    def train(self, *, horizon, batch_size, gamma, render = False):
        avg_length = 0
        avg_reward = 0.
        
        self.agent.zero_grad()
        for b in range(batch_size):
            rollout = self.agent.episode(horizon, render=(render, 0.01))[:-1]
            rewards, logprobs = rollout.rewards, rollout.logprobs
            returns = compute_returns(rewards, gamma)
            
            # compute some metrics to track
            avg_length = ((avg_length * b) + len(rollout)) // (b + 1)
            avg_reward = ((avg_reward * b) + rewards.sum()) // (b + 1)

            policyloss = - returns * logprobs
            loss = policyloss.sum()
            loss /= batch_size
            loss.backward()
        
        self.agent.step()

        return avg_reward, avg_length

class ActorCritic(object):
    ''' REINFORCE with Value-baseline. '''

    def __init__(self, agent):
        super().__init__()
        self.agent = agent # Track the agent

    def train(self, *, horizon, batch_size, gamma, entropy_reg, render = False):
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

            advantage = returns - values.detach().squeeze()
            policyloss = - advantage * logprobs
            valueloss = torch.nn.functional.mse_loss(values.squeeze(), returns, reduction='sum')
            loss = policyloss.sum() + valueloss - entropy_reg * entropyloss.sum()
            loss /= batch_size
            loss.backward()
        
        self.agent.step()

        return avg_reward, avg_length

class A2C(object):
    ''' Advantage Actor Critic (A2C). '''

    def __init__(self, agent):
        super().__init__()
        self.agent = agent # Track the agent

    def train(self, *, horizon, batch_size, gamma, entropy_reg, render = False):
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
            policyloss = - advantage.detach() * logprobs
            valueloss = torch.nn.functional.mse_loss(values.squeeze(), returns, reduction='sum')
            loss = policyloss.sum() + valueloss - entropy_reg * entropyloss.sum()
            loss /= batch_size
            loss.backward()
        
        self.agent.step()

        return avg_reward, avg_length

class PPO(object):
    ''' Proximal Policy Optimization (PPO) with clipping. '''

    def __init__(self, agent):
        super().__init__()
        self.agent = agent # Track the agent

    def train(self, *, horizon, gamma, entropy_reg, k_epochs, clip, render = False):
        base_rollout = self.agent.episode(horizon, render=(render, 0.01))[:-1]
        base_rewards, base_logprobs = base_rollout.rewards, base_rollout.logprobs
        base_returns = compute_returns(base_rewards, gamma)
        
        avg_length = len(base_rollout)
        avg_reward = base_rewards.sum()

        for _ in range(k_epochs):
            rollout = self.agent.evaluate(base_rollout)
            logprobs, entropy = rollout.logprobs, rollout.entropy
            values, = rollout.others

            ratios = (logprobs - base_logprobs.detach()).exp()
            
            advantage = base_returns - values.detach().squeeze()
            policyloss = - torch.min(ratios, torch.clamp(ratios, 1 - clip, 1 + clip)) * advantage
            valueloss = torch.nn.functional.mse_loss(values.squeeze(), base_returns, reduction='sum')
            entropyloss = - entropy_reg * entropy

            loss = policyloss.sum() + valueloss + entropyloss.sum()
            
            self.agent.zero_grad()
            loss.backward()
            self.agent.step()

        return avg_reward, avg_length