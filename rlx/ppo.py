import torch

from .utils import compute_returns, compute_bootstrapped_returns

class PPO(object):
    ''' Proximal Policy Optimization (PPO) with clipping. '''

    def __init__(self, agent):
        super().__init__()
        self.agent = agent # Track the agent

    def train(self, **kwargs):
        horizon, gamma, entropy_reg, k_epochs, clip, render = kwargs['horizon'], kwargs['gamma'], \
                kwargs['entropy_reg'], kwargs['ppo_k'], kwargs['ppo_clip'], kwargs['render']

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
            
            advantage = base_returns - values.squeeze()
            advantage = (advantage - advantage.mean()) / advantage.std()
            policyloss = - torch.min(ratios, torch.clamp(ratios, 1 - clip, 1 + clip)) * advantage.detach()
            valueloss = advantage.pow(2)
            entropyloss = - entropy_reg * entropy

            loss = policyloss.sum() + valueloss.sum() + entropyloss.sum()
            
            self.agent.zero_grad()
            loss.backward()
            self.agent.step(clip=None)

        return avg_reward, avg_length