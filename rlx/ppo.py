import torch

from .utils import compute_returns, compute_bootstrapped_returns

class PPO(object):
    ''' Proximal Policy Optimization (PPO) with clipping. '''

    def __init__(self, agent):
        super().__init__()
        self.agent = agent # Track the agent

    def train(self, global_network_state, global_env_state, *, horizon, batch_size=8, gamma=0.99, entropy_reg=1e-2, render=False,
                k_epochs=4, ppo_clip=0.2, **kwargs):
        standardize = False if 'standardize_return' not in kwargs.keys() else kwargs['standardize_return']
        grad_clip = kwargs['grad_clip']
        
        avg_length, avg_reward = 0., 0.

        batch_rollouts = []
        for b in range(batch_size):
            with torch.no_grad():
                base_rollout = self.agent.episode(horizon, global_network_state, global_env_state, render=render)[:-1]
            base_rewards, base_logprobs = base_rollout.rewards, base_rollout.logprobs
            base_returns = compute_returns(base_rewards, gamma)
            batch_rollouts.append((base_rollout, base_logprobs, base_returns))

            # compute some metrics to track
            avg_length = ((avg_length * b) + len(base_rollout)) / (b + 1)
            avg_reward = ((avg_reward * b) + base_rewards.sum()) / (b + 1)

        for _ in range(k_epochs):
            self.agent.zero_grad()
            for b in range(batch_size):
                base_rollout, base_logprobs, base_returns = batch_rollouts[b]

                rollout = self.agent.evaluate(base_rollout)
                logprobs, entropy = rollout.logprobs, rollout.entropy
                values, = rollout.others

                ratios = (logprobs - base_logprobs.detach()).exp()
                
                advantage = base_returns - values.squeeze()
                if standardize and advantage.numel() != 1:
                    advantage = (advantage - advantage.mean()) / advantage.std()

                policyloss = - torch.min(ratios, torch.clamp(ratios, 1 - ppo_clip, 1 + ppo_clip)) * advantage.detach()
                valueloss = advantage.pow(2)
                entropyloss = - entropy_reg * entropy

                loss = policyloss.sum() + valueloss.sum() + entropyloss.sum()
                loss = loss / batch_size
                loss.backward()
            self.agent.step(grad_clip)

        return avg_reward, avg_length