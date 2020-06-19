import torch

from .pgalgo import PGAlgorithm

class PPO(PGAlgorithm):
    ''' Proximal Policy Optimization (PPO) with clipping. '''

    def train(self, global_network_state, global_env_state, *, horizon, batch_size=8, gamma=0.99, entropy_reg=1e-2, render=False,
                k_epochs=4, ppo_clip=0.2, value_reg=0.5, **kwargs):
        standardize = False if 'standardize_return' not in kwargs.keys() else kwargs['standardize_return']
        grad_clip = None if 'grad_clip' not in kwargs.keys() else kwargs['grad_clip']
        
        avg_length, avg_reward = 0., 0.

        batch_rollouts = []
        for b in range(batch_size):
            with torch.set_grad_enabled(False):
                base_rollout = self.agent(self.network).episode(horizon, global_network_state, global_env_state,
                    dry=not self.reccurrent, render=render)[:-1]
                base_rollout.mc_returns(gamma)

                # compute some metrics to track
                avg_length = ((avg_length * b) + len(base_rollout)) / (b + 1)
                avg_reward = ((avg_reward * b) + base_rollout.rewards.sum()) / (b + 1)

                if not self.reccurrent:
                    base_rollout = base_rollout.vectorize()
                    base_rollout = self.agent(self.network).evaluate(base_rollout)
                base_returns, base_logprobs = base_rollout.returns, base_rollout.logprobs
                batch_rollouts.append((base_rollout, base_logprobs, base_returns))

        for _ in range(k_epochs):
            self.zero_grad()
            for b in range(batch_size):
                base_rollout, base_logprobs, base_returns = batch_rollouts[b]

                rollout = self.agent(self.network).evaluate(base_rollout, recurrence=False)
                logprobs, entropy = rollout.logprobs, rollout.entropy
                values, = rollout.others

                ratios = (logprobs - base_logprobs.detach()).exp()

                advantage = base_returns - values
                if standardize and advantage.numel() != 1:
                    advantage = (advantage - advantage.mean()) / advantage.std()

                policyloss = - torch.min(ratios, torch.clamp(ratios, 1 - ppo_clip, 1 + ppo_clip)) * advantage.detach()
                valueloss = advantage.pow(2)
                entropyloss = - entropy_reg * entropy

                loss = policyloss.mean() + value_reg * valueloss.mean() + entropyloss.mean()
                loss = loss / batch_size
                loss.backward()
            self.step(grad_clip)

        return avg_reward, avg_length