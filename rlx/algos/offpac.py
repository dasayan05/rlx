import copy
import torch

from .pgalgo import PGAlgorithm

class OffPolicyActorCritic(PGAlgorithm):
    ''' Off-Policy REINFORCE with Value-baseline. '''

    def populate_replay_buffer(self, behavior, horizon, global_network_state, global_env_state, n_episodes, render=False):
        self.buffer = []
        self.buffer_usage, self.i_batch = 0, 0

        avg_length, avg_reward = 0., 0.
        with torch.set_grad_enabled(False):
            for b in range(n_episodes):
                bh_rollout = self.agent(behavior).episode(horizon, global_network_state, global_env_state,
                    dry=False, render=render)[:-1]
                
                # compute some metrics to track
                avg_length = ((avg_length * b) + len(bh_rollout)) / (b + 1)
                avg_reward = ((avg_reward * b) + bh_rollout.rewards.sum()) / (b + 1)

                bh_rollout.mc_returns()
                bh_logprobs = bh_rollout.logprobs
                bh_rollout = bh_rollout.dry()

                if not self.reccurrent:
                    bh_logprobs = bh_logprobs.view(-1, 1)
                    bh_rollout = bh_rollout.vectorize()
                self.buffer.append((bh_rollout, bh_logprobs))
        return avg_reward, avg_length

    def train(self, global_network_state, global_env_state, *, behavior, horizon, batch_size=4, gamma=0.99, entropy_reg=1e-2,
            value_reg=0.5, buffer_size=100, max_buffer_usage=5, render=False, **kwargs):
        standardize = False if 'standardize_return' not in kwargs.keys() else kwargs['standardize_return']
        grad_clip = None if 'grad_clip' not in kwargs.keys() else kwargs['grad_clip']

        if not hasattr(self, 'buffer'):
            self.avg_reward, self.avg_length = self.populate_replay_buffer(behavior, horizon,
                global_network_state, global_env_state, buffer_size, render)

        batch = self.buffer[self.i_batch:min(self.i_batch + batch_size, buffer_size)]

        self.zero_grad()
        for b, (bh_roll, bh_lp) in enumerate(batch):
            rollout = self.agent(self.network).evaluate(bh_roll, recurrence=False)

            returns, logprobs = rollout.returns, rollout.logprobs
            values, = rollout.others
            entropyloss = rollout.entropy
            
            # The importance ratio
            ratio = (logprobs.detach() - bh_lp.detach()).exp()

            advantage = returns - values
            if standardize and advantage.numel() != 1:
                advantage = (advantage - advantage.mean()) / advantage.std()

            policyloss = - ratio * advantage.detach() * logprobs
            valueloss = advantage.pow(2)
            loss = policyloss.mean() + value_reg * valueloss.mean() - entropy_reg * entropyloss.mean()
            loss /= batch_size
            loss.backward()
        self.step(grad_clip)

        self.i_batch += batch_size
        if self.i_batch >= buffer_size:
            self.i_batch = 0
            self.buffer_usage += 1
            if self.buffer_usage == max_buffer_usage:
                behavior.load_state_dict(self.network.state_dict())
                self.avg_reward, self.avg_length = self.populate_replay_buffer(behavior, horizon,
                global_network_state, global_env_state, buffer_size, render)

        return self.avg_reward, self.avg_length