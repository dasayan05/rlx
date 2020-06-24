import copy
import torch

from .pgalgo import PGAlgorithm
from ..replay import Replay

class OffPolicyActorCritic(PGAlgorithm):
    ''' Off-Policy REINFORCE with Value-baseline. '''

    def populate_replay_buffer(self, buffer, behavior, horizon, global_network_state, global_env_state,
            n_episodes, render=False):
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
                buffer.populate_buffer((bh_rollout, bh_logprobs))
        return avg_reward, avg_length

    def train(self, global_network_state, global_env_state, *, behavior, horizon, batch_size=4, gamma=0.99, entropy_reg=1e-2,
            value_reg=0.5, buffer_size=100, render=False, **kwargs):
        standardize = False if 'standardize_return' not in kwargs.keys() else kwargs['standardize_return']
        grad_clip = None if 'grad_clip' not in kwargs.keys() else kwargs['grad_clip']

        if not hasattr(self, 'replay'):
            self.replay = Replay(buffer_size)
            self.update_count = 0
        
        self.avg_reward, self.avg_length = self.populate_replay_buffer(self.replay, behavior, horizon,
                global_network_state, global_env_state, batch_size, render)

        batch = self.replay.get_batch(batch_size)

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
        self.update_count += 1

        if self.update_count % 10 == 0:
            behavior.load_state_dict(self.network.state_dict())

        return self.avg_reward, self.avg_length