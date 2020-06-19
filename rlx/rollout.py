import torch

class Rollout(object):
    """ Contains and manages one single rollout/episode """

    def __init__(self, device = None, ctor = None):
        '''
        Constructs a Rollout object
        Arguments:
            device: The device to place all tensors of the rollout
        '''
        super().__init__()

        # The internal data containers
        if ctor is None:
            self._states, self._actions, self._rewards = [], [], []
            self._returns = []
            self._action_dist, self._others = [], []
        else:
            self._states, self._actions, self._rewards, \
            self._returns, \
            self._action_dist, self._others = ctor

        # If 'device' not provided, go ahead based on availability
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu') if device is None else device

        # Flag indicating presence of recurrence
        self.recurrent = False

    def __getitem__(self, index):
        rollout_ = Rollout(device=self.device, ctor=(
                    self._states[index],
                    self._actions[index],
                    self._rewards[index],
                    # The below could be empty if any returns hasn't been computed
                    self._returns[index]  if len(self._returns) != 0 else [],
                    # The below ones are empty if it is dry rollout
                    self._action_dist[index] if len(self._action_dist) != 0 else [],
                    self._others[index] if len(self._others) != 0 else []
                ))

        return rollout_

    def dry(self):
        rollout_ = Rollout(device=self.device, ctor=(
                    self._states, self._actions, self._rewards,
                    self._returns,
                    [], # 'action_dist' is empty in case of dry rollout
                    [], # 'others' is empty in case of dry rollout
                ))

        return rollout_

    @property
    def rewards(self):
        ''' Returns the sequence of rewards (Tensorized) '''
        return torch.cat(self._rewards, dim=-1)

    def mc_returns(self, gamma=1.0):
        ''' Computes Monte-Carlo returns for each timesteps (and optionally returns) '''
        returns = [self._rewards[-1],]
        for r in reversed(self._rewards[:-1]):
            returns.insert(0, (r + gamma * returns[0]))
        self._returns = returns
        
        return self.returns # this is the '.returns' property

    @property
    def returns(self):
        return torch.cat(self._returns, dim=-1)

    @returns.setter
    def returns(self, custom_returns):
        ''' Set a custom return in case simple Monte-Carlo is not enough '''
        assert custom_returns.shape[-1] == len(self), 'custom returns should have the same length as the rollout'
        self._returns = [q.unsqueeze(-1) for q in torch.unbind(custom_returns.detach(), -1)]

    @property
    def logprobs(self):
        ''' Returns the sequence of log-probabilities, i.e. log pi(a|s))) (Tensorized) '''
        assert len(self._action_dist) != 0, 'Rollout must NOT be dry in order to compute logprobs'
        return torch.cat([ d.log_prob(*a) for a, d in zip(self._actions, self._action_dist) ], dim=-1)
    
    @property
    def others(self):
        ''' Extra info per time step (e.g., value) '''
        assert len(self._others) != 0, 'Rollout must NOT be dry in order to access any part of computation graph'
        n_others = len(self._others[0])
        if n_others != 0:
            return tuple(torch.cat([others[index] for others in self._others], dim=-1)
                                            for index in range(n_others))
    @property
    def entropy(self):
        ''' Returns the sequence of entropy at every timestep '''
        assert len(self._action_dist) != 0, 'Rollout must NOT be dry in order to compute entropy'
        return torch.cat([d.entropy() for d in self._action_dist], dim=-1)

    def __len__(self):
        ''' Returns the length of the rollout '''
        return len(self._states)

    def __iter__(self):
        ''' Iterator Protocol: Iterates over each the experience tuple (s,a,r,..) '''
        self.t = 0
        return self

    def __next__(self):
        ''' Iterator Protocol: Returns the next experience tuple (s,a,r,..) '''
        if self.t < len(self):
            (rs, s), a, r = self._states[self.t], self._actions[self.t], self._rewards[self.t]
            g = self._returns[self.t] if len(self._returns) != 0 else None
            action_dist = self._action_dist[self.t] if len(self._action_dist) != 0 else None
            other = self._others[self.t] if len(self._others) != 0 else None
            self.t += 1
            return ((rs, s), a, r, g), action_dist, other
        else:
            raise StopIteration

    def __lshift__(self, rhs):
        ''' Inserts a new experience tuple into the rollout '''
        state, action, reward, action_dist, *others = rhs

        self.recurrent = state[0] is not None
        # (s, a, r) is mandetory
        self._states.append( state )
        self._actions.append( action )
        self._rewards.append( reward )

        # action distribution and others are optional
        if (action_dist is not None) and (others is not None):
            self._action_dist.append( action_dist )
            self._others.append( tuple(others) )

    def vectorize(self):
        ''' Vectorize the rollout; mainly for efficiency purpose '''
        
        # TODO: Maybe we can handle this generally
        assert len(self._action_dist) == 0, 'Only dry rollouts can be vectorized'
        
        n_actions = len(self._actions[0]) # no. of action components

        vec_states = torch.cat([q for _, q in self._states], dim=0) # first dimension must be batch
        if self.recurrent:
            vec_recur_states = torch.cat([q for q, _ in self._states], dim=0)
        else:
            vec_recur_states = None
        vec_actions = tuple(torch.cat([q[ia] for q in self._actions], dim=0) for ia in range(n_actions))
        vec_rewards = self.rewards.view(-1, 1) # TODO: Do we really need this ?
        
        rollout_ = Rollout(device=self.device, ctor=(
                [(vec_recur_states, vec_states),],
                [vec_actions,],
                [vec_rewards,],
                [self.returns.view(-1, 1),] if len(self._returns) != 0 else [],
                [], # It's dry rollout, so not needed
                [], # It's dry rollout, so not needed
            )
        )
        return rollout_