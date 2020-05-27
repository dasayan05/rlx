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
            self._states, self._actions, self._rewards, self._action_dist, self._others = [], [], [], [], []
        else:
            self._states, self._actions, self._rewards, self._action_dist, self._others = ctor

        # If 'device' not provided, go ahead based on availability
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu') if device is None else device

    def __getitem__(self, index):
        rollout_ = Rollout(device=self.device, ctor=(
                    self._states[index],
                    self._actions[index],
                    self._rewards[index],
                    self._action_dist[index],
                    self._others[index]
                ))

        return rollout_

    @property
    def rewards(self):
        ''' Returns the sequence of rewards (Tensorized) '''
        return torch.tensor(self._rewards, device=self.device)

    @property
    def logprobs(self):
        ''' Returns the sequence of log-probabilities, i.e. log pi(a|s))) (Tensorized) '''
        return torch.cat([ d.log_prob(*a) for a, d in zip(self._actions, self._action_dist) ])
    
    @property
    def others(self):
        ''' Extra info per time step (e.g., value) '''
        n_others = len(self._others[0])
        if n_others != 0:
            return tuple(torch.cat([others[index] for others in self._others], dim=0)
                                            for index in range(n_others))
    @property
    def entropy(self):
        ''' Returns the sequence of entropy at every timestep '''
        return torch.cat([d.entropy() for d in self._action_dist])

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
            s, a, r = self._states[self.t], self._actions[self.t], self._rewards[self.t]
            action_dist = self._action_dist[self.t]
            other = self._others[self.t]
            self.t += 1
            return (s, a, r), action_dist, other
        else:
            raise StopIteration

    def __lshift__(self, rhs):
        ''' Inserts a new experience tuple into the rollout '''
        state, action, reward, action_dist, *others = rhs
        self._states.append( state )
        self._actions.append( action )
        self._rewards.append( reward )
        self._action_dist.append( action_dist )
        self._others.append( others )