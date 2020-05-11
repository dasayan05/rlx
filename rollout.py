import torch

class Rollout(object):
    """ Contains and manages one single rollout/episode """

    def __init__(self, device = None):
        '''
        Constructs a Rollout object
        Arguments:
            device: The device to place all tensors of the rollout
        '''
        super().__init__()

        # The internal data containers
        self.__states, self.__actions, self.__rewards = [], [], []
        self.__logprobs = []
        self.__others = []

        # If 'device' not provided, go ahead based on availability
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu') if device is None else device

    @property
    def rewards(self):
        ''' Returns the sequence of rewards (Tensorized) '''
        return torch.tensor(self.__rewards, device=self.device)

    @property
    def logprobs(self):
        ''' Returns the sequence of log-probabilities, i.e. log pi(a|s))) (Tensorized) '''
        return torch.cat(self.__logprobs, dim=-1)

    @property
    def others(self):
        ''' Extra info per time step (e.g., value) '''
        n_others = len(self.__others[0])
        if n_others != 0:
            return tuple(torch.tensor([other[index] for other in self.__others], device=self.device)
                        for index in range(n_others))

    def __len__(self):
        ''' Returns the length of the rollout '''
        return len(self.__states)

    def __iter__(self):
        ''' Iterator Protocol: Iterates over each the experience tuple (s,a,r,..) '''
        self.t = 0
        return self

    def __next__(self):
        ''' Iterator Protocol: Returns the next experience tuple (s,a,r,..) '''
        if self.t < len(self):
            s, a, r = self.__states[self.t], self.__actions[self.t], self.__rewards[self.t]
            logprob = self.__logprobs[self.t]
            other = self.__data[self.t]
            self.t += 1
            return (s, a, r), logprob, other
        else:
            raise StopIteration

    def __lshift__(self, rhs):
        ''' Inserts a new experience tuple into the rollout '''
        state, action, reward, logprob, *other = rhs
        self.__states.append( state )
        self.__actions.append( action )
        self.__rewards.append( reward )
        self.__logprobs.append( logprob )
        self.__others.append( other )