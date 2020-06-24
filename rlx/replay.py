import random
from collections import deque

from .rollout import Rollout

class Replay(object):
    ''' Experiecne replay memory '''

    def __init__(self, max_size=100):
        # The replay buffer (as a deque)
        self.buffer = deque([], max_size)

    def populate_buffer(self, rollout):
        self.buffer.append(rollout)

    def get_batch(self, k):
        k = min(k, len(self.buffer))
        return random.sample(self.buffer, k=k)

    def __len__(self):
        return len(self.buffer)