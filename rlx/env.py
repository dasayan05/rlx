import gym
import abc
import numpy as np

class Environment(object):
    """
    The base class for all environments. This is a bit more flexible than 'gym.Env'.
    It supports a multi-component action-tuple and a global state of the environment.
    """

    def __init__(self):
        # Just one observation space and multi-component action-space tuple
        self.observation_space = None
        self.action_spaces = (None,)

    @abc.abstractmethod
    def reset(self, global_state=None):
        raise NotImplementedError('Implement this method in derived classes')

    @abc.abstractmethod
    def step(self, *actions):
        raise NotImplementedError('Implement this method in derived classes')

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def render(self, **kwargs):
        raise NotImplementedError('Implement this method in derived classes')

class CartPolev0(Environment):
    
    def __init__(self):
        # The original 'CartPole-v0' environment from gym
        self._gymenv = gym.make('CartPole-v0')

        self.observation_space = self._gymenv.observation_space
        self.action_spaces = (self._gymenv.action_space,)

    def reset(self, global_state=None):
        return self._gymenv.reset()

    def step(self, *actions):
        assert len(actions) == 1, 'CartPole-v0 has only one action component'
        return self._gymenv.step(*tuple(a.item() for a in actions))

    def close(self):
        self._gymenv.close()

    def render(self, **kwargs):
        self._gymenv.render(**kwargs)

class IncompleteCartPolev0(Environment):
    
    def __init__(self):
        # The original 'CartPole-v0' environment from gym
        self._gymenv = gym.make('CartPole-v0')

        low, high = self._gymenv.observation_space.low, self._gymenv.observation_space.high
        self.observation_space = gym.spaces.Box(low[:-1], high[:-1], dtype=np.float32)
        self.action_spaces = (self._gymenv.action_space,)

    def reset(self, global_state=None):
        return self._gymenv.reset()[:-1]

    def step(self, *actions):
        assert len(actions) == 1, 'CartPole-v0 has only one action component'
        next_state, rew, done, H = self._gymenv.step(*tuple(a.item() for a in actions))
        return next_state[:-1], rew, done, H

    def close(self):
        self._gymenv.close()

    def render(self, **kwargs):
        self._gymenv.render(**kwargs)

class CartPolev1(Environment):
    
    def __init__(self):
        # The original 'CartPole-v1' environment from gym
        self._gymenv = gym.make('CartPole-v1')

        self.observation_space = self._gymenv.observation_space
        self.action_spaces = (self._gymenv.action_space,)

    def reset(self, global_state=None):
        return self._gymenv.reset()

    def step(self, *actions):
        assert len(actions) == 1, 'CartPole-v0 has only one action component'
        return self._gymenv.step(*tuple(a.item() for a in actions))

    def close(self):
        self._gymenv.close()

    def render(self, **kwargs):
        self._gymenv.render(**kwargs)

class IncompleteCartPolev1(Environment):
    
    def __init__(self):
        # The original 'CartPole-v1' environment from gym
        self._gymenv = gym.make('CartPole-v1')

        low, high = self._gymenv.observation_space.low, self._gymenv.observation_space.high
        self.observation_space = gym.spaces.Box(low[:-1], high[:-1], dtype=np.float32)
        self.action_spaces = (self._gymenv.action_space,)

    def reset(self, global_state=None):
        return self._gymenv.reset()[:-1]

    def step(self, *actions):
        assert len(actions) == 1, 'CartPole-v0 has only one action component'
        next_state, rew, done, H = self._gymenv.step(*tuple(a.item() for a in actions))
        return next_state[:-1], rew, done, H

    def close(self):
        self._gymenv.close()

    def render(self, **kwargs):
        self._gymenv.render(**kwargs)