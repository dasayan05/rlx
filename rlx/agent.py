import abc, time
import torch

from .rollout import Rollout
from .env import Environment
from .policy import Parametric

class PGAgent(object):
    """ Encapsulation of an Agent """

    def __init__(self, env, policy, policy_kwargs={}, device=None, lr=1e-4):
        '''
        Constructs an Agent from and 'env' and 'policy'.
        Arguments:
            env: An environment respecting the 'env.Environment' API
            policy: A subclass of the 'policy.Parametric'
            device: Default device of operation
            lr: Learning rate (TODO: Make a better interface for optimizers)
            policy_kwargs: kwargs that go into 'policy' instantiation
        '''
        super().__init__()

        # Track arguments
        assert isinstance(env, Environment), "env object must be an instance of 'env.Environment'"
        self.environment = env
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu') if device is None else device

        # The internal learnable object
        assert issubclass(policy, Parametric), "policy must be a subclass of 'policy.Parametric'"
        self.network = policy(self.environment.observation_space, self.environment.action_spaces, **policy_kwargs)
        self.network = self.network.to(self.device)

        # Optimizer instance
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)

    def reset(self, global_state=None):
        ''' Resets the environment and returns an initial state '''
        return torch.from_numpy(self.environment.reset(global_state=global_state)).float().to(self.device)

    def timestep(self, *states):
        ''' Given a state-tuple, returns the action distribution and any other predicted stuff '''

        # Concat the 'global_state', if any.
        full_state = torch.cat([state for state in states], dim=-1)

        action_dist, *others = self.network(full_state) # invoke the policy

        return (action_dist, *others)

    def evaluate(self, rollout, global_network_state=None):
        ''' Given a rollout, evaluate it against current policy '''

        rollout_new = Rollout(device=self.device)

        self.network.reset()
        for (state, action, reward), _, _ in rollout:
            state_tuple = (state,) if global_network_state is None else (state, global_network_state)

            action_dist, *others = self.timestep(*state_tuple)
            rollout_new << (state, action, reward, action_dist, *others)

        return rollout_new

    def episode(self, horizon, global_network_state=None, global_env_state=None, render=(False, 0)):
        '''
        Samples and returns an entire rollout (as 'Rollout' instance).
        Arguments:
            horizon: Maximum length of the episode.
            global_state: A global state for the whole episode. (TODO: Look into this interface)
            render: A 2-tuple (bool, float) containing whether want to render and an optional time delay
        '''

        ep_reward = 0 # total reward for full episode

        state = self.reset(global_state=global_env_state) # prepares for a new episode
        self.network.reset()

        rollout = Rollout(device=self.device)

        # loop for many time-steps
        for t in range(horizon):
            state_tuple = (state,) if global_network_state is None else (state, global_network_state)
            
            # Rendering (with optional time delay)
            is_render, delay = render
            if is_render:
                self.environment.render()
                time.sleep(delay)
            
            action_dist, *others = self.timestep(*state_tuple)
    
            action = action_dist.sample() # sample an action
            
            # Transition to new state and retrieve a reward
            next_state, reward, done, _ = self.environment.step(*[a.to(torch.device('cpu')).numpy() for a in action])
            next_state = torch.from_numpy(next_state).float().to(self.device)
            
            rollout << (state, action, reward, action_dist, *others) # record one experience tuple
    
            state = next_state # update current state
            
            if done: break

        # One last entry for the last state (sometimes required)
        state_tuple = (state,) if global_network_state is None else (state, global_network_state)
        action_dist, *others = self.timestep(*state_tuple)
        action = action_dist.sample()

        rollout << (state, action, 0.0, action_dist, *others)

        return rollout

    def zero_grad(self):
        ''' Similar to PyTorch's boilerplate 'optim.zero_grad()' '''
        self.optimizer.zero_grad()

    def step(self, clip=1e-2):
        ''' Similar to PyTorch's boilerplate 'optim.step()' '''
        
        # Optional gradient clipping
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), clip)
        
        self.optimizer.step()