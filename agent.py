import abc, time
import torch

from rollout import Rollout

class PGAgent(object):
    """ Encapsulation of an Agent """

    def __init__(self, env, policy, device=None, lr=1e-4):
        '''
        Constructs an Agent from and 'env' and 'policy'.
        Arguments:
            env: An environment respecting the 'gym.Environment' API
            policy: A subclass of the 'policy.Parametric'
            device: Default device of operation
            lr: Learning rate (TODO: Make a better interface for optimizers)
        '''
        super().__init__()

        # Track arguments
        self.environment = env
        self.policy = policy
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu') if device is None else device

        # The internal learnable object
        self.network = self.policy((self.environment.observation_space,), (self.environment.action_space,), n_hidden=128)
        if torch.cuda.is_available():
            self.network = self.network.to(device)

        # Optimizer instance
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    def reset(self):
        ''' Resets the environment and returns an initial state '''
        return torch.from_numpy(self.environment.reset()).float().to(self.device)

    def timestep(self, *states):
        ''' Given a state-tuple, returns the rest of an experience tuple '''

        # Concat the 'global_state', if any.
        full_state = torch.cat([state for state in states], dim=-1)

        action_dist, *others = self.network(full_state) # invoke the policy
        action = action_dist.sample() # sample an action
        
        # Transition to new state and retrieve a reward
        st, reward, done, _ = self.environment.step(*[a.item() for a in action])
        next_state = torch.from_numpy(st).float().to(self.device) # update current state
        logprob = action_dist.log_prob(*action)

        return (action, logprob, reward, next_state, done, *others)

    def episode(self, horizon, global_state=None, detach=False, render=(False, 0)):
        '''
        Samples and returns an entire rollout (as 'Rollout' instance).
        Arguments:
            horizon: Maximum length of the episode.
            global_state: A global state for the whole episode. (TODO: Look into this interface)
            detach: TODO: Yet to implement a better interface for detaching.
            render: A 2-tuple (bool, float) containing whether want to render and an optional time delay
        '''

        ep_reward = 0 # total reward for full episode

        state = self.reset() # prepares for a new episode
        self.network.reset()

        rollout = Rollout(device=self.device)

        # loop for many time-steps
        for t in range(horizon):
            state_tuple = (state,) if global_state is None else (state, global_state)
            
            # Rendering (with optional time delay)
            is_render, delay = render
            if is_render:
                self.environment.render()
                time.sleep(delay)
            
            action, logprob, reward, next_state, done, *others = self.timestep(*state_tuple)
            rollout << (state, action, reward, logprob, *others)
            state = next_state
            
            if done: break

        return rollout

    def zero_grad(self):
        ''' Similar to PyTorch's boilerplate 'optim.zero_grad()' '''
        self.optimizer.zero_grad()

    def step(self):
        ''' Similar to PyTorch's boilerplate 'optim.step()' '''
        self.optimizer.step()