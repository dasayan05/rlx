import abc, time
import torch

from .rollout import Rollout
from .env import Environment
from .policy import Parametric

class PGAgent(object):
    """ Encapsulation of an Agent """

    def __init__(self, env, device=None):
        '''
        Constructs an Agent for the given environment. This agent is almost state-less.
        Arguments:
            env: An environment respecting the 'env.Environment' API
            device: Default device of operation
        '''
        super().__init__()
        assert isinstance(env, Environment), "env object must be an instance of 'env.Environment'"

        # Track arguments
        self.environment = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

    def __call__(self, network):
        ''' Augments an agent with a network. The agent cannot work without this. '''
        new_agent = PGAgent(self.environment, device=self.device)
        setattr(new_agent, 'network', network)
        return new_agent

    def cuda(self):
        self.device = torch.device('cuda')
        return self

    def reset(self, global_state=None):
        ''' Resets the environment and returns an initial state '''
        return torch.from_numpy(self.environment.reset(global_state=global_state)).float().to(self.device)

    def timestep(self, recurr_state, full_state):
        ''' Given a state-tuple, returns the action distribution and any other predicted stuff '''

        next_recurr_state, action_dist, *others = self.network(recurr_state, full_state) # invoke the policy

        return (next_recurr_state, action_dist, *others)

    def evaluate(self, rollout):
        ''' Given a rollout, evaluate it against current policy '''

        rollout_new = Rollout(device=self.device)

        # TODO: Investigate this logic, it might be flawed
        for ((recur_state, full_state), action, reward), _, _ in rollout:
            if recur_state is not None:
                recur_state = recur_state.detach()
            _, action_dist, *others = self.timestep(recur_state, full_state)
            rollout_new << ((recur_state, full_state), action, reward, action_dist, *others)

        return rollout_new

    def episode(self, horizon, global_network_state=None, global_env_state=None, render=False):
        '''
        Samples and returns an entire rollout (as 'Rollout' instance).
        Arguments:
            horizon: Maximum length of the episode.
            global_state: A global state for the whole episode. (TODO: Look into this interface)
            render: A 2-tuple (bool, float) containing whether want to render and an optional time delay
        '''

        ep_reward = 0 # total reward for full episode

        state = self.reset(global_state=global_env_state) # prepares for a new episode
        state = state.unsqueeze(0)
        recurr_state = global_network_state

        rollout = Rollout(device=self.device)

        # loop for many time-steps
        for t in range(horizon):
            full_state = torch.cat([state,] if global_network_state is None else [state, global_network_state], dim=-1)
            
            # Rendering
            if render:
                self.environment.render()
            
            next_recurr_state, action_dist, *others = self.timestep(recurr_state, full_state)
            action = action_dist.sample() # sample an action
            
            # Transition to new state and retrieve a reward
            next_state, reward, done, _ = self.environment.step(*[a.to(torch.device('cpu')).numpy() for a in action])
            next_state = torch.from_numpy(next_state).float().to(self.device)
            
            rollout << ((recurr_state, full_state), action, reward, action_dist, *others) # record one experience tuple
    
            state = next_state # update current state
            state = state.unsqueeze(0)
            recurr_state = next_recurr_state # update current recurrent state
            
            if done: break

        # One last entry for the last state (sometimes required)
        full_state = torch.cat([state,] if global_network_state is None else [state, global_network_state], dim=-1)
        _, action_dist, *others = self.timestep(recurr_state, full_state)
        action = action_dist.sample()

        rollout << ((recurr_state, full_state), action, 0.0, action_dist, *others) # R = 0.0 is dummy, TODO: DO something

        return rollout
