import torch, gym, numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

class PVNetwork(torch.nn.Module):
    def __init__(self, n_states, n_actions, n_affine = 128):
        super().__init__()

        # Track arguments for further use
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_affine = n_affine

        # Layer definitions
        self.affine = torch.nn.Linear(self.n_states, self.n_affine)
        self.pi = torch.nn.Linear(self.n_affine, self.n_actions)
        self.value = torch.nn.Linear(self.n_affine, 1)
    
    def forward(self, x):
        h = F.relu(self.affine(x))
        p = F.softmax(self.pi(h), dim=-1)
        v = self.value(h)
        return v, p

class Agent(object):
    def __init__(self, env):
        super().__init__()

        # Track arguments
        self.environment = env
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        # Internal objects
        self.pvnet = PVNetwork(self.n_states, self.n_actions)
        self.pvoptim = torch.optim.Adam(self.pvnet.parameters())

    def reset(self):
        self.state = torch.from_numpy(self.environment.reset()).float()
        self.rewards, self.logprobs, self.values, self.entropy = [], [], [], []

    def take_action(self):
        value, policy = self.pvnet(self.state)
        actions = Categorical(policy)

        # sample an action
        action = actions.sample()
        
        # Transition to new state and retrieve a reward
        st, rw, done, _ = self.environment.step(action.item())
        self.state = torch.from_numpy(st).float() # update current state

        self.logprobs.append(actions.log_prob(action))
        self.rewards.append(torch.tensor(rw, requires_grad=False))
        self.values.append(value)
        self.entropy.append( - sum(policy.mean() * torch.log(policy)) )

        return rw, done

    def __compute_returns(self):
        self.returns = [] # Bootstrapped
        v = self.values[-1]
        for t in reversed(range(len(self.rewards))):
            self.returns.insert(0, self.rewards[t] + args.gamma * v)
            v = self.returns[0]

    def compute_loss(self):
        self.__compute_returns()
        self.policylosses, self.valuelosses = [], []
        for val, ret, lp in zip(self.values[:-1], self.returns, self.logprobs):
            advantage = ret - val
            self.policylosses.append(- advantage.detach() * lp)
            # breakpoint()
            self.valuelosses.append(0.5 * advantage.pow(2))

        return sum(self.policylosses), sum(self.valuelosses), sum(self.entropy)

def main( args ):
    from itertools import count

    # The CartPole-v1 environment from OpenAI Gym
    agent = Agent(gym.make('CartPole-v1'))
    logger = SummaryWriter(f'exp/{args.tag}')

    # average episodic reward
    avg_ep_reward = 0

    # loop for many episodes
    for episode in count(1):
        ep_reward = 0

        agent.reset() # prepares for a new episode
        # loop for many time-steps
        for t in range(10000):
            if args.render and episode % args.interval == 0:
                agent.environment.render()
            
            r, done = agent.take_action()
            ep_reward += r
            if done:
                break
        
        # One last 'value' is needed for bootstrapping
        value, _ = agent.pvnet(agent.state)
        agent.values.append( value )
        
        avg_ep_reward = 0.05 * ep_reward + (1 - 0.05) * avg_ep_reward

        # Training section
        agent.pvoptim.zero_grad()

        ploss, vloss, eloss = agent.compute_loss()
        loss = ploss + vloss + 0.001 * eloss

        loss.backward()

        agent.pvoptim.step()

        if episode % args.interval == 0:
            print(f'Average reward till this episode: {avg_ep_reward}')
            logger.add_scalar('avg_reward', avg_ep_reward, global_step=episode)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, required=False, default=0.99, help='Discount factor')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--interval', type=int, required=False, default=50, help='Logging freq')
    parser.add_argument('--tag', type=str, required=True, help='Identifier for experiment')

    args = parser.parse_args()
    main( args )