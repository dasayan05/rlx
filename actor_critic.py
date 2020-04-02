import torch, gym, numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from models import PolicyNetwork, ValueNetwork

class Agent(object):
    def __init__(self, env):
        super().__init__()

        # Track arguments
        self.environment = env
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        # Internal objects
        self.mseloss = torch.nn.MSELoss()
        self.policynet = PolicyNetwork(self.n_states, self.n_actions)
        self.valuenet = ValueNetwork(self.n_states)
        self.pvoptim = torch.optim.Adam([*self.policynet.parameters(), *self.valuenet.parameters()])

    def reset(self):
        self.state = torch.from_numpy(self.environment.reset()).float()
        self.rewards, self.logprobs, self.values = [], [], []

    def take_action(self):
        value, policy = self.valuenet(self.state), self.policynet(self.state)
        actions = Categorical(policy)

        # sample an action
        action = actions.sample()
        
        # Transition to new state and retrieve a reward
        st, rw, done, _ = self.environment.step(action.item())
        self.state = torch.from_numpy(st).float() # update current state

        self.logprobs.append(actions.log_prob(action))
        self.rewards.append(rw)
        self.values.append(value)

        return rw, done

    def __compute_returns(self):
        self.returns = [self.rewards[-1]]
        for r in reversed(self.rewards[:-1]):
            self.returns.insert(0, r + args.gamma * self.returns[0])

        self.returns = torch.tensor(self.returns)
        self.returns = (self.returns - self.returns.mean()) / self.returns.std()

    def compute_loss(self):
        self.__compute_returns()
        self.policylosses, self.valuelosses = [], []
        for val, ret, lp in zip(self.values, self.returns, self.logprobs):
            advantage = ret - val.detach()
            self.policylosses.append(- advantage * lp)
            self.valuelosses.append(self.mseloss(val, ret.view(1,)))

        return sum(self.policylosses), sum(self.valuelosses)

def main( args ):
    from itertools import count

    # The CartPole-v0 environment from OpenAI Gym
    agent = Agent(gym.make('CartPole-v0'))
    logger = SummaryWriter(f'exp/{args.tag}')

    # average episodic reward
    avg_ep_reward = 0

    # loop for many episodes
    for episode in count(1):
        ep_reward = 0

        agent.reset() # prepares for a new episode
        # loop for many time-steps
        for t in count(1):
            if args.render and episode % args.interval == 0:
                agent.environment.render()
            
            r, done = agent.take_action()
            ep_reward += r
            if done:
                break
        
        avg_ep_reward = 0.05 * ep_reward + (1 - 0.05) * avg_ep_reward

        # Training section
        agent.pvoptim.zero_grad()

        ploss, vloss = agent.compute_loss()
        loss = ploss + vloss

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
    parser.add_argument('--interval', type=int, required=False, default=10, help='Logging freq')
    parser.add_argument('--tag', type=str, required=True, help='Identifier for experiment')

    args = parser.parse_args()
    main( args )