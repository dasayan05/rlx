import torch, gym, numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from models import PolicyNetwork

class Agent(object):
    def __init__(self, env):
        super().__init__()

        # Track arguments
        self.environment = env
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Internal objects
        self.policynet = PolicyNetwork(self.n_states, self.n_actions).to(self.device)
        self.policyoptim = torch.optim.Adam(self.policynet.parameters())

    def reset(self):
        self.state = torch.from_numpy(self.environment.reset()).float().to(self.device)
        self.rewards, self.logprobs = [], []

    def take_action(self):
        actions = Categorical(self.policynet(self.state))

        # sample an action
        action = actions.sample()
        
        # Transition to new state and retrieve a reward
        st, rw, done, _ = self.environment.step(action.item())
        self.state = torch.from_numpy(st).float().to(self.device)

        self.logprobs.append(actions.log_prob(action).view(1,))
        self.rewards.append(rw)

        return rw, done

    def __compute_returns(self):
        self.returns = [self.rewards[-1]]
        for r in reversed(self.rewards[:-1]):
            self.returns.insert(0, r + args.gamma * self.returns[0])

    def compute_loss(self):
        self.__compute_returns()
        self.returns = torch.tensor(self.returns, device=self.device)
        self.returns = (self.returns - self.returns.mean()) / self.returns.std()
        self.logprobs = torch.cat(self.logprobs, 0)
        
        policyloss = - self.returns * self.logprobs
        return policyloss.sum()

def main( args ):
    # The CartPole-v0 environment from OpenAI Gym
    agent = Agent(gym.make(args.env))
    logger = SummaryWriter(f'exp/{args.tag}')

    # average episodic reward
    running_reward = 0

    # loop for many episodes
    for episode in range(args.max_episode):
        ep_reward = 0

        agent.reset() # prepares for a new episode
        # loop for many time-steps
        for t in range(1000):
            if args.render and episode % args.interval == 0:
                agent.environment.render()
            
            r, done = agent.take_action()
            ep_reward += r
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # Training section
        agent.policyoptim.zero_grad()
        loss = agent.compute_loss()
        loss.backward()
        agent.policyoptim.step()

        if episode % args.interval == 0:
            print(f'Running reward at episode {episode}: {running_reward}')
            logger.add_scalar('avg_reward', running_reward, global_step=episode)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, required=False, default=0.99, help='Discount factor')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--tag', type=str, required=True, help='Identifier for experiment')
    parser.add_argument('--interval', type=int, required=False, default=10, help='Logging freq')
    parser.add_argument('--max_episode', type=int, required=False, default=500, help='Maximum no. of episodes')
    parser.add_argument('--env', type=str, required=True, help='Gym environment')

    args = parser.parse_args()
    main( args )