import os
import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import PGAgent
from policy import DiscreteMLPPolicy, DiscreteRNNPolicy

class PGReinforce(PGAgent):

    def timestep(self, state):
        action_dist = self(state) # invoke the policy

        action = action_dist.sample() # sample an action
        
        # Transition to new state and retrieve a reward
        st, rw, done, _ = self.environment.step(action.item())
        state = torch.from_numpy(st).float().to(self.device) # update current state

        self.logprobs.append(action_dist.log_prob(action).view(1,))
        self.rewards.append(rw)

        return state, rw, done

    def compute_loss(self):
        # compute return (sum of future rewards)
        self.returns = [self.rewards[-1]]
        for r in reversed(self.rewards[:-1]):
            self.returns.insert(0, r + args.gamma * self.returns[0])

        self.returns = torch.tensor(self.returns, device=self.device)
        self.returns = (self.returns - self.returns.mean()) / self.returns.std()
        self.logprobs = torch.cat(self.logprobs, 0)
        
        policyloss = - self.returns * self.logprobs
        return policyloss.sum()

def main( args ):
    # The CartPole-v0 environment from OpenAI Gym
    agent = PGReinforce(gym.make(args.env), DiscreteRNNPolicy,
        storages=['rewards', 'logprobs', 'values'], device=torch.device('cuda'))
    logger = SummaryWriter(os.path.join(args.base, f'exp/{args.tag}'))

    # average episodic reward
    running_reward = 0

    # loop for many episodes
    for episode in range(args.max_episode):
        ep_reward, _, length = agent.episode(args.horizon, render=args.render, interval=args.interval)
        agent.train()

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        if episode % args.interval == 0:
            print(f'Running reward at episode {episode}: {running_reward}; Length: {length}')
            logger.add_scalar('avg_reward', running_reward, global_step=episode)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=False, default='.', help='Base folder (for condor)')
    parser.add_argument('--gamma', type=float, required=False, default=0.99, help='Discount factor')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--tag', type=str, required=True, help='Identifier for experiment')
    parser.add_argument('--interval', type=int, required=False, default=10, help='Logging freq')
    parser.add_argument('--max_episode', type=int, required=False, default=500, help='Maximum no. of episodes')
    parser.add_argument('--horizon', type=int, required=False, default=1000, help='Maximum no. of timesteps')
    parser.add_argument('--env', type=str, required=True, help='Gym environment')

    args = parser.parse_args()
    main( args )