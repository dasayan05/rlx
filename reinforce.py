import os
import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import PGAgent
from utils import compute_returns
from policy import DiscreteMLPPolicy, DiscreteRNNPolicy

def main( args ):
    agent = PGAgent(gym.make(args.env), DiscreteRNNPolicy, device=torch.device('cuda'))
    logger = SummaryWriter(os.path.join(args.base, f'exp/{args.tag}'))

    # average episodic reward
    running_reward = 0.

    # loop for many episodes
    for episode in range(args.max_episode):
        avg_length = 0
        
        agent.zero_grad()
        for b in range(args.batch_size):
            rollout = agent.episode(args.horizon)
            avg_length = ((avg_length * b) + len(rollout)) // (b + 1)
            rewards, logprobs = rollout.rewards, rollout.logprobs
            returns = compute_returns(rewards, args.gamma).to(rollout.device)

            policyloss = - returns * logprobs
            loss = policyloss.sum()
            loss /= args.batch_size
            loss.backward()
        agent.step()

        running_reward = 0.05 * rewards.sum().detach().item() + (1 - 0.05) * running_reward
        if episode % args.interval == 0:
            print(f'[{episode:5d}/{args.max_episode}] Running reward: {running_reward:>4.2f}, Avg. Length: {avg_length:3d}')
            logger.add_scalar('avg_reward', running_reward, global_step=episode)
            logger.add_scalar('length', avg_length, global_step=episode)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=False, default='.', help='Base folder (for condor)')
    parser.add_argument('--gamma', type=float, required=False, default=0.999, help='Discount factor')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--tag', type=str, required=True, help='Identifier for experiment')
    parser.add_argument('--interval', type=int, required=False, default=10, help='Logging freq')
    parser.add_argument('--batch_size', type=int, required=False, default=8, help='Batch size')
    parser.add_argument('--max_episode', type=int, required=False, default=500, help='Maximum no. of episodes')
    parser.add_argument('--horizon', type=int, required=False, default=1000, help='Maximum no. of timesteps')
    parser.add_argument('--env', type=str, required=True, help='Gym environment')

    args = parser.parse_args()
    main( args )