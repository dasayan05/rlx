import os
import gym
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent import PGAgent
from utils import compute_returns
from policy import DiscreteMLPPolicy, DiscreteRNNPolicy

def main( args ):
    Policy = DiscreteRNNPolicy if args.policytype == 'rnn' else DiscreteMLPPolicy
    agent = PGAgent(gym.make(args.env), Policy, device=torch.device('cuda'))
    logger = SummaryWriter(os.path.join(args.base, f'exp/{args.tag}'))

    # average episodic reward
    running_reward = 0.

    # TQDM Formatting
    TQDMBar = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' + \
                    'Reward: {postfix[0][r]:>3.2f}, ' + \
                    'Length: {postfix[0][l]:3d}]'

    with tqdm(total=args.max_episode, bar_format=TQDMBar, disable=None, postfix=[dict(r=0.,l=0)]) as tqEpisodes:
        
        # loop for many episodes
        for episode in range(args.max_episode):
            avg_length = 0
            
            agent.zero_grad()
            for b in range(args.batch_size):
                rollout = agent.episode(args.horizon)
                avg_length = ((avg_length * b) + len(rollout)) // (b + 1)
                rewards, logprobs = rollout.rewards, rollout.logprobs
                returns = compute_returns(rewards, args.gamma)

                policyloss = - returns * logprobs
                loss = policyloss.sum()
                loss /= args.batch_size
                loss.backward()
            agent.step()

            running_reward = 0.05 * rewards.sum().detach().item() + (1 - 0.05) * running_reward
            if episode % args.interval == 0:
                if tqEpisodes.disable:
                    print(f'[{episode:5d}/{args.max_episode}] Running reward: {running_reward:>4.2f}, Avg. Length: {avg_length:3d}')
                logger.add_scalar('reward', running_reward, global_step=episode)
                logger.add_scalar('length', avg_length, global_step=episode)
            
            # TQDM update stuff
            tqEpisodes.postfix[0]['r'] = running_reward
            tqEpisodes.postfix[0]['l'] = avg_length
            tqEpisodes.update()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=False, default='.', help='Base folder (for condor)')
    parser.add_argument('--gamma', type=float, required=False, default=0.999, help='Discount factor')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--policytype', type=str, required=True, choices=['rnn', 'mlp'], help='Type of policy')
    parser.add_argument('--tag', type=str, required=True, help='Identifier for experiment')
    parser.add_argument('--interval', type=int, required=False, default=10, help='Logging freq')
    parser.add_argument('--batch_size', type=int, required=False, default=8, help='Batch size')
    parser.add_argument('--max_episode', type=int, required=False, default=500, help='Maximum no. of episodes')
    parser.add_argument('--horizon', type=int, required=False, default=1000, help='Maximum no. of timesteps')
    parser.add_argument('--env', type=str, required=True, help='Gym environment')

    args = parser.parse_args()
    main( args )