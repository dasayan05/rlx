import os
import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import PGAgent
from policy import DiscreteMLPPolicyValue, DiscreteRNNPolicyValue

def main( args ):
    # The CartPole-v0 environment from OpenAI Gym
    agent = PGAgent(gym.make(args.env), DiscreteRNNPolicyValue, device=torch.device('cuda'))
    logger = SummaryWriter(os.path.join(args.base, f'exp/{args.tag}'))

    # average episodic reward
    running_reward = 0.

    # loop for many episodes
    for episode in range(args.max_episode):
        agent.zero_grad()
        avg_rollout_length = 0.
        for b in range(args.batch_size):
            rollout = agent.episode(args.horizon); avg_rollout_length = ((avg_rollout_length * b) + len(rollout)) / (b + 1)
            logprobs = rollout.logprobs()
            rewards = rollout.rewards()
            returns = rollout.returns(args.gamma); returns = returns.to(rollout.device)
            returns = (returns - returns.mean()) / returns.std()
            values, = rollout.others(); values = torch.cat(values).squeeze().to(rollout.device)

            advantage = returns - values.detach()
            policyloss = - advantage * logprobs
            valueloss = 0.5 * advantage.pow(2)
            loss = policyloss.sum() + valueloss.sum()
            loss /= args.batch_size
            loss.backward()
        agent.step()

        running_reward = 0.05 * rewards.sum().detach().item() + (1 - 0.05) * running_reward
        if episode % args.interval == 0:
            print(f'Running reward at episode {episode}: {running_reward:.3f}; Length: {int(avg_rollout_length):3d}')
            logger.add_scalar('avg_reward', running_reward, global_step=episode)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=False, default='.', help='Base folder (for condor)')
    parser.add_argument('--gamma', type=float, required=False, default=0.999, help='Discount factor')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--interval', type=int, required=False, default=10, help='Logging freq')
    parser.add_argument('--batch_size', type=int, required=False, default=5, help='Batch size')
    parser.add_argument('--tag', type=str, required=True, help='Identifier for experiment')
    parser.add_argument('--max_episode', type=int, required=False, default=500, help='Maximum no. of episodes')
    parser.add_argument('--horizon', type=int, required=False, default=1000, help='Maximum no. of timesteps')
    parser.add_argument('--env', type=str, required=True, help='Gym environment')

    args = parser.parse_args()
    main( args )