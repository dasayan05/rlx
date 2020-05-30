import os
import gym, torch
from tqdm import tqdm

from rlx import PGAgent, REINFORCE, ActorCritic, PPO, A2C
from rlx.policy import (DiscreteMLPPolicyValue,
                        DiscreteRNNPolicyValue,
                        DiscreteMLPPolicy,
                        DiscreteRNNPolicy)
from rlx.env import (CartPolev0, CartPolev1)

PGAlgos = {
    'rf': REINFORCE,
    'ac': ActorCritic,
    'a2c': A2C,
    'ppo': PPO
}

GYMEnvs = {
    'CartPole-v0': CartPolev0,
    'CartPole-v1': CartPolev1
}

def main( args ):
    from torch.utils.tensorboard import SummaryWriter

    environment = GYMEnvs[args.env]()
    Policy = (DiscreteRNNPolicyValue if args.algo != 'rf' else DiscreteRNNPolicy) if args.policytype == 'rnn' else \
                (DiscreteMLPPolicyValue if args.algo != 'rf' else DiscreteMLPPolicy)

    agent = PGAgent(environment, Policy, policy_kwargs={'n_hidden': 256}, device=torch.device('cuda'))

    algorithm = PGAlgos[args.algo](agent)
    train_args = {
        'horizon': args.horizon,
        'gamma': args.gamma,
        'entropy_reg': args.entropy_reg,
        'ppo_k': args.ppo_k_epochs,
        'batch_size': args.batch_size,
        'ppo_clip': args.ppo_clip,
        'render': args.render,
        'standardize_return': True
    }
    
    # logging object (TensorBoard)
    if len(args.tbdir) != 0:
        logger = SummaryWriter(os.path.join(args.base, f'{args.tbdir}/{args.tbtag}'))

    # TQDM Formatting
    TQDMBar = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' + \
                    'Reward: {postfix[0][r]:>3.2f}, ' + \
                    'Length: {postfix[0][l]:3d}]'

    with tqdm(total=args.max_episode, bar_format=TQDMBar, disable=None, postfix=[dict(r=0.,l=0)]) as tqEpisodes:
        
        # average episodic reward
        running_reward = 0.

        # loop for many episodes
        for episode in range(args.max_episode):
            
            avg_reward, avg_length = algorithm.train(None, None, **train_args)

            running_reward = 0.05 * avg_reward + (1 - 0.05) * running_reward
            if episode % args.interval == 0:
                if tqEpisodes.disable:
                    print(f'[{episode:5d}/{args.max_episode}] Running reward: {running_reward:>4.2f}, Avg. Length: {avg_length:3d}')
                if len(args.tbdir) != 0:
                    logger.add_scalar('reward', running_reward, global_step=episode)
                    logger.add_scalar('length', avg_length, global_step=episode)

            # TQDM update stuff
            if not tqEpisodes.disable:
                tqEpisodes.postfix[0]['r'] = running_reward
                tqEpisodes.postfix[0]['l'] = avg_length
                tqEpisodes.update()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=False, default='.', help='Base folder (everything is relative to this)')
    parser.add_argument('--tbdir', type=str, required=False, default='', help='folder name for TensorBoard logging (empty if no TB)')
    parser.add_argument('--tbtag', type=str, required=False, default='rltag', help='Unique identifier for experiment (for TensorBoard)')
    parser.add_argument('--algo', type=str, required=True, choices=['rf', 'ac', 'a2c', 'ppo'], help='Which algorithm to use')
    parser.add_argument('--gamma', type=float, required=False, default=0.999, help='Discount factor')
    parser.add_argument('--render', action='store_true', help='Render environment while sampling episodes')
    parser.add_argument('--policytype', type=str, required=True, choices=['rnn', 'mlp'], help='Type of policy (MLP or RNN)')
    parser.add_argument('--interval', type=int, required=False, default=10, help='Logging frequency')
    parser.add_argument('-K', '--ppo_k_epochs', type=int, required=False, default=4, help='How many iterations of trusted updates')
    parser.add_argument('--batch_size', type=int, required=False, default=8, help='Batch size')
    parser.add_argument('--entropy_reg', type=float, required=False, default=1e-2, help='Regularizer weight for entropy')
    parser.add_argument('--ppo_clip', type=float, required=False, default=0.2, help='PPO clipping parameter (usually 0.2)')
    parser.add_argument('--max_episode', type=int, required=False, default=1000, help='Maximum no. of episodes')
    parser.add_argument('--horizon', type=int, required=False, default=500, help='Maximum no. of timesteps')
    parser.add_argument('--env', type=str, required=True, choices=['CartPole-v0', 'CartPole-v1'], help='Gym environment name (string)')

    args = parser.parse_args()
    main( args )