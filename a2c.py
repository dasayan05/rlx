import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import PGAgent
from policy import DiscreteMLPPolicyValue

class PGA2C(PGAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def timestep(self, state):
        value, action_dist = self(state)

        action = action_dist.sample() # sample an action
        
        # Transition to new state and retrieve a reward
        st, rw, done, _ = self.environment.step(action.item())
        state = torch.from_numpy(st).float().to(self.device) # update current state

        self.logprobs.append(action_dist.log_prob(action).view(1,))
        self.rewards.append(rw)
        self.values.append(value)
        self.entropy.append( - sum(action_dist.probs * action_dist.probs.log()) )

        return state, rw, done

    def compute_loss(self):
        self.returns = [] # Bootstrapped
        v = self.values[-1]
        for t in reversed(range(len(self.rewards))):
            self.returns.insert(0, self.rewards[t] + args.gamma * v)
            v = self.returns[0]

        self.values = torch.cat(self.values, 0).to(self.device)
        self.returns = torch.cat(self.returns, 0).to(self.device)
        self.returns = (self.returns - self.returns.mean()) / self.returns.std()
        self.logprobs = torch.cat(self.logprobs, 0).to(self.device)

        advantage = self.returns - self.values[:-1]
        policyloss = - advantage.detach() * self.logprobs
        valueloss = 0.5 * advantage.pow(2)

        return policyloss.sum() + valueloss.sum() - sum(self.entropy) * 0.01

def main( args ):
    # The CartPole-v0 environment from OpenAI Gym
    agent = PGA2C(gym.make(args.env), DiscreteMLPPolicyValue,
        storages=['rewards', 'logprobs', 'values', 'entropy'], device=torch.device('cuda'))
    logger = SummaryWriter(f'exp/{args.tag}')

    # average episodic reward
    running_reward = 0

    # loop for many episodes
    for episode in range(args.max_episode):
        ep_reward, last_state = agent.episode(args.horizon, render=args.render, interval=args.interval)
        # little hack, need to fix the interface
        value = agent.network.valuenet(last_state); agent.values.append( value )
        agent.train()
        
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        if episode % args.interval == 0:
            print(f'Running reward at episode {episode}: {running_reward}')
            logger.add_scalar('avg_reward', running_reward, global_step=episode)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, required=False, default=0.99, help='Discount factor')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--interval', type=int, required=False, default=10, help='Logging freq')
    parser.add_argument('--tag', type=str, required=True, help='Identifier for experiment')
    parser.add_argument('--max_episode', type=int, required=False, default=500, help='Maximum no. of episodes')
    parser.add_argument('--horizon', type=int, required=False, default=1000, help='Maximum no. of timesteps')
    parser.add_argument('--env', type=str, required=True, help='Gym environment')

    args = parser.parse_args()
    main( args )