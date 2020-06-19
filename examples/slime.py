# The "Slime Volleball" game from David Ha
# get it from here https://github.com/hardmaru/slimevolleygym

import slimevolleygym as slime
import time, gym
import matplotlib.pyplot as plt
import torch, numpy as np
from torch.utils.tensorboard import SummaryWriter

from rlx import Environment, Parametric, PGAgent, ActionDistribution
from rlx import REINFORCE, ActorCritic, A2C, PPO

class SlimeMLPPolicy(Parametric):
    def __init__(self, observation_space, action_spaces, *, n_hidden=16):
        ''' Constructs every learnable item from environment specifications. '''

        super().__init__(observation_space, action_spaces)
        assert len(self.action_spaces) == 1, "Only one action component"

        # Track arguments for further use
        self.n_state = self.observation_space.shape[0]
        self.n_action = self.action_spaces[0].n

        # Layer definitions
        self.layer1 = torch.nn.Linear(self.n_state, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_hidden // 2)
        self.layerout = torch.nn.Linear(n_hidden // 2, self.n_action)

        # state-value
        self.V = torch.nn.Linear(n_hidden // 2, 1)

    def forward(self, *states):
        _, state = states
        l1 = torch.nn.functional.leaky_relu(self.layer1(state))
        l2 = torch.nn.functional.leaky_relu(self.layer2(l1))
        action_probs = torch.softmax(self.layerout(l2), -1)
        act = torch.distributions.Categorical(action_probs)

        v = self.V(l2)
        return None, ActionDistribution(act), v

class SlimeVolleyv0(Environment):
    action_map = [
        [0, 0, 0], # NOOP
        [1, 0, 0], # LEFT (forward)
        [1, 0, 1], # UPLEFT (forward jump)
        [0, 0, 1], # UP (jump)
        [0, 1, 1], # UPRIGHT (backward jump)
        [0, 1, 0], # RIGHT (backward)
    ]
    
    def __init__(self, base='.'):
        # The 'SlimeVolley-v0' environment from David Ha (@hardmaru)
        self._gymenv = gym.make('SlimeVolley-v0')

        self.observation_space = self._gymenv.observation_space
        self.action_spaces = (gym.spaces.Discrete(6),)

        # The opponent policy (exactly same architecture as the agent policy)
        self.opponent = SlimeMLPPolicy(self.observation_space,
            self.action_spaces, n_hidden=64)
        self.opponent.requires_grad_(False) # no need to train
        # negate all x-coordinate to enable tranferrability
        self.ball_state_mask = torch.tensor([-1., 1., -1., 1.],
                dtype=torch.float32, requires_grad=False)

        self.ep_count = 0
        self.base = base

    def reset(self, global_state=None):
        self.ep_count += 1
        self.step_count = 0
        self.last_state = self._gymenv.reset() # keep track of last state, needed for opponent
        return self.last_state

    def step(self, *actions):
        self.step_count += 1
        assert len(actions) == 1, 'SlimeVolley-v0 has only one action component'
        action = SlimeVolleyv0.action_map[actions[0].item()] # agent's action

        # select opponent's move
        with torch.no_grad():
            state = torch.from_numpy(self.last_state.astype(np.float32))
            state[:4], state[-4:] = state[-4:], state[:4] # agent and opponent swaps
            state[4:8] = state[4:8] * self.ball_state_mask # ball direction reverts
            
            _, opp_act_dist, _ = self.opponent(None, state.view(1, -1))
        opp_action = opp_act_dist.sample()[0].item()
        opp_action = SlimeVolleyv0.action_map[opp_action]

        obs, _, _, info = self._gymenv.step(action, opp_action)

        is_dead = info['ale.lives'] < 5
        is_opp_dead = info['ale.otherLives'] < 5
        
        R = -1e-2 # regular reward
        R += +10. if is_opp_dead else 0.
        R += -10. if is_dead else 0.

        done = is_dead or is_opp_dead
        self.last_state = obs
        return obs, R, done, { }

    def close(self):
        self._gymenv.close()

    def render(self, **kwargs):
        x = self._gymenv.render(mode='rgb_array',**kwargs)
        ep_path = os.path.join(self.base, 'ep_' + str(self.ep_count))
        if not os.path.exists(ep_path):
            os.mkdir(ep_path)
        plt.imsave(os.path.join(ep_path, f'{self.step_count}.png'), x[:,:,:])
        # time.sleep(0.01)

ALGOMAP = {
    'rf': REINFORCE,
    'ac': ActorCritic,
    'a2c': A2C,
    'ppo': PPO
}

if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=False, default='.', help='Base folder')
    parser.add_argument('--expdir', type=str, required=False, default='exp')
    parser.add_argument('--tag', type=str, required=True, help='Unique tag for experiments')
    parser.add_argument('--algo', type=str, required=True, choices=['rf', 'ac', 'a2c', 'ppo'])
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    args = parser.parse_args()
    logpath = os.path.join(args.base, args.expdir, 'slime_' + args.tag)

    env = SlimeVolleyv0(base=logpath)
    network = SlimeMLPPolicy(env.observation_space, env.action_spaces, n_hidden=64)
    agent = PGAgent(env)
    if torch.cuda.is_available():
        network, agent = network.cuda(), agent.cuda()
    algorithm = ALGOMAP[args.algo](agent, network, optimizer='adam', optim_kwargs={'lr':1e-4})

    logger = SummaryWriter(logpath, flush_secs=10)

    agent_modelpath = os.path.join(logpath, 'agent.pth')
    if os.path.exists(agent_modelpath):
        agent_model = torch.load(agent_modelpath)
        network.load_state_dict(agent_model)
        print('agent loaded')
    oppn_modelpath = os.path.join(logpath, 'opponent.pth')
    if os.path.exists(oppn_modelpath):
        oppn_model = torch.load(oppn_modelpath)
        env.opponent.load_state_dict(oppn_model) # transfer the knowledge for self-play
        env.opponent.requires_grad_(False)
        print('opponent loaded')

    running_reward = 0.
    running_length = 0.
    for e in range(int(1e5)):
        r, l = algorithm.train(None, None, batch_size=args.batch_size, horizon=750,
            grad_clip=0.1, gamma=0.999, standardize_return=False, render=False)
        running_reward = 0.05 * r + (1 - 0.05) * running_reward
        running_length = 0.05 * l + (1 - 0.05) * running_length
        if e % 10 == 0:
            logger.add_scalar('reward', running_reward, global_step=e)
            logger.add_scalar('length', running_length, global_step=e)
        if e % 1000 == 0:
            torch.save(network.state_dict(), os.path.join(logpath, f'agent_{e//1000}.pth'))
