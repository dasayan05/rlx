# rlx: A modular and generic Deep RL library
---

### Introcution

Few months back, I was looking for a good library to get started with Deep Reinforement Learning (Deep RL). After a bit of searching, I found out that there are only few of them. Moreover, they are geared more towards reproduction of state-of-the-art algorithms on very specific tasks (e.g. Atari games etc.). There was no library which is truely generic. Also, I noticed that different researchers/developers have their own standalone implementation of different RL algorithms with very different abstractions that make it difficult to adopt any one of them.

I decided to start writing one myself on top of [PyTorch](https://pytorch.org/). I intend to make this one
1. Generic (i.e., can be adopted for any task at hand)
2. Modular (i.e., have meaningful and intuitive abstraction)
3. Easy to write new algorithms, i.e., algorithms can be literally translated to code

Here's an example of **PPO** implementation with `rlx`

```
base_rollout = agent.episode(horizon) # sample an episode as a 'Rollout' object
base_rewards, base_logprobs = base_rollout.rewards, base_rollout.logprobs # 'rewards' and 'logprobs' for all timesteps
base_returns = compute_returns(base_rewards, gamma) # Monte-carlo estimates of 'returns'

for _ in range(k_epochs):
	rollout = agent.evaluate(base_rollout) # 'evaluate' an episode against a policy and get a new 'Rollout' object
	logprobs, entropy = rollout.logprobs, rollout.entropy # get 'logprobs' and 'entropy' for all timesteps
	values, = rollout.others # .. also 'value' estimates
	
	ratios = (logprobs - base_logprobs.detach()).exp()
	advantage = base_returns - values.squeeze()
	policyloss = - torch.min(ratios, torch.clamp(ratios, 1 - clip, 1 + clip)) * advantage.detach()
	valueloss = advantage.pow(2)
	entropyloss = - entropy_reg * entropy
	loss = policyloss.sum() + valueloss.sum() + entropyloss.sum()
	agent.zero_grad()
	loss.backward()
	agent.step()
```

This is all you have to write to get PPO running. Its extremely easy to make manual engineering into the algorithms. The design is centered around the primary data structure "Rollout" which holds a sequence of experience tuples containing action distributions, value-estimates and internally keeps track of the computation graph. Right now it only has 4 Policy Gradient algorithms

1. Vanilla REINFORCE
2. REINFORCE with Value-baseline
3. A2C
4. PPO with clipping

**If I have time, I will write more docs.**

## Installation and usage

Right now, there is no `pip` package, its just this repo. You can install it by cloning it and doing
```
pip install .
```

For example usage, follow the `rlx/__main__.py` script. Also, after installation, you can test an algorithm by
```
python -m rlx --algo ppo --policytype mlp -K4 --env CartPole-v0 --clip 0.2
```

#### TODO:

1. More SOTA Policy Gradient algorithms (ACER, TRPO, etc.) and also Q-learning (DeepQ etc.)
2. Multiprocessing/Parallelization support