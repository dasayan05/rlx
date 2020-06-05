import torch

def compute_returns(rewards, gamma = 1.0, standardize = False):
    '''
    Computes 'return' (sum of future rewards, optionally discounted)
    Arguments:
        rewards: A sequence of rewards
        gamma: (optional) Discount factor
    '''
    returns = [rewards[-1].view(1,),]
    for r in reversed(rewards[:-1]):
        returns.insert(0, (r + gamma * returns[0]).view(1,))
    returns = torch.cat(returns, dim=-1)
    return (returns - returns.mean()) / returns.std() if standardize else returns

def compute_bootstrapped_returns(rewards, end_v, gamma = 1.0, standardize = False):
    returns = []
    v = end_v
    for t in reversed(range(len(rewards))):
        returns.insert(0, rewards[t] + gamma * v)
        v = returns[0]
    returns = torch.cat(returns, dim=-1)
    return (returns - returns.mean()) / returns.std() if standardize else returns
