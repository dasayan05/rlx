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
