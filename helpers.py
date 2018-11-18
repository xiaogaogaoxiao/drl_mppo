import random
import numpy as np


def argmax(pairs):

    """
    Given an iterable of pairs return the key corresponding to the greatest
    value.

    Parameters:
    :param pairs : iterable of pairs

    Outputs:
    :returns argmax_val : key corresponding to greatest value
    """

    argmax_val = max(pairs, key=lambda x: x[1])[0]

    return argmax_val


def argmax_index(values):

    """
    Given an iterable of values return the index of the greatest value.

    Parameters
    ----------
    :param values : iterable
           Iterable of values.

    Returns
    -------
    :returns argmax_idx : int
             Index of the greatest value in the input iterable.
    """

    argmax_idx = argmax(enumerate(values))

    return argmax_idx


def eps_greedy(epsilon, action_space, prediction):

    """
    Makes an epsilon greedy choice over the action space.

    Parameters
    ----------
    :param epsilon : float
           Value of epsilon for exploration.
    :param action_space : ndarray
           All available portfolio allocation choices.
    :param prediction : ndarray
           Predicted Q-values for all actions in the action space.

    Returns
    -------
    :returns action : ndarray
             Desired portfolio allocation.
    """

    eps = np.random.rand()

    if eps < epsilon:
        action = random.choice(action_space)  # random action
    else:
        action = action_space[argmax_index(prediction)]  # greedy action

    return action


def compute_rl_return(rewards, discount_rate):

    """
    Returns discounted rewards.

    Parameters
    ----------
    :param rewards : np.array
           Rewards at each timestep of the episode.
    :param discount_rate : float
           Discount rate for future rewards.

    Returns
    -------
    :returns rl_return : np.array
             RL returns for each timestep of the episode.
    """

    rl_return = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0

    for t in reversed(range(len(rewards))):
        running_add = running_add * discount_rate + rewards[t]
        rl_return[t] = running_add

    return rl_return.reshape(-1)


def all_close(iterator):

    """
    Checks if the elements of the iterator are all close to each other.

    Parameters
    ----------
    :param iterator : iterator

    Returns
    -------
    :returns _ : boolean
    """

    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(np.allclose(first, rest) for rest in iterator)

    except StopIteration:
        return True


def compute_trade(portfolio, desired_allocation, tcost):

    """
    Computes the trade necessary ot arrive at a desired portfolio allocation.

    Parameters
    ----------
    :param portfolio : ndarray
           Array containing absolute size of portfolio holdings for risk-free
           and risky asset.
    :param desired_allocation : float
           Desired allocation to the risky asset after the trade incl.
           transaction costs.
    :param tcost : float
           Linear transaction cost factor.

    Returns
    -------
    :returns trade : ndarray
             Changes in positions (=trades) to be made to arrive at desired
             portfolio allocation.
    """

    p = portfolio
    a = desired_allocation

    if p[1] / np.sum(p) <= a:  # need to buy risky asset
        trade = [-(p[1] - a * np.sum(p)) /
                 (-1.0 - a * tcost),
                 (p[1] - a * np.sum(p)) /
                 (-1.0 - a * tcost)]
    elif p[1] / np.sum(p) > a:  # need to sell risky asset
        trade = [-(p[1] - a * np.sum(p))
                 / (-1.0 + a * tcost),
                 (p[1] - a * np.sum(p)) /
                 (-1.0 + a * tcost)]
    else:
        trade = [0, 0]

    return trade


def crra_utility(w, theta):

    """
    Computes the CRRA utility for a given level of wealth and risk aversion.

    Parameters
    ----------
    :param w : list
           Different wealth levels.
    :param theta : int
           Investor's risk aversion coefficient.

    Returns
    -------
    :returns u : ndarray
             Utility levels corresponding to input wealth levels given the
             input risk aversion coefficient.
    """

    if theta == 1.:
        u = np.log(w)
    else:
        u = (np.asarray(w) ** (1. - theta) - 1.) / (1. - theta)

    return u


def compute_opt_weight(env, t):

    """
    Computes the optimal weight of the risky asset for a given environment
    at a time t.

    Arguments
    ---------
    :param env : Environment instance
           Environment instance specifying the RL environment.
    :param t : int
           Period in episode for which the optimal weight of the risky asset
           should be computed.

    Returns
    -------
    :returns opt : float
             The optimal weight of the risky asset in the given environment
             in period t.
    """

    env.time = t

    # regime in time t:
    idxt = [t in v["periods"] for v in env.regimes.values()].index(True)
    rt = list(env.regimes.keys())[idxt]

    mu = env.regimes[rt]["mu"]
    sigma = env.regimes[rt]["sigma"]

    opt = (mu[1] - mu[0] + (sigma[1]**2)/2) / (env.theta * sigma[1]**2)

    return opt
