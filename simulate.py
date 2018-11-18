import numpy as np
import math
import copy
from helpers import compute_trade, argmax_index, compute_opt_weight


def portfolio_safe(eval_episodes, environment):

    """
    Simulates a no-risk portfolio strategy in a given environment.

    Parameters
    ----------
    :param eval_episodes : int
           Number of episodes to simulate.
    :param environment : Env instance
           Environment in which to simulate the portfolio strategy.

    Returns
    -------
    :returns final_u : ndarray
             Array containing utility of terminal wealth for each simulated
             episode.
    :returns alloc_to_risk : ndarray
             Array containing the share of wealth invested into the risky
             asset in each period for all simulated episodes.
    :returns ret : ndarray
             Array containing the simple gross returns realized in each
             period for all simulated episodes.
    """

    print("Simulating safe portfolio strategy.")

    env = copy.deepcopy(environment)

    final_u = []
    alloc_to_risk = [[] for _ in range(eval_episodes)]
    ret = []

    np.random.seed(111)

    for episode in range(eval_episodes):
        env.reset()
        while not env.done:
            np.random.rand()  # random context
            action = 0.
            trade = compute_trade(env.p, action, env.tcost)
            env.trade(trade)
            assert math.isclose(env.p[1] / np.sum(env.p), action)
            alloc_to_risk[episode].append(env.p[1]/np.sum(env.p))
            sgr = env.update()
            ret.append(sgr)
        final_u.append(env.get_utility())

    final_u = np.array(final_u)
    alloc_to_risk = np.array(alloc_to_risk)
    ret = np.array(ret)

    return final_u, alloc_to_risk, ret


def portfolio_risky(eval_episodes, environment):

    """
    Simulates a full-risk portfolio strategy in a given environment.

    Parameters
    ----------
    :param eval_episodes : int
           Number of episodes to simulate.
    :param environment : Env instance
           Environment in which to simulate the portfolio strategy.

    Returns
    -------
    :returns final_u : ndarray
             Array containing utility of terminal wealth for each simulated
             episode.
    :returns alloc_to_risk : ndarray
             Array containing the share of wealth invested into the risky
             asset in each period for all simulated episodes.
    :returns ret : ndarray
             Array containing the simple gross returns realized in each
             period for all simulated episodes.
    """

    print("Simulating risky portfolio strategy.")

    env = copy.deepcopy(environment)

    final_u = []
    alloc_to_risk = [[] for _ in range(eval_episodes)]
    ret = []

    np.random.seed(111)

    for episode in range(eval_episodes):
        env.reset()
        while not env.done:
            np.random.rand()  # random context
            action = 1.
            trade = compute_trade(env.p, action, env.tcost)
            env.trade(trade)
            assert math.isclose(env.p[1] / np.sum(env.p), action)
            alloc_to_risk[episode].append(env.p[1]/np.sum(env.p))
            sgr = env.update()
            ret.append(sgr)
        final_u.append(env.get_utility())

    final_u = np.array(final_u)
    alloc_to_risk = np.array(alloc_to_risk)
    ret = np.array(ret)

    return final_u, alloc_to_risk, ret


def portfolio_myopic(eval_episodes, environment):

    """
    Simulates the optimal myopic portfolio strategy in a given environment.

    Parameters
    ----------
    :param eval_episodes : int
           Number of episodes to simulate.
    :param environment : Env instance
           Environment in which to simulate the portfolio strategy.

    Returns
    -------
    :returns final_u : ndarray
             Array containing utility of terminal wealth for each simulated
             episode.
    :returns alloc_to_risk : ndarray
             Array containing the share of wealth invested into the risky
             asset in each period for all simulated episodes.
    :returns ret : ndarray
             Array containing the simple gross returns realized in each
             period for all simulated episodes.
    """

    print("Simulating myopic portfolio strategy.")

    env = copy.deepcopy(environment)

    final_u = []
    alloc_to_risk = [[] for _ in range(eval_episodes)]
    ret = []

    np.random.seed(111)

    for episode in range(eval_episodes):
        env.reset()
        while not env.done:

            # compute optimal allocation to risky asset:
            optimal_risky_weight = compute_opt_weight(env, env.time)

            # no shorting constraint:
            if optimal_risky_weight > 1:
                optimal_risky_weight = 1.0
            elif optimal_risky_weight < 0:
                optimal_risky_weight = 0.0

            np.random.rand()  # random context
            trade = compute_trade(env.p, optimal_risky_weight, env.tcost)
            env.trade(trade)
            assert math.isclose(env.p[1] / np.sum(env.p), optimal_risky_weight)
            alloc_to_risk[episode].append(env.p[1]/np.sum(env.p))
            sgr = env.update()
            ret.append(sgr)
        final_u.append(env.get_utility())

    final_u = np.array(final_u)
    alloc_to_risk = np.array(alloc_to_risk)
    ret = np.array(ret)

    return final_u, alloc_to_risk, ret


def portfolio_qtab(eval_episodes, environment, agent):

    """
    Simulates a portfolio strategy suggested by a trained tabular Q-learning
    agent in a given environment.

    Parameters
    ----------
    :param eval_episodes : int
           Number of episodes to simulate.
    :param environment : Env instance
           Environment in which to simulate the portfolio strategy.
    :param agent : AgentQtab instance
           Tabular Q-learning agent (preferably trained in the same
           environment as is simulated).

    Returns
    -------
    :returns final_u : ndarray
             Array containing utility of terminal wealth for each simulated
             episode.
    :returns alloc_to_risk : ndarray
             Array containing the share of wealth invested into the risky
             asset in each period for all simulated episodes.
    :returns ret : ndarray
             Array containing the simple gross returns realized in each
             period for all simulated episodes.
    """

    print("Simulating tabular Q-learning portfolio strategy.")

    env = copy.deepcopy(environment)

    final_u = []
    alloc_to_risk = [[] for _ in range(eval_episodes)]
    ret = []

    np.random.seed(111)

    for episode in range(eval_episodes):
        env.reset()
        while not env.done:
            np.random.rand()  # random context
            action = agent.action_space[argmax_index(agent.q_tab)]
            trade = compute_trade(env.p, action, env.tcost)
            env.trade(trade)
            assert math.isclose(env.p[1] / np.sum(env.p), action)
            alloc_to_risk[episode].append(env.p[1]/np.sum(env.p))
            sgr = env.update()
            ret.append(sgr)
        final_u.append(env.get_utility())

    final_u = np.array(final_u)
    alloc_to_risk = np.array(alloc_to_risk)
    ret = np.array(ret)

    return final_u, alloc_to_risk, ret


def portfolio_dqn(eval_episodes, environment, agent):

    """
    Simulates a portfolio strategy suggested by a trained DQN
    agent in a given environment.

    Parameters
    ----------
    :param eval_episodes : int
           Number of episodes to simulate.
    :param environment : Env instance
           Environment in which to simulate the portfolio strategy.
    :param agent : AgentDQN instance
           DQN agent (preferably trained in the same environment as is
           simulated).

    Returns
    -------
    :returns final_u : ndarray
             Array containing utility of terminal wealth for each simulated
             episode.
    :returns alloc_to_risk : ndarray
             Array containing the share of wealth invested into the risky
             asset in each period for all simulated episodes.
    :returns ret : ndarray
             Array containing the simple gross returns realized in each
             period for all simulated episodes.
    """

    print("Simulating DQN portfolio strategy.")

    env = copy.deepcopy(environment)

    final_u = []
    alloc_to_risk = [[] for _ in range(eval_episodes)]
    ret = []

    np.random.seed(111)

    for episode in range(eval_episodes):
        env.reset()
        while not env.done:
            np.random.rand()  # random context
            state = env.get_state()
            q_pred = np.squeeze(agent.qnn.predict(state))
            action = agent.action_space[argmax_index(q_pred)]
            trade = compute_trade(env.p, action, env.tcost)
            env.trade(trade)
            assert math.isclose(env.p[1]/np.sum(env.p), action)
            alloc_to_risk[episode].append(env.p[1]/np.sum(env.p))
            sgr = env.update()
            ret.append(sgr)
        final_u.append(env.get_utility())

    final_u = np.array(final_u)
    alloc_to_risk = np.array(alloc_to_risk)
    ret = np.array(ret)

    return final_u, alloc_to_risk, ret


def portfolio_reinforce(eval_episodes, environment, agent):

    """
    Simulates a portfolio strategy suggested by a trained REINFORCE agent in a
    given environment.

    Parameters
    ----------
    :param eval_episodes : int
           Number of episodes to simulate.
    :param environment : Env instance
           Environment in which to simulate the portfolio strategy.
    :param agent : AgentReinforce instance
           REINFORCE agent (preferably trained in the same environment as is
           simulated).

    Returns
    -------
    :returns final_u : ndarray
             Array containing utility of terminal wealth for each simulated
             episode.
    :returns alloc_to_risk : ndarray
             Array containing the share of wealth invested into the risky
             asset in each period for all simulated episodes.
    :returns ret : ndarray
             Array containing the simple gross returns realized in each
             period for all simulated episodes.
    """

    print("Simulating REINFORCE portfolio strategy.")

    env = copy.deepcopy(environment)

    final_u = []
    alloc_to_risk = [[] for _ in range(eval_episodes)]
    ret = []

    np.random.seed(111)

    for episode in range(eval_episodes):
        env.reset()
        while not env.done:

            s = env.get_state()
            a = agent.choose_action(s)
            trade = compute_trade(env.p, a, env.tcost)
            env.trade(trade)
            assert math.isclose(env.p[1] / np.sum(env.p), a)
            alloc_to_risk[episode].append(env.p[1] / np.sum(env.p))
            sgr = env.update()
            ret.append(sgr)

        final_u.append(env.get_utility())

    final_u = np.array(final_u)
    alloc_to_risk = np.array(alloc_to_risk)
    ret = np.array(ret)

    return final_u, alloc_to_risk, ret


def portfolio_ac(eval_episodes, environment, agent):

    """
    Simulates a portfolio strategy suggested by a trained actor-critic agent
    in a given environment.

    Parameters
    ----------
    :param eval_episodes : int
           Number of episodes to simulate.
    :param environment : Env instance
           Environment in which to simulate the portfolio strategy.
    :param agent : AgentAC instance
           Actor-critic agent (preferably trained in the same environment as is
           simulated).

    Returns
    -------
    :returns final_u : ndarray
             Array containing utility of terminal wealth for each simulated
             episode.
    :returns alloc_to_risk : ndarray
             Array containing the share of wealth invested into the risky
             asset in each period for all simulated episodes.
    :returns ret : ndarray
             Array containing the simple gross returns realized in each
             period for all simulated episodes.
    """

    print("Simulating actor-critic portfolio strategy.")

    env = copy.deepcopy(environment)

    final_u = []
    alloc_to_risk = [[] for _ in range(eval_episodes)]
    ret = []

    np.random.seed(111)

    for episode in range(eval_episodes):

        env.reset()

        while not env.done:

            s = env.get_state()
            a = agent.choose_action(s)
            trade = compute_trade(env.p, a, env.tcost)
            env.trade(trade)
            assert math.isclose(env.p[1] / np.sum(env.p), a)
            alloc_to_risk[episode].append(env.p[1] / np.sum(env.p))
            sgr = env.update()
            ret.append(sgr)

        final_u.append(env.get_utility())

    final_u = np.array(final_u)
    alloc_to_risk = np.array(alloc_to_risk)
    ret = np.array(ret)

    return final_u, alloc_to_risk, ret
