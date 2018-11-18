import numpy as np
import itertools
import pandas as pd
from environment import Env
from agents import AgentQtab
from train import train_qtab
from simulate import portfolio_safe, portfolio_myopic, portfolio_risky, \
    portfolio_qtab
from helpers import all_close

# suppress scientific notation for numpy and pandas printouts:
np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:5,.5f}'.format

if __name__ == '__main__':

    # HYPERPARAMETERS #

    # algorithm setup:
    train_episodes = 100000
    eval_episodes = 10000

    # agent variables:
    dim_state = 2
    dim_actions = 11
    gamma = 1.
    epsilon = 1.
    epsilon_decay = 1e-6
    batch_size = 2
    lr = 0.001
    lr_decay = 1e-5

    # environment variables:
    start = "random"
    tcost = 0.0
    horizon = 1
    w = 1000.
    theta = 1.
    mu = np.array([0, 0])
    sigma = np.array([0, 1])

    # initiliaze the RL-agent:
    agent = AgentQtab(dim_state=dim_state,
                      dim_actions=dim_actions,
                      lr=lr,
                      lr_decay=lr_decay,
                      gamma=gamma,
                      eps=epsilon,
                      eps_decay=epsilon_decay)

    # initiliaze the environment:
    env = Env(start=start,
              tcost=tcost,
              horizon=horizon,
              w=w,
              theta=theta,
              mu=mu,
              sigma=sigma)

    # TRAINING #

    print("\n===TRAINING===\n")

    trained_agent, train_states, train_actions, train_rewards,\
        train_new_states, train_pred = train_qtab(agent, env, train_episodes)

    # SIMULATION #

    print("\n===SIMULATION===\n")

    fu_safe, alloc_safe, ret_safe = portfolio_safe(
        eval_episodes=eval_episodes,
        environment=env
    )

    fu_myopic, alloc_myopic, ret_myopic = portfolio_myopic(
        eval_episodes=eval_episodes,
        environment=env
    )

    fu_risky, alloc_risky, ret_risky = portfolio_risky(
        eval_episodes=eval_episodes,
        environment=env
    )

    fu_qtab, alloc_qtab, ret_qtab = portfolio_qtab(
        eval_episodes=eval_episodes,
        environment=env,
        agent=trained_agent
    )

    # EVALUATION #

    print("\n===EVALUATION===\n")

    actions = pd.DataFrame(train_actions)
    rewards = pd.DataFrame(train_rewards)
    pred = pd.DataFrame(train_pred)

    list_train_actions = list(itertools.chain.from_iterable(train_actions))
    list_train_rewards = list(itertools.chain.from_iterable(train_rewards))

    # map rewards to each particular action:
    train_action_rewards = {}
    for a, r in zip(list_train_actions, list_train_rewards):
        train_action_rewards.setdefault(a, []).append(r)

    # number of rewards for each action = number of times action was taken:
    train_count_actions = {}
    for k, v in train_action_rewards.items():
        train_count_actions[k] = len(v)

    # calculate average reward for each action:
    train_action_rewards_mean = {}
    for k, v in train_action_rewards.items():
        train_action_rewards_mean[k] = np.mean(v)

    data_tar = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in
                                  train_action_rewards.items()]))

    # compare Q-table action values with average reward for each action:
    comp = pd.DataFrame(train_count_actions, index=["count"]).T
    comp["mean_rew"] = pd.Series(train_action_rewards_mean, index=comp.index)
    comp["trained_q"] = pd.Series(trained_agent.q_tab, index=comp.index)
    comp["initial_q"] = pd.Series(agent.q_tab, index=comp.index)
    print("Learned Q-values vs. average rewards for each action:")
    print(comp.to_string())

    # check if returns during the simulation were the same:
    print("\nReturns during simulation are identical:",
          all_close([ret_safe, ret_myopic, ret_risky, ret_qtab]))

    # average allocation to risky asset for different portfolio strategies:
    df_alloc = pd.DataFrame({"1. safe": np.squeeze(alloc_safe),
                             "2. myopic": np.squeeze(alloc_myopic),
                             "3. risky": np.squeeze(alloc_risky),
                             "4. q-learning": np.squeeze(alloc_qtab)})
    print("\nAverage allocation to risky asset for different strategies "
          "during simulation:")
    print(df_alloc.mean().to_string())

    # average final utility for different portfolio strategies:
    df_fu = pd.DataFrame({"1. safe": fu_safe,
                          "2. myopic": fu_myopic,
                          "3. risky": fu_risky,
                          "4. q-learning": fu_qtab})
    print("\nAverage terminal utility of wealth for different strategies "
          "during simulation:")
    print(df_fu.mean().to_string())

    # DATA EXPORT #

    actions.to_csv("actions_train_qtab.csv")
    rewards.to_csv("rewards_train_qtab.csv")
    pred.to_csv("qval_train_qtab.csv")
    data_tar.to_csv("data_tar_qtab.csv")
    comp.to_csv("pred_change_qtab.csv")
    df_alloc.to_csv("alloc_sim_qtab.csv")
    df_fu.to_csv("fu_sim_qtab.csv")
