import numpy as np
import itertools
import pandas as pd
from keras.optimizers import Adam, SGD
from environment import Env
from agents import AgentAC
from train import train_ac
from simulate import portfolio_safe, portfolio_myopic, portfolio_risky, \
    portfolio_ac
from helpers import all_close

# suppress scientific notation for numpy and pandas printouts:
np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:5,.5f}'.format

if __name__ == '__main__':

    # HYPERPARAMETERS #

    # algorithm setup:
    train_episodes = 10000
    eval_episodes = 1000
    pi_update = 10

    # agent variables:
    dim_state = 2
    dim_actions = 11
    hidden_dims_pi = (64, 64, 64)
    hidden_dims_cr = (128, 128, 128)
    optimizer_pi = Adam()
    optimizer_cr = Adam()
    gamma = 1.

    # environment variables:
    start = "random"
    tcost = 0.0
    horizon = 1
    w = 1000.
    theta = 1.
    mu = np.array([0, 0])
    sigma = np.array([0, 0.001])

    # initiliaze the RL-agent:
    agent = AgentAC(dim_state=dim_state,
                    dim_actions=dim_actions,
                    hidden_dims_pi=hidden_dims_pi,
                    hidden_dims_cr=hidden_dims_cr,
                    optimizer_pi=optimizer_pi,
                    optimizer_cr=optimizer_cr,
                    gamma=gamma)

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

    trained_agent, train_loss_pi, train_loss_cr, train_states, \
        train_actions, train_rewards = train_ac(agent=agent,
                                                environment=env,
                                                episodes=train_episodes,
                                                policy_update=pi_update)

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

    fu_ac, alloc_ac, ret_ac = portfolio_ac(
        eval_episodes=eval_episodes,
        environment=env,
        agent=trained_agent
    )

    # EVALUATION #

    print("\n===EVALUATION===\n")

    # create start state representation:
    start_state = np.array([[0 / env.horizon] + [env.w / env.w]])

    actions = pd.DataFrame(train_actions)
    rewards = pd.DataFrame(train_rewards)

    init_pi = pd.DataFrame(agent.pi.predict(start_state),
                           index=["initial_policy"]).T
    train_pi = pd.DataFrame(trained_agent.pi.predict(start_state),
                            index=["trained_policy"]).T
    comp = pd.concat([init_pi, train_pi], axis=1)
    print("Learned vs. initial policy action probabilities and state "
          "values for the start state:")
    print(comp.to_string())

    init_v = np.squeeze(agent.cr.predict(start_state))
    train_v = np.squeeze(trained_agent.cr.predict(start_state))

    print("\ninitial start state state value:", init_v)
    print("trained start state state value:", train_v)

    # check if returns during the simulation were the same:
    print("\nReturns during simulation are identical:",
          all_close([ret_safe, ret_myopic, ret_risky, ret_ac]))

    # average allocation to risky asset for different portfolio strategies:
    data = np.hstack((alloc_safe, alloc_myopic, alloc_risky, alloc_ac))
    iterables = [["1. safe", "2. myopic", "3. risky", "4. actor-critic"],
                 list(range(env.horizon))]
    col = pd.MultiIndex.from_product(iterables)
    df_alloc = pd.DataFrame(data,
                            index=list(range(eval_episodes)),
                            columns=col)

    print("\nAverage allocation to risky asset for different strategies "
          "during simulation:")
    print(df_alloc.mean().to_string())

    # average final utility for different portfolio strategies:
    df_fu = pd.DataFrame({"1. safe": fu_safe,
                          "2. myopic": fu_myopic,
                          "3. risky": fu_risky,
                          "4. actor-critic": fu_ac})
    print("\nAverage terminal utility of wealth for different strategies "
          "during simulation:")
    print(df_fu.mean().to_string())

    # DATA EXPORT #

    pd.DataFrame(train_loss_pi).to_csv("loss_train_pi_ac.csv")
    pd.DataFrame(train_loss_cr).to_csv("loss_train_cr_ac.csv")
    actions.to_csv("actions_train_ac.csv")
    rewards.to_csv("rewards_train_ac.csv")
    comp.to_csv("pred_qnn_change_ac.csv")
    df_alloc.to_csv("alloc_sim_ac.csv")
    df_fu.to_csv("fu_sim_ac.csv")
