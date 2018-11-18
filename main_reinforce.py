import numpy as np
import itertools
import pandas as pd
from keras.optimizers import Adam, SGD
from environment import Env
from agents import AgentReinforce
from train import train_reinforce
from simulate import portfolio_safe, portfolio_myopic, portfolio_risky, \
    portfolio_reinforce
from helpers import all_close

# suppress scientific notation for numpy and pandas printouts:
np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:5,.5f}'.format

if __name__ == '__main__':

    # HYPERPARAMETERS #

    # algorithm setup:
    train_episodes = 100000
    eval_episodes = 10000
    pi_update = 1000

    # agent variables:
    dim_state = 2
    dim_actions = 11
    hidden_dims = (64, 64, 64)
    optimizer = Adam()
    gamma = 1.

    # environment variables:
    start = "random"
    tcost = 0.0
    horizon = 1
    w = 1000.
    theta = 1.
    mu = np.array([0, 0])
    sigma = np.array([0, 1])

    # initiliaze the RL-agent:
    agent = AgentReinforce(dim_state=dim_state,
                           dim_actions=dim_actions,
                           hidden_dims=hidden_dims,
                           optimizer=optimizer,
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

    trained_agent, train_loss, train_states, \
        train_actions, train_rewards = train_reinforce(agent=agent,
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

    fu_reinforce, alloc_reinforce, ret_reinforce = portfolio_reinforce(
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
    print("Learned vs. initial policy action probabilities for the start "
          "state:")
    print(comp.to_string())

    # check if returns during the simulation were the same:
    print("\nReturns during simulation are identical:",
          all_close([ret_safe, ret_myopic, ret_risky, ret_reinforce]))

    # average allocation to risky asset for different portfolio strategies:
    data = np.hstack((alloc_safe, alloc_myopic, alloc_risky, alloc_reinforce))
    iterables = [["1. safe", "2. myopic", "3. risky", "4. REINFORCE"],
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
                          "4. REINFORCE": fu_reinforce})
    print("\nAverage terminal utility of wealth for different strategies "
          "during simulation:")
    print(df_fu.mean().to_string())

    # DATA EXPORT #

    pd.DataFrame(train_loss).to_csv("loss_train_reinforce.csv")
    actions.to_csv("actions_train_reinforce.csv")
    rewards.to_csv("rewards_train_reinforce.csv")
    comp.to_csv("pred_qnn_change_reinforce.csv")
    df_alloc.to_csv("alloc_sim_reinforce.csv")
    df_fu.to_csv("fu_sim_reinforce.csv")
