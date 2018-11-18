import numpy as np
import pandas as pd
import gc
from keras.optimizers import Adam
from hyperopt import tpe, Trials, hp, fmin
from environment import Env
from agents import AgentDQN
from train import train_dqn
from simulate import portfolio_safe, portfolio_myopic, portfolio_risky, \
    portfolio_dqn
from helpers import all_close, crra_utility
from param_optimization import HPS
from line_profiler import LineProfiler


# suppress scientific notation for numpy and pandas printouts:
np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:5,.8f}'.format

if __name__ == '__main__':

    # 1. SETUP #

    print("\n===SETUP===\n")

    # algorithm setup:
    train_episodes = 500000
    eval_episodes = 10000
    batch_size = 64
    init_d_size = 20000
    max_d_size = 1000000
    target_update = 2500
    freeze_after = 3000000000000
    checktime = False
    pretraining = True

    # agent variables:
    dim_state = 2
    dim_actions = 11
    hidden_dims = (128, 128, 128)
    optimizer = Adam(lr=1e-7)
    gamma = 1
    epsilon = 1
    epsilon_decay = 5e-5
    frozen = False
    pretrained = False

    # environment variables:
    start = "random"
    tcost = 0.0
    horizon = 2
    w = 1000000.
    theta = 1.0
    regimes = {1: {"mu": np.array([np.log(1.02), 0.07]),
                   "sigma": np.array([0, 0.5]),
                   "periods": np.arange(300, 300)},
               2: {"mu": np.array([0, 0]),
                   "sigma": np.array([0, 1]),
                   "periods": np.arange(0, horizon)},
               3: {"mu": np.array([0, -0.00007]),
                   "sigma": np.array([0, 0.015]),
                   "periods": np.arange(300, 300)},
               4: {"mu": np.array([np.log(1.00008), 0.0001123]),
                   "sigma": np.array([0, 0.009]),
                   "periods": np.concatenate([np.arange(300, 300),
                                              np.arange(300, 300)])}
               }

    # evaluation variables:
    eval_w_start = 0.1
    eval_w_end = 2.1
    eval_w_points = 201
    param_searches = 0
    space = {"lr": hp.uniform("lr", 1e-10, 1e-4),
             "lr decay": hp.uniform("lr decay", 0, 1e-2),
             "batch size": hp.choice("batch size", [16, 32, 64, 128,
                                                    256]),
             "target update": hp.quniform("target update", 10, 500, 10)}

    # initiliaze the RL-agent:
    agent = AgentDQN(dim_state=dim_state,
                     dim_actions=dim_actions,
                     hidden_dims=hidden_dims,
                     optimizer=optimizer,
                     gamma=gamma,
                     eps=epsilon,
                     eps_decay=epsilon_decay,
                     frozen=frozen,
                     pretrained=pretrained)

    # initiliaze the environment:
    env = Env(start=start,
              tcost=tcost,
              horizon=horizon,
              w=w,
              theta=theta,
              regimes=regimes)

    print("-- Training episodes =", train_episodes)
    print("-- Evaluation episodes =", eval_episodes)
    print("-- Batch size =", batch_size)
    print("-- Initial replay memory size =", init_d_size)
    print("-- Maximum replay memory size =", max_d_size)
    print("-- Update target network after", target_update, "episodes.")
    print("-- Dimension of state representation =", dim_state)
    print("-- Number of actions (different possible portfolio allocations) =",
          dim_actions)
    print("-- Number of hidden layers =", len(hidden_dims))
    print("-- Nodes in each hidden layer = ", str(hidden_dims)[1:-1])
    print("-- Optimizer configuration:")
    print(pd.DataFrame.from_dict(optimizer.get_config(), orient="index"))
    print("-- Discount factor for TD-target = ", gamma)
    print("-- Epsilon starting value (for exploration) =", epsilon)
    print("-- Epsilon decay factor =", epsilon_decay)
    print("-- Transaction cost factor =", tcost)
    print("-- Investor's investment horizon =", horizon)
    print("-- Investor's initial wealth =", w)
    print("-- Investor's risk aversion factor =", theta)
    print("-- Asset log-return distribution parameters in different regimes:")
    print(pd.DataFrame.from_dict(regimes))
    print("-- Evaluation state space (from, to, steps):", eval_w_start,
          eval_w_end, eval_w_points)

    reg_periods = [v["periods"] for v in regimes.values()]
    reg_lengths = [len(r) for r in reg_periods]

    if np.sum(np.array(reg_lengths) > 0) == 1:
        print("-- Independently and identically (!) distributed asset "
              "returns.")
        if horizon == 2:
            print("-- Number of hyperparameter searches:", param_searches)

    # 2. SETUP SANITY CHECK #

    print("\n===SETUP SANITY CHECK===\n")

    # assert that regime periods cover full horizon:
    assert np.array_equal(np.unique(np.concatenate(reg_periods)),
                          np.arange(horizon)) is True

    # assert that regime periods do not overlap:
    assert len(np.concatenate(reg_periods)) is len(np.unique(np.concatenate(
        reg_periods)))

    # Annualized volatility on risky asset in different regimes:
    for k, v in regimes.items():
        ann_vol = np.squeeze(v["sigma"][1] * np.sqrt(252))
        print("Annualized volatility of log-returns on risky asset "
              "in regime", k, "=", "{0:.5f}".format(ann_vol))

    # terminal wealth statistics for all risky allocation:
    lw = []
    for i in range(train_episodes + round(init_d_size/horizon)):
        env.reset()
        while not env.done:
            _, _, _, _ = env.take_action(action=1.0)
        lw.append(np.sum(env.p))

    print("\nIf fully invested into the risky asset (over ",
          train_episodes + round(init_d_size / horizon),
          "runs):")
    print("-- Mean terminal wealth =", "{0:.2f}".format(np.mean(lw)))
    print("-- Maximum terminal wealth =", "{0:.2f}".format(np.max(lw)))
    print("-- Minimum terminal wealth =", "{0:.2f}".format(np.min(lw)))
    print("-- Utility of mean terminal wealth =",
          "{0:.5f}".format(crra_utility(np.mean(lw), env.theta)))
    print("-- Mean utility of terminal wealth =",
          "{0:.5f}".format(np.mean(crra_utility(lw, env.theta))))
    print("-- For comparison: initial utility =",
          "{0:.5f}".format(crra_utility(env.w, env.theta)))
    del lw

    # compute expected utility of wealth and utility of expected wealth for
    # all actions (assumption: same action in each period):
    # goal 1: see if risk aversion has effect
    # goal 2: see if noise dominates average with current settings
    lu = [[] for _ in range(dim_actions)]
    lw = [[] for _ in range(dim_actions)]
    e_u_fw = []
    u_e_fw = []
    for i, a in enumerate(agent.action_space):
        total_episodes = train_episodes + round(init_d_size/horizon)
        share_action_in_total_episodes = (total_episodes/horizon)/dim_actions
        # simulate each action strategy x times:
        for _ in range(round(share_action_in_total_episodes)):
            env.reset()
            while not env.done:
                env.take_action(action=a)
            lw[i].append(np.sum(env.p))
            lu[i].append(env.get_utility())
        e_u_fw.append(np.mean(lu[i]))
        u_e_fw.append(crra_utility(np.mean(lw[i]), env.theta))

    euw_vs_uew = pd.DataFrame({"exp_u_w": e_u_fw,
                               "u_exp_w": u_e_fw})
    euw_vs_uew.to_csv("euw_vs_uew_dqn.csv")

    del euw_vs_uew, lu, lw, e_u_fw, u_e_fw
    gc.collect()

    pd.options.display.float_format = '{:5,.5f}'.format

    # 3. HYPERPARAMETER OPTIMIZATION #

    print("\n===HYPERPARAMETER OPTIMIZATION===\n")

    if param_searches > 0:

        print("Hyperparameter search desired.")

        if pretraining is True:

            print("Hyperparameter search is not possible.)")
            print("Reason: Not configured in conjuction with pretraining\ "
                  "the network.")

        else:

            if horizon == 2 and np.sum(np.array(reg_lengths) > 0) == 1:

                print("Hyperparameter search is possible.")
                print("Reason: We can compute the true Q-values as the "
                      "environment consists of two periods and has a "
                      "stationary return distribution. Hence, we can compute "
                      "the mean squared error between the true and estimated"
                      "Q-values given certain algorithm parameters. We "
                      "then minimize this MSE by tuning the "
                      "hyperparameters.")

                hps = HPS(dim_state, dim_actions, hidden_dims, gamma, epsilon,
                          epsilon_decay, frozen, pretrained, start, tcost,
                          horizon, w, theta, regimes, train_episodes,
                          init_d_size, max_d_size, freeze_after)

                x, y = hps.data(eval_w_start, eval_w_end, eval_w_points)

                trials = Trials()

                print("\nSearching for optimal hyperparameters.\n")
                best = fmin(fn=hps.objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=param_searches,
                            trials=trials)

                best["batch size"] = [16, 32, 64, 128, 256][best["batch "
                                                                 "size"]]
                opt_param = pd.DataFrame.from_dict(best, orient="index")
                opt_param.columns = ["value"]
                print("The optimal parameter values for different "
                      "hyperparameters are:")
                print(opt_param)

                optimizer = Adam(lr=best["lr"], decay=best["lr decay"])
                batch_size = best["batch size"]
                target_update = best["target update"]

            else:

                print("Hyperparameter search is not possible.")
                print("Reason: cannot optimize hyperparameters because we "
                      "lack the true Q-values and therefore can't properly "
                      "evaluate the Q-value estimates.")

    else:

        print("Hyperparameter search not desired.")

    # 4. TRAINING #

    print("\n===TRAINING===\n")

    if pretraining:

        print("Pre-training.")

        # re-initiliaze the RL-agent:
        agent = AgentDQN(dim_state=dim_state,
                         dim_actions=dim_actions,
                         hidden_dims=hidden_dims,
                         optimizer=Adam(),
                         gamma=gamma,
                         eps=epsilon,
                         eps_decay=epsilon_decay,
                         frozen=frozen,
                         pretrained=pretrained)

        trained_agent, train_loss_1, train_states_1, \
            train_actions_1, train_rewards_1, train_new_states_1, \
            train_pred_1 = train_dqn(agent=agent,
                                     environment=env,
                                     episodes=3000,
                                     batch_size=512,
                                     init_d_size=500000,
                                     max_d_size=500000,
                                     target_update=200,
                                     freeze_after=freeze_after)

        states_1 = pd.DataFrame(train_states_1)
        actions_1 = pd.DataFrame(train_actions_1)
        rewards_1 = pd.DataFrame(train_rewards_1)
        new_states_1 = pd.DataFrame(train_new_states_1)
        del train_states_1, train_actions_1, train_rewards_1, \
            train_new_states_1
        log_1 = pd.concat([states_1, actions_1, rewards_1, new_states_1],
                          axis=1)
        loss_1 = pd.DataFrame(train_loss_1)
        pred_1 = pd.DataFrame(train_pred_1)
        del train_loss_1, train_pred_1
        gc.collect()

        # change optimizer to real training optimizer:
        trained_agent.optimizer = optimizer

        print("Fine-grained training.")

        trained_agent, train_loss_2, train_states_2, \
            train_actions_2, train_rewards_2, train_new_states_2, \
            train_pred_2 = train_dqn(agent=trained_agent,
                                     environment=env,
                                     episodes=train_episodes,
                                     batch_size=batch_size,
                                     init_d_size=init_d_size,
                                     max_d_size=max_d_size,
                                     target_update=target_update,
                                     freeze_after=freeze_after)

        # convert lists with training data to pandas dataframes:
        states_2 = pd.DataFrame(train_states_2)
        actions_2 = pd.DataFrame(train_actions_2)
        rewards_2 = pd.DataFrame(train_rewards_2)
        new_states_2 = pd.DataFrame(train_new_states_2)
        del train_states_2, train_actions_2, train_rewards_2, \
            train_new_states_2
        log_2 = pd.concat([states_2, actions_2, rewards_2, new_states_2],
                          axis=1)
        loss_2 = pd.DataFrame(train_loss_2)
        pred_2 = pd.DataFrame(train_pred_2)  # predictions for start state
        del train_loss_2, train_pred_2
        gc.collect()

        actions = pd.concat([actions_1, actions_2], axis=0)
        rewards = pd.concat([rewards_1, rewards_2], axis=0)
        log = pd.concat([log_1, log_2], axis=0)
        loss = pd.concat([loss_1, loss_2], axis=0)
        pred = pd.concat([pred_1, pred_2], axis=0)

        del states_1, actions_1, rewards_1, new_states_1, log_1, pred_1, loss_1
        del states_2, actions_2, rewards_2, new_states_2, log_2, pred_2, loss_2
        gc.collect()

    else:

        agent = AgentDQN(dim_state=dim_state,
                         dim_actions=dim_actions,
                         hidden_dims=hidden_dims,
                         optimizer=optimizer,
                         gamma=gamma,
                         eps=epsilon,
                         eps_decay=epsilon_decay,
                         frozen=frozen,
                         pretrained=pretrained)

        trained_agent, train_loss, train_states, train_actions, \
            train_rewards, train_new_states, train_pred = train_dqn(
                agent=agent,
                environment=env,
                episodes=train_episodes,
                batch_size=batch_size,
                init_d_size=init_d_size,
                max_d_size=max_d_size,
                target_update=target_update,
                freeze_after=freeze_after)

        # convert lists with training data to pandas dataframes:
        states = pd.DataFrame(train_states)
        actions = pd.DataFrame(train_actions)
        rewards = pd.DataFrame(train_rewards)
        new_states = pd.DataFrame(train_new_states)
        del train_states, train_actions, train_rewards, \
            train_new_states
        log = pd.concat([states, actions, rewards, new_states], axis=1)
        loss = pd.DataFrame(train_loss)
        pred = pd.DataFrame(train_pred)  # predictions for start state
        del states, new_states, train_loss, train_pred
        gc.collect()

    actions.to_csv("actions_train_dqn.csv")
    rewards.to_csv("rewards_train_dqn.csv")
    log.to_csv("log_train_dqn.csv")
    loss.to_csv("loss_train_dqn.csv")
    pred.to_csv("qval_train_dqn.csv")

    trained_agent.qnn.save("q_estimator_dqn.h5")

    del actions, rewards, log, pred, loss
    gc.collect()

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

    fu_dqn, alloc_dqn, ret_dqn = portfolio_dqn(
        eval_episodes=eval_episodes,
        environment=env,
        agent=trained_agent
    )

    # check if returns during the simulation were the same:
    print("\nReturns during simulation are identical:",
          all_close([ret_safe, ret_myopic, ret_risky, ret_dqn]))

    del ret_safe, ret_myopic, ret_risky, ret_dqn
    gc.collect()

    # EVALUATION #

    print("\n===EVALUATION===\n")

    # average allocation to risky asset for different portfolio strategies:
    alloc_data = np.hstack((alloc_safe, alloc_myopic, alloc_risky, alloc_dqn))
    iterables = [["1. safe", "2. myopic", "3. risky", "4. DQN"],
                 list(range(env.horizon))]
    col = pd.MultiIndex.from_product(iterables)
    df_alloc = pd.DataFrame(alloc_data,
                            index=list(range(eval_episodes)),
                            columns=col)

    print("Average allocation to risky asset for different strategies "
          "during simulation:")
    print(df_alloc.mean().to_string())

    del alloc_safe, alloc_myopic, alloc_risky, alloc_dqn
    gc.collect()

    # average final utility for different portfolio strategies:
    df_fu = pd.DataFrame({"1. safe": fu_safe,
                          "2. myopic": fu_myopic,
                          "3. risky": fu_risky,
                          "4. DQN": fu_dqn})
    print("\nAverage terminal utility of wealth for different strategies "
          "during simulation:")
    print(df_fu.mean().to_string())

    del fu_safe, fu_myopic, fu_risky, fu_dqn
    gc.collect()

    # create start state representation:
    start_state = np.array([[0 / env.horizon] + [env.w / env.w]])

    # create prediction data for all states [x, y]:

    # combination of all possible states:
    x1 = np.arange(0, horizon) / horizon
    x2 = np.linspace(eval_w_start, eval_w_end, eval_w_points)
    test_states = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)

    # q-value estimations for all actions over states [x1, x2] - in 2 formats:
    list_state_time = []
    list_state_wealth = []
    list_actions = []
    list_qvalues = []
    pred_overview = {}

    for a in range(trained_agent.output_dims):
        pred_action = pd.DataFrame(index=x1, columns=x2)
        for s in test_states:
            x_idx = np.squeeze(np.where(s[0] == x1))
            y_idx = np.squeeze(np.where(s[1] == x2))
            s = s.reshape(-1, trained_agent.input_dims)
            z = np.squeeze(trained_agent.qnn.predict(s))[a]
            pred_action.iloc[x_idx.item(), y_idx.item()] = z
            list_state_time.append(np.squeeze(s)[0])
            list_state_wealth.append(np.squeeze(s)[1])
            list_actions.append(a)
            list_qvalues.append(z)

        pred_overview[a] = pred_action

    pred_overview_2 = pd.DataFrame({"action": list_actions,
                                    "state_time": list_state_time,
                                    "state_wealth": list_state_wealth,
                                    "q_value": list_qvalues})

    # compare initial and trained Q-value estimates for the start state:
    init_q = pd.DataFrame(agent.qnn.predict(start_state),
                          index=["initial_q"]).T
    train_q = pd.DataFrame(trained_agent.qnn.predict(start_state),
                           index=["trained_q"]).T
    comp = pd.concat([init_q, train_q], axis=1)
    print("\nLearned vs. initial Q-value estimations for the start state:")
    print(comp.to_string())

    # DATA EXPORT #

    print("\n===DATA EXPORT===\n")

    comp.to_csv("pred_qnn_change_dqn.csv")
    df_alloc.to_csv("alloc_sim_dqn.csv")
    df_fu.to_csv("fu_sim_dqn.csv")
    for a, pred_a in pred_overview.items():
        filename = "pred_action_" + str(a) + "_dqn.csv"
        pred_a.to_csv(filename)
    pred_overview_2.to_csv("pred_actions_dqn.csv")

    print("Exported data to current working directory.")

    # PERFORMANCE CHECK #

    if checktime:
        print("\n===PERFORMANCE CHECK===\n")
        lp = LineProfiler()
        lp_wrapper = lp(train_dqn)
        lp_wrapper(agent=agent,
                   environment=env,
                   episodes=train_episodes,
                   batch_size=batch_size,
                   init_d_size=init_d_size,
                   max_d_size=max_d_size,
                   target_update=target_update)
        lp.print_stats()
