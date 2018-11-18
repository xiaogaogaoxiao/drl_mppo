import random
import numpy as np
import copy


def train_qtab(agent, environment, episodes):

    """
    Trains a tabular Q-learning agent.

    Parameters
    ----------
    :param agent : AgentQtab instance
           Tabular Q-learning agent to be trained.
    :param environment : Env instance
           Environment instance specifying and simulating the dynamics of
           the environment the agent is in.
    :param episodes : int
           Number of episodes to train the agent.

    Returns
    -------
    :returns agent : AgentQtab instance
             Trained agent.
    :returns states : list of lists
             States experienced in each episode.
    :returns actions : list of lists
             Actions taken in each episode.
    :returns rewards : list of lists
             Rewards received in each episode.
    :returns new_states : list of lists
             New states experienced in each episode.
    :returns pred : list of ndarrays
             Q-value predictions made during the training.
    """

    print("Training tabular Q-learning agent.")

    agent = copy.deepcopy(agent)
    env = copy.deepcopy(environment)

    # create lists to save variables:
    states = [[] for _ in range(episodes)]
    actions = [[] for _ in range(episodes)]
    rewards = [[] for _ in range(episodes)]
    new_states = [[] for _ in range(episodes)]
    pred = []

    for episode in range(episodes):

        # reset environment to start state:
        env.reset()

        # increase iteration count in agent:
        agent.iter += 1

        # train agent:
        while not env.done:

            s = env.get_state()
            a = agent.choose_action()
            r, ss, done, _ = env.take_action(a)

            states[episode].append(s)
            actions[episode].append(a)
            rewards[episode].append(r)
            new_states[episode].append(ss)

            q_tab = agent.update(a, r)
            pred.append(q_tab)

    return agent, states, actions, rewards, new_states, pred


def train_dqn(agent, environment, episodes, batch_size, init_d_size,
              max_d_size, target_update, freeze_after):

    """
    Trains a DQN agent.

    Parameters
    ----------
    :param agent : AgentDQN instance
           DQN agent to be trained.
    :param environment : Env instance
           Environment instance specifying and simulating the dynamics of
           the environment the agent is in.
    :param episodes : int
           Number of episodes to train the agent.
    :param batch_size : int
           Number of (s, a, r, s', done) tuples to draw from replay memory
           and train the Q-estimator network on.
    :param init_d_size : int
           Initial size of the replay memory.
    :param max_d_size : int
           Maximum size of the replay memory.
    :param target_update : int
           Number of episodes after which to update the target estimator
           network.
    :param freeze_after : int
           Numeber of episodes to train the shared weights in the neural
           network.

    Returns
    -------
    :returns agent : AgentDQN instance
             The trained agent with the updated Q-estimator network.
    :returns losses : list
             Value of the loss function at each update of the Q-estimator
             network during training.
    :returns states : list of lists
             States encountered in each episode during training.
    :returns actions : list of lists
             Actions taken in each episode during training.
    :returns rewards : list of lists
             Rewards received in each episode during training.
    :returns new_states : list of lists
             New states agent encountered after each action in each episode
             during training.
    :returns pred : list
             Q-value estimations for the start state at the beginning of each
             episode.
    """

    print("Training DQN agent.")

    # initialize start state agent and environment:
    agent = agent.copy()
    env = copy.deepcopy(environment)

    # populate replay memory:
    d = []

    print("-- Populating replay memory.")

    while len(d) < init_d_size:

        # reset environment to start state:
        env.reset()
        # randomize initial wealth:
        env.p *= np.random.uniform(0.1, 2)

        while not env.done:
            s = env.get_state()
            a = random.choice(agent.action_space)
            r, ss, done, _ = env.take_action(a)

            d.append((s, a, r, ss, done))

    print("-- Finished populating replay memory.")

    # create lists to save variables:
    states = [[] for _ in range(episodes)]
    actions = [[] for _ in range(episodes)]
    rewards = [[] for _ in range(episodes)]
    new_states = [[] for _ in range(episodes)]
    losses = []
    pred = []

    for episode in range(episodes):

        if (episode + 1) % 1000 == 0:
            print("-- Training episode ", episode + 1, ".", sep="")

        # reset environment to start state:
        env.reset()
        start_state = env.get_state()

        # randomize initial wealth:
        env.p *= np.random.uniform(0.1, 2)

        if episode % target_update == 0:
            agent.tnn.set_weights(agent.qnn.get_weights())

        agent.iter += 1

        if agent.iter == freeze_after + 1:
            agent.freeze()

        # train agent:
        while not env.done:

            s = env.get_state()
            a = agent.choose_action(s)
            r, ss, done, _ = env.take_action(a)

            if len(d) < max_d_size:
                d.append((s, a, r, ss, done))
            else:
                del d[0]
                d.append((s, a, r, ss, done))

            states[episode].append(s)
            actions[episode].append(a)
            rewards[episode].append(r)
            new_states[episode].append(ss)

            sample = random.sample(d, batch_size)
            sb, ab, rb, ssb, doneb = map(np.array, zip(*sample))

            _, loss = agent.update(sb, ab, rb, ssb, doneb)
            losses.append(loss)

        q_start = np.squeeze(agent.qnn.predict(start_state))
        pred.append(q_start)

    return agent, losses, states, actions, rewards, new_states, pred


def train_reinforce(agent, environment, episodes, policy_update):

    """
    Trains a REINFORCE agent.

    Parameters
    ----------
    :param agent : AgentREINFORCE instance
           REINFORCE agent to be trained.
    :param environment : Env instance
           Environment instance specifying and simulating the dynamics of
           the environment the agent is in.
    :param episodes : int
           Number of episodes to train the agent.
    :param policy_update : int
           Number of episodes after which to update the policy network.

    Returns
    -------
    :returns agent : AgentREINFORCE instance
             The trained agent with the updated policy network.
    :returns losses : list
             Value of the loss function at each update of the policy
             network during training.
    :returns states : list of lists
             States encountered in each episode during training.
    :returns actions : list of lists
             Actions taken in each episode during training.
    :returns rewards : list of lists
             Rewards received in each episode during training.
    """

    print("Training REINFORCE agent.")

    agent = agent.copy()
    env = copy.deepcopy(environment)

    # create lists to save variables:
    states = [[] for _ in range(episodes)]
    actions = [[] for _ in range(episodes)]
    rewards = [[] for _ in range(episodes)]
    losses = []

    last_update_episode = 0

    for episode in range(episodes):

        env.reset()

        while not env.done:

            s = env.get_state()
            a = agent.choose_action(s)
            r, ss, done, _ = env.take_action(a)

            states[episode].append(s)
            actions[episode].append(a)
            rewards[episode].append(r)

        if (episode + 1) % policy_update == 0:

            start = last_update_episode
            end = episode + 1

            su = np.array(states[start:end]).reshape(-1, agent.input_dims)
            au = np.array(actions[start:end]).reshape(-1, 1)
            ru = np.array(rewards[start:end]).reshape(-1, 1)

            loss = agent.update(su, au, ru)
            losses.append(loss)

            last_update_episode = episode + 1

    return agent, losses, states, actions, rewards


def train_ac(agent, environment, episodes, policy_update):

    """
    Trains an actor-critic agent.

    Parameters
    ----------
    :param agent : AgentAC instance
           Actor-critic agent to be trained.
    :param environment : Env instance
           Environment instance specifying and simulating the dynamics of
           the environment the agent is in.
    :param episodes : int
           Number of episodes to train the agent.
    :param policy_update : int
           Number of episodes after which to update the policy network.

    Returns
    -------
    :returns agent : AgentAC instance
             The trained agent with the updated policy and critic network.
    :returns losses_pi : list
             Value of the loss function at each update of the policy
             network during training.
    :returns losses_cr : list
             Value of the loss function at each update of the critic
             network during training.
    :returns states : list of lists
             States encountered in each episode during training.
    :returns actions : list of lists
             Actions taken in each episode during training.
    :returns rewards : list of lists
             Rewards received in each episode during training.
    """

    print("Training actor-critic agent.")

    agent = agent.copy()
    env = copy.deepcopy(environment)

    # create lists to save variables:
    states = [[] for _ in range(episodes)]
    actions = [[] for _ in range(episodes)]
    rewards = [[] for _ in range(episodes)]
    targets = [[] for _ in range(episodes)]
    advantages = [[] for _ in range(episodes)]
    losses_pi = []
    losses_cr = []

    last_update_episode = 0

    for episode in range(episodes):

        env.reset()

        while not env.done:

            s = env.get_state()
            a = agent.choose_action(s)
            r, ss, done, _ = env.take_action(a)

            t = agent.get_targets(r, ss, done)
            adv = agent.get_advantages(s, r, ss, done)

            states[episode].append(s)
            actions[episode].append(a)
            rewards[episode].append(r)
            targets[episode].append(t)
            advantages[episode].append(adv)

        su = np.array(states[episode]).reshape(-1, agent.input_dims)
        tu = np.array(targets[episode]).reshape(-1, 1)
        loss = agent.update_cr(su, tu)
        losses_cr.append(loss)

        if (episode + 1) % policy_update == 0:

            start = last_update_episode
            end = episode + 1

            su = np.array(states[start:end]).reshape(-1, agent.input_dims)
            au = np.array(actions[start:end]).reshape(-1, 1)
            ru = np.array(rewards[start:end]).reshape(-1, 1)

            loss = agent.update_pi(su, au, ru, [], [])
            losses_pi.append(loss)

            last_update_episode = episode + 1

    return agent, losses_pi, losses_cr, states, actions, rewards
