import numpy as np
import copy
from keras.models import Model, Input
from keras.layers import Dense, Activation, Dropout, BatchNormalization
import keras.backend as k
from helpers import eps_greedy, compute_rl_return, argmax_index


class AgentQtab:

    """
    Class for tabular Q-learning agent for a single period portfolio
    allocation problem.
    """

    def __init__(self, dim_state, dim_actions, lr, lr_decay, gamma, eps,
                 eps_decay):

        """
        Initializes tabular Q-learning agent.

        Parameters
        ----------
        :param dim_state : int
               Dimension of state representation.
        :param dim_actions : int
               Number of available portfolio allocation choices.
        :param lr : float
               Learning rate for Q-learning value update.
        :param lr_decay : float
               Learning rate decay factor. Actual learning rate is
               alpha = lr / (1 + lr_decay * iteration)
        :param gamma : float
               Discount factor for Q-target.
        :param eps : float
               Epsilon value driving exploration.
        :param eps_decay : float
               Epsilon value decay factor. Actual epsilon value is
               epsilon = eps / (1 + eps_decay * iteration)

        Returns
        -------
        None.
        """

        self.input_dims = dim_state
        self.output_dims = dim_actions
        self.lr = lr
        self.lr_decay = lr_decay
        self.eps = eps
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.iter = 0

        self.action_space = np.linspace(0, 1, self.output_dims)

        self.__build_q_tab()

    def __build_q_tab(self):

        """
        Initializes Q-value table randomly.

        Parameters
        ----------
        None.

        Returns
        -------
        None
        """

        self.q_tab = np.random.rand(self.output_dims)

    def update(self, action, reward):

        """
        Updates the Q-value table.

        Parameters
        ----------
        :param action : float
               Action taken last time step, i.e. portfolio allocation made.
        :param reward : float
               Reward received for this time step as a result.

        Returns
        -------
        :returns pred : ndarray
                 Q-values from Q-table before the update.
        """

        pred = copy.deepcopy(self.q_tab)

        act_id = np.where(action == self.action_space)

        self.q_tab[act_id] += self.lr / (1 + self.lr_decay * self.iter) * (
                    reward - self.q_tab[act_id])

        return pred

    def choose_action(self):

        """
        Chooses an action from the action space epsilon-greedily according
        to Q-value estimations.

        Parameters
        ----------
        None.

        Returns
        -------
        :returns a : float
                 Action to be taken, i.e. portfolio allocation to be made.
        """

        a = eps_greedy(self.eps / (1 + self.eps_decay * self.iter),
                       self.action_space, self.q_tab)

        return a


class AgentDQN:

    """
    Class for DQN agent for a multi-period portfolio allocation problem.
    """

    def __init__(self, dim_state, dim_actions, hidden_dims, optimizer,
                 gamma, eps, eps_decay, frozen, pretrained):

        """
        Initializes DQN agent.

        Parameters
        ----------
        :param dim_state : int
               Dimension of state representation.
        :param dim_actions : int
               Number of available portfolio allocation choices.
        :param hidden_dims : tuple
               Specifies the network architecture. Length of tuple is the
               number of hidden layers; value of each tuple element is the
               number of nodes in the respective hidden layer.
        :param optimizer : keras.optimizer object
               Optimizer object minimizing the MSE between neural network
               Q-value predictions and Q-targets.
        :param gamma : float
               Discount factor for Q-target.
        :param eps : float
               Epsilon value driving exploration.
        :param eps_decay : float
               Epsilon value decay factor. Actual epsilon value is
               epsilon = eps / (1 + eps_decay * iteration)

        Returns
        -------
        None.
        """

        self.input_dims = dim_state
        self.output_dims = dim_actions
        self.hidden_dims = hidden_dims
        self.optimizer = optimizer
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.iter = 0
        self.frozen = frozen
        self.pretrained = pretrained

        self.action_space = np.linspace(0, 1, dim_actions)

        self.__build_dqn()

    def __build_dqn(self):

        """
        Builds neural network to estimate the Q-values.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        inputs = Input(shape=(self.input_dims,))
        net = inputs
        for h_dim in self.hidden_dims:
            net = Dense(h_dim, kernel_initializer='he_uniform')(net)
            net = Activation("elu")(net)

        outputs = Dense(self.output_dims)(net)
        outputs = Activation("linear")(outputs)
        self.qnn = Model(inputs=inputs, outputs=outputs)
        self.tnn = Model(inputs=inputs, outputs=outputs)

        if self.frozen:
            for layer in self.qnn.layers[:-2]:
                layer.trainable = False

        self.qnn.compile(optimizer=self.optimizer, loss="mse")
        self.tnn.compile(optimizer=self.optimizer, loss="mse")

        if self.pretrained:
            self.qnn.load_weights("q_estimator_dqn.h5")
            self.tnn.load_weights("q_estimator_dqn.h5")

    def update(self, states, actions, rewards, new_states, done):

        """
        Updates the neural network estimating the Q-values.

        Parameters
        ----------
        :param states : ndarray
               Batch of experienced states.
        :param actions : ndarray
               Batch of taken actions.
        :param rewards : ndarray
               Batch of rewards received from action in state.
        :param new_states : ndarray
               Batch of new states after action in state.
        :param done : ndarray
               Batch of booleans indicating whether episode was finished,
               i.e. whether new_state is a terminal state.

        Returns
        -------
        :returns q_s : ndarray
                 Q-value predictions made for the input states before the
                 update
        :returns loss : float
                 Value of the loss function on the current batch of inputs.
        """

        batchsize = actions.shape[0]
        states = states.reshape(-1, self.input_dims)
        new_states = new_states.reshape(-1, self.input_dims)

        act_id = [np.where(a == self.action_space) for a in actions]
        act_id = np.array(act_id).reshape(-1,)

        # set targets to predictions, such that error = 0:
        q_s = np.squeeze(self.qnn.predict(states))
        targets = q_s.copy()

        # assert that prediction error = 0:
        if not np.allclose(q_s, targets):
            print(q_s)
            print(targets)
            raise AssertionError()

        # choose new_actions for new_states:
        q_ss = np.squeeze(self.qnn.predict(new_states))

        if batchsize == 1:
            act_new_id = argmax_index(q_ss)
        else:
            act_new_id = np.array(list(map(argmax_index, q_ss)))

        # calculate q-values for new_states with target network:
        qt_ss = np.squeeze(self.tnn.predict(new_states))

        # select target q-values for new_actions:
        if batchsize == 1:
            qt_ss_a = qt_ss[act_new_id]
        else:
            qt_ss_a = qt_ss[np.arange(batchsize), act_new_id]

        # update targets:
        if batchsize == 1:
            targets[act_id] = rewards + np.invert(done) * self.gamma * qt_ss_a
            targets = np.array([targets])
        else:
            for i in np.arange(batchsize):
                targets[i, act_id[i]] = rewards[i] + np.invert(done[i]) * \
                                    self.gamma * qt_ss_a[i]

        # compute loss on current batch:
        loss = self.qnn.evaluate(states, targets, verbose=False)

        # train q-network on targets:
        self.qnn.fit(states, targets, verbose=False)

        return q_s, loss

    def choose_action(self, state):

        """
        Chooses an action from the action space epsilon-greedily according
        to Q-value estimations.

        Parameters
        ----------
        :param state : ndarray
               State the agent is currently in.

        Returns
        -------
        :returns a : float
                 Action to be taken, i.e. portfolio allocation to be made.
        """

        q_s = np.squeeze(self.qnn.predict(state))
        a = eps_greedy(self.eps / (1 + self.eps_decay * self.iter),
                       self.action_space, q_s)

        return a

    def freeze(self):

        """
        Freezes weights in all layers of the Q-value estimating neural
        network but the last one.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        for layer in self.qnn.layers[:-2]:
            layer.trainable = False

        self.qnn.compile(optimizer=self.optimizer, loss="mse")

        self.frozen = True

    def copy(self):

        """
        Creates a copy of the agent and its current configuration.

        Parameters
        ----------
        None.

        Returns
        -------
        :returns new_agent : AgentDQN instance
                 Copy of the current agent.
        """

        new_agent = AgentDQN(self.input_dims, self.output_dims,
                             self.hidden_dims, self.optimizer, self.gamma,
                             self.eps, self.eps_decay, self.frozen,
                             self.pretrained)
        new_agent.qnn.set_weights(self.qnn.get_weights())
        new_agent.tnn.set_weights(self.tnn.get_weights())

        return new_agent


class AgentReinforce:

    """
    Class for REINFORCE agent for a multi-period portfolio allocation problem.
    """

    def __init__(self, dim_state, dim_actions, hidden_dims, optimizer, gamma):

        """
        Initializes REINFORCE agent.

        Parameters
        ----------
        :param dim_state : int
               Dimension of state representation.
        :param dim_actions : int
               Number of available portfolio allocation choices.
        :param hidden_dims : tuple
               Specifies the network architecture. Length of tuple is the
               number of hidden layers; value of each tuple element is the
               number of nodes in the respective hidden layer.
        :param optimizer : keras.optimizer object
               Optimizer object minimizing the MSE between neural network
               Q-value predictions and Q-targets.
        :param gamma : float
               Discount factor for Q-target.

        Returns
        -------
        None.
        """

        self.input_dims = dim_state
        self.output_dims = dim_actions
        self.hidden_dims = hidden_dims
        self.optimizer = optimizer
        self.gamma = gamma

        self.action_space = np.linspace(0, 1, dim_actions)

        self.__build_policy_network()
        self.__build_train_fn()

    def __build_policy_network(self):

        """
        Hidden method building the policy network which maps from state to
        action probabilities.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        inputs = Input(shape=(self.input_dims,))
        net = inputs

        for h_dim in self.hidden_dims:
            net = Dense(h_dim)(net)
            net = Activation("relu")(net)

        outputs = Dense(self.output_dims)(net)
        outputs = Activation("softmax")(outputs)
        self.pi = Model(inputs=inputs, outputs=outputs)

    def __build_train_fn(self):

        """
        Creates the function to train the policy network.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        # placeholder for the action probabilities:
        a_prob_place = self.pi.output

        # placeholder for the onehot encoded action vector
        a_onehot_place = k.placeholder(shape=(None, self.output_dims),
                                       name="action_onehot")

        # placeholder for the discounter rewards at each timestep:
        discount_r_place = k.placeholder(shape=(None,),
                                         name="discount_reward")

        # compute the action probability of the taken action:
        action_prob = k.sum(a_prob_place * a_onehot_place, axis=1)

        # compute the log action probability:
        log_a_prob = k.log(action_prob)

        # multiply the log action probability with the discounted rewards:
        loss = log_a_prob * discount_r_place
        loss = k.sum(loss)

        solver = self.optimizer

        updates = solver.get_updates(params=self.pi.trainable_weights,
                                     loss=loss)

        self.train_fn = k.function(inputs=[self.pi.input,
                                           a_onehot_place,
                                           discount_r_place],
                                   outputs=[loss],
                                   updates=updates)

    def choose_action(self, state):

        """
        Chooses an action from the action space according
        to current policy.

        Parameters
        ----------
        :param state : ndarray
               State the agent is currently in.

        Returns
        -------
        :returns a : float
                 Action to be taken, i.e. portfolio allocation to be made.
        """

        a_probs = np.squeeze(self.pi.predict(state))
        a = np.random.choice(self.action_space, p=a_probs)

        return a

    def update(self, states, actions, rewards):

        """
        Updates the policy neural network.

        Parameters
        ----------
        :param states : ndarray
               Batch of experienced states.
        :param actions : ndarray
               Batch of taken actions.
        :param rewards : ndarray
               Batch of rewards received from action in state.

        Returns
        -------
        :returns loss : float
                 Value of the loss function on the current batch of inputs.
        """

        # encode actions taken during episodes in one-hot vectors:
        a_onehot = np.array([a == self.action_space for a in actions])

        # compute the rl-returns for each state-action pair during the episode:
        rl_return = compute_rl_return(rewards, self.gamma)

        assert states.shape[1] == self.input_dims
        assert a_onehot.shape[0] == states.shape[0]
        assert a_onehot.shape[1] == self.output_dims
        assert len(rl_return.shape) == 1

        loss = self.train_fn([states, a_onehot, rl_return])

        return loss

    def copy(self):

        """
        Creates a copy of the agent and its current configuration.

        Parameters
        ----------
        None.

        Returns
        -------
        :returns new_agent : AgentREINFORCE instance
                 Copy of the current agent.
        """

        new_agent = AgentReinforce(self.input_dims, self.output_dims,
                                   self.hidden_dims, self.optimizer,
                                   self.gamma)
        new_agent.pi.set_weights(self.pi.get_weights())

        return new_agent


class AgentAC:

    """
    Class for actor-critic agent for a multi-period portfolio allocation
    problem.
    """

    def __init__(self, dim_state, dim_actions, hidden_dims_pi,
                 hidden_dims_cr, optimizer_pi, optimizer_cr, gamma):

        self.input_dims = dim_state
        self.output_dims_pi = dim_actions
        self.output_dims_cr = 1
        self.hidden_dims_pi = hidden_dims_pi
        self.hidden_dims_cr = hidden_dims_cr
        self.optimizer_pi = optimizer_pi
        self.optimizer_cr = optimizer_cr
        self.gamma = gamma

        self.action_space = np.linspace(0, 1, dim_actions)

        self.__build_policy_network()
        self.__build_train_fn()
        self.__build_critic_network()

    def __build_policy_network(self):

        """
        Hidden method building the policy network which maps from state to
        action probabilities.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        inputs = Input(shape=(self.input_dims,))
        net = inputs

        for h_dim in self.hidden_dims_pi:
            net = Dense(h_dim)(net)
            net = Activation("relu")(net)

        outputs = Dense(self.output_dims_pi)(net)
        outputs = Activation("softmax")(outputs)
        self.pi = Model(inputs=inputs, outputs=outputs)

    def __build_critic_network(self):

        """
        Hidden method building the policy network which maps from state to
        action advantages.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        inputs = Input(shape=(self.input_dims,))
        net = inputs

        for h_dim in self.hidden_dims_cr:
            net = Dense(h_dim, kernel_initializer="he_uniform")(net)
            net = Activation("elu")(net)

        outputs = Dense(self.output_dims_cr)(net)
        self.cr = Model(inputs=inputs, outputs=outputs)
        self.cr.compile(optimizer=self.optimizer_cr, loss="mse")

    def __build_train_fn(self):

        """
        Creates the function to train the policy network.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        # placeholder for the action probabilities:
        a_prob_place = self.pi.output

        # placeholder for the onehot encoded action vector
        a_onehot_place = k.placeholder(shape=(None, self.output_dims_pi),
                                       name="action_onehot")

        # placeholder for the discounter rewards at each timestep:
        discount_r_place = k.placeholder(shape=(None,),
                                         name="discount_reward")

        # compute the action probability of the taken action:
        action_prob = k.sum(a_prob_place * a_onehot_place, axis=1) + 1e-10

        # compute the log action probability:
        log_a_prob = k.log(action_prob)

        # multiply the log action probability with the discounted rewards:
        loss = log_a_prob * discount_r_place
        loss = k.sum(loss)

        solver = self.optimizer_pi

        updates = solver.get_updates(params=self.pi.trainable_weights,
                                     loss=loss)

        self.train_fn = k.function(inputs=[self.pi.input,
                                           a_onehot_place,
                                           discount_r_place],
                                   outputs=[loss],
                                   updates=updates)

    def choose_action(self, state):

        """
        Chooses an action from the action space according
        to current policy.

        Parameters
        ----------
        :param state : ndarray
               State the agent is currently in.

        Returns
        -------
        :returns a : float
                 Action to be taken, i.e. portfolio allocation to be made.
        """

        a_probs = np.squeeze(self.pi.predict(state))
        a = np.random.choice(self.action_space, p=a_probs)

        return a

    def update_pi(self, states, actions, rewards):

        """
        Updates the policy neural network.

        Parameters
        ----------
        :param states : ndarray
               Batch of experienced states.
        :param actions : ndarray
               Batch of taken actions.
        :param rewards : ndarray
               Batch of rewards received from action in state.

        Returns
        -------
        :returns loss : float
                 Value of the loss function on the current batch of inputs.
        """

        # encode actions taken during episodes in one-hot vectors:
        a_onehot = np.array([a == self.action_space for a in actions])

        # compute the rl-returns for each state-action pair during the episode:
        rl_return = compute_rl_return(rewards, self.gamma)

        assert states.shape[1] == self.input_dims
        assert a_onehot.shape[0] == states.shape[0]
        assert a_onehot.shape[1] == self.output_dims_pi
        assert len(rl_return.shape) == 1

        loss = self.train_fn([states, a_onehot, rl_return])

        return loss

    def get_advantages(self, states, rewards, new_states, done):

        """
        Computes the advantages for the given s,a,r,s,d experiences.

        Parameters
        ----------
        :param states : ndarray
               Batch of encountered states.
        :param rewards : ndarray
               Batch of received rewards.
        :param new_states : ndarray
               Batch of successive states.
        :param done : ndarray
               Batch indicating whether corresponding successive state is
               teminal.

        Returns
        -------
        :returns adv : ndarray
                 Advantages for the inputs.
        """

        s = states
        r = rewards
        ss = new_states

        adv = r + np.invert(done) * self.gamma * self.cr.predict(ss) - \
            self.cr.predict(s)
        adv = np.squeeze(adv)

        return adv

    def get_targets(self, rewards, new_states, done):

        """
        Computes the targets for the given inputs.

        Parameters
        ----------
        :param rewards : ndarray
               Batch of received rewards.
        :param new_states : ndarray
               Batch of successive states.
        :param done : ndarray
               Batch indicating whether corresponding successive state is
               teminal.

        Returns
        -------
        :returns t : ndarray
                 Targets for critic network.
        """

        r = rewards
        ss = new_states

        t = r + np.invert(done) * self.gamma * self.cr.predict(ss)
        t = np.squeeze(t)

        return t

    def update_cr(self, states, targets):

        """
        Updates the critic network.

        Parameters
        ----------
        :param states : ndarray
               Batch of encountered states.
        :param targets : ndarray
               Targets for critic network.

        Returns
        -------
        :returns loss : ndarray
                 Loss on the current batch of states.
        """

        loss = self.cr.evaluate(states, targets, verbose=False)
        self.cr.fit(states, targets, verbose=False)

        return loss

    def copy(self):

        """
        Creates a copy of the agent and its current configuration.

        Parameters
        ----------
        None.

        Returns
        -------
        :returns new_agent : AgentAC instance
                 Copy of the current agent.
        """

        new_agent = AgentAC(self.input_dims, self.output_dims_pi,
                            self.hidden_dims_pi, self.hidden_dims_cr,
                            self.optimizer_pi, self.optimizer_cr, self.gamma)
        new_agent.pi.set_weights(self.pi.get_weights())
        new_agent.cr.set_weights(self.cr.get_weights())

        return new_agent
