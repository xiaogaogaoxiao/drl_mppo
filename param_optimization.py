import numpy as np
import pandas as pd
from environment import Env
from agents import AgentDQN
from keras.optimizers import Adam
from train import train_dqn
from helpers import compute_opt_weight
from hyperopt import STATUS_OK


# suppress scientific notation for numpy and pandas printouts:
np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:5,.5f}'.format


class HPS:

    """
    Initializes hyperparameter search instance.
    """

    def __init__(self, dim_state, dim_actions, hidden_dims, gamma, epsilon,
                 epsilon_decay, frozen, pretrained, start, tcost, horizon, w,
                 theta, regimes, train_episodes, init_d_size, max_d_size,
                 freeze_after):

        self.dim_state = dim_state
        self.dim_actions = dim_actions
        self.hidden_dims = hidden_dims
        self.optimizer = Adam()
        self.gamma = gamma
        self.eps = epsilon
        self.eps_decay = epsilon_decay
        self.frozen = frozen
        self.pretrained = pretrained
        self.start = start
        self.tcost = tcost
        self.horizon = horizon
        self.w = w
        self.theta = theta
        self.regimes = regimes
        self.train_episodes = train_episodes
        self.init_d_size = init_d_size
        self.max_d_size = max_d_size
        self.freeze_after = freeze_after

    def data(self, eval_w_start, eval_w_end, eval_w_points):

        """
        Generates the state space over which the Q-value approximating network
        is to be evaluated and computes the true Q-values. Only works for
        environments with 2 periods per episode.

        Arguments
        ---------
        :param eval_w_start : float
               Lowest value of wealth component of the state in the evaluation
               state space.
        :param eval_w_end : float
               Highest value of wealth component of the state in the
               evaluation state space.
        :param eval_w_points : int
               Number of evenly spaced wealth components of the state
               between the lowest and highest value.

        Returns
        -------
        :returns x_train : ndarray
                 All states in evaluation state space.
        :returns y_train : ndarray
                 True Q-values for all states in the evaluation state space.
        """

        # initiliaze the RL-agent:
        agent = AgentDQN(dim_state=self.dim_state,
                         dim_actions=self.dim_actions,
                         hidden_dims=self.hidden_dims,
                         optimizer=Adam(),
                         gamma=self.gamma,
                         eps=self.eps,
                         eps_decay=self.eps_decay,
                         frozen=self.frozen,
                         pretrained=self.pretrained)

        # initiliaze the environment:
        env = Env(start=self.start,
                  tcost=self.tcost,
                  horizon=self.horizon,
                  w=self.w,
                  theta=self.theta,
                  regimes=self.regimes)

        assert env.horizon == 2

        x1 = np.arange(0, env.horizon) / env.horizon
        x2 = np.linspace(eval_w_start, eval_w_end, eval_w_points)
        x_train = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)

        # which regimes operate in t=0 and t=1:
        idx0 = [0 in v["periods"] for v in env.regimes.values()].index(True)
        idx1 = [1 in v["periods"] for v in env.regimes.values()].index(True)
        r0 = list(env.regimes.keys())[idx0]  # regime in t=0
        r1 = list(env.regimes.keys())[idx1]  # regime in t=1

        mu0 = env.regimes[r0]["mu"]  # log-returns at t=0
        mu1 = env.regimes[r1]["mu"]  # log-returns at t=1
        sigma0 = env.regimes[r0]["sigma"]
        sigma1 = env.regimes[r1]["sigma"]

        assert np.array_equal(mu0, mu1) is True
        assert np.array_equal(sigma0, sigma1) is True

        y_train = []
        for s in x_train:
            if s[0] == 0:
                opt_w = compute_opt_weight(env, 1)
                tq = 2 * mu0[0] +\
                    (agent.action_space + opt_w) * \
                    (mu0[1] - mu0[0] + sigma0[1] ** 2 / 2) - \
                    0.5 * (agent.action_space ** 2 + opt_w ** 2) * \
                    sigma0[1] ** 2 + np.log(s[1])
            else:
                tq = mu0[0] + agent.action_space * (mu0[1] - mu0[0]) \
                     + \
                     0.5 * agent.action_space * (1 - agent.action_space) * \
                     sigma0[1] ** 2 + np.log(s[1])
            y_train.append(tq)

        self.x = x_train
        self.y = np.array(y_train)

        return self.x, self.y

    def objective(self, params):

        """
        Computes the mean squared error between the networks predictions and
        the true Q-values. Network is trained based on the input parameters.

        Arguments
        ---------
        :param params : dict
               Dictionary containing values for hyperparameters to be
               optimized.

        Returns
        -------
        :returns : dict
                 Dictionary containint mean squared error between true and
                 estimated (using the parameter configuration from params)
                 Q-values and the status of the optimization.
        """

        a = params["lr"]
        b = params["lr decay"]
        c = params["batch size"]
        d = params["target update"]

        # initiliaze the RL-agent:
        agent = AgentDQN(dim_state=self.dim_state,
                         dim_actions=self.dim_actions,
                         hidden_dims=self.hidden_dims,
                         optimizer=Adam(lr=a, decay=b),
                         gamma=self.gamma,
                         eps=self.eps,
                         eps_decay=self.eps_decay,
                         frozen=self.frozen,
                         pretrained=self.pretrained)

        # initiliaze the environment:
        env = Env(start=self.start,
                  tcost=self.tcost,
                  horizon=self.horizon,
                  w=self.w,
                  theta=self.theta,
                  regimes=self.regimes)

        trained_agent, _, _, _, _, _, _ = train_dqn(agent, env,
                                                    self.train_episodes,
                                                    c, self.init_d_size,
                                                    self.max_d_size, d,
                                                    self.freeze_after)

        pred = trained_agent.qnn.predict(self.x)
        true = self.y

        mse = np.mean((pred - true) ** 2)

        return {"loss": mse, "status": STATUS_OK}
