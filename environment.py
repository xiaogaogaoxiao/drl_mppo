import numpy as np
import math
from helpers import compute_trade, crra_utility


class Env:

    def __init__(self, *args, **kwargs):

        """
        Initializes the environment the agent is in by calling specialized
        "init"-method. Saves the passed arguments as default arguments.

        Parameters
        ----------
        :param args : arguments
        :param kwargs : keyword arguments

        Returns
        -------
        None.
        """

        self.__default_args__ = args
        self.__default_kwargs__ = kwargs
        self.init(*args, **kwargs)

    def init(self, start, tcost, horizon, w, theta, regimes):

        """
        Specialized method to initialize the the environment instance.

        Parameters
        ----------
        :param start : string
               Specifies how the initial wealth is distributed over the two
               assets.
        :param tcost : float
               Specifies the proportional cost each trade incurs.
        :param horizon : int
               Specifies the investment horizon and thus number of periods
               per episode.
        :param w : float
               Specifies the investor's initial wealth.
        :param theta : float
               Specifies the investor's risk aversion factor in the CRRA
               utility function.
        :param regimes : dict
               Specifies different regimes of the financial market.

        Returns
        -------
        None.
        """

        self.start = start
        self.tcost = tcost
        self.horizon = horizon
        self.w = w
        self.theta = theta
        self.regimes = regimes

        self.time = 0
        self.n = 2
        self.done = False

        for v in list(self.regimes.values()):
            assert v["mu"].shape[0] == v["sigma"].shape[0] == self.n

        if start is 'random':
            # initializes portfolio weights uniform randomly for n assets:
            self.p = np.random.rand(self.n)
            self.p /= np.sum(self.p)
            # converts portfolio weights to dollar values:
            self.p = self.p * self.w

        if start is 'safe':
            # initialize portfolio as empty np.array:
            self.p = np.zeros(self.n)
            # invest full wealth into safe asset:
            self.p[0] = self.w

        if start is 'risky':
            # initialize portfolio as empty np.array:
            self.p = np.zeros(self.n)
            # invest full wealth into risky asset:
            self.p[1] = self.w

        self.init_u = self.get_utility()

    def get_utility(self):

        """
        Returns the utility for current portfolio.

        Parameters
        ----------
        None.

        Returns
        -------
        :returns u : float
                 Utility value of current portfolio.
        """

        w = np.sum(self.p)
        u = crra_utility(w, theta=self.theta)

        return u

    def trade(self, trade_vec):

        """
        Implements trade specified in the input trade vector in the portfolio.

        Parameters
        ----------
        :param trade_vec : ndarray
               Desired absolute changes in portfolio.

        Returns
        -------
        None.
        """

        # apply changes in position resulting from trading:
        self.p = self.p + trade_vec
        # transaction costs are subtracted from cash:
        self.p[0] = self.p[0] - np.sum(self.tcost*np.abs(trade_vec[1:]))

    def update(self):

        """
        Updates the environment, i.e. realizes the returns on the portfolio
        and increments time variable.

        Parameters
        ----------
        None.

        Returns
        -------
        :returns sgr : ndarray
                 Realized simple gross returns at this timestep.
        """

        for k, v in self.regimes.items():
            if self.time in v["periods"]:
                current_regime = k

        temp_mu = self.regimes[current_regime]["mu"]
        temp_sigma = self.regimes[current_regime]["sigma"]

        # compute the log-returns for all assets:
        log_r = temp_sigma * np.random.randn(self.n) + temp_mu
        # compute the simple gross returns for all assets:
        sgr = np.exp(log_r)
        # update portfolio according to asset returns:
        self.p *= sgr
        # increment time:
        self.time += 1
        # check if episode is finished:
        if self.time == self.horizon:
            self.done = True

        return sgr

    def get_state(self):

        """
        Gets the current state of the agent.

        Parameters
        ----------
        None.

        Returns
        -------
        :returns s : ndarray
                 The current state configuration.
        """

        s = np.array([[self.time/self.horizon] + [np.sum(self.p)/self.w]])

        return s

    def take_action(self, action):

        """
        Takes the agents action as an input and simulates the state
        transition. Returns the reward, the next state, a done flag, and the
        simple gross returns that have been realized on the assets in this
        step.

        Parameters
        ----------
        :param action : float
               The action the agent wants to take in terms of the desired
               share of the risky asset in the portfolio.

        Returns
        -------
        :returns r : float
                 The reward from taking the desired action.
        :returns ss : ndarray
                 The state the agent is in after taking the action.
        :returns self.done : bool
                 Flag whether the environment is in a terminal state.
        :returns sgr : ndarray
                 Simple gross returns realized on the portfolio holdings in
                 this period.
        """

        # compute trade vector to implement the desired allocation action:
        trade = compute_trade(self.p, action, self.tcost)

        # make the trade:
        self.trade(trade)

        # assert that the trade gives desired allocation:
        assert math.isclose(self.p[1] / np.sum(self.p), action)

        # realize returns for this timestep:
        sgr = self.update()

        # compute reward:
        if self.done:
            r = self.get_utility() - self.init_u
        else:
            r = 0

        # observe new state:
        ss = self.get_state()

        return r, ss, self.done, sgr

    def reset(self):

        """
        Resets the environment to the state it has been initialized to.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        self.init(*self.__default_args__, **self.__default_kwargs__)
