# drl_mppo
Deep reinforcement learning for multi-period portfolio optimization


#### agents.py 
- specifies the agents for tabular Q-learning, DQN, REINFORCE policy gradient and actor-critic policy gradient

#### environment.py
- specifies the environment for the reinforcement learning algorithms

#### helpers.py
- contains helpful functions

#### train.py
- specifies the training algorithm for the different agents

#### simulate.py
- specifies the simulation process for the different agents

#### param_optimization.py
- implement random search parameter optimization

#### main_*.py
- main files take the input parameters for the agents and the environment, train the agent, and compare the trained agent strategy against three benchmark strategies on the same return series

#### analysis_*.R
- analyze .csv outputs of main_dqn.py models
