#### PREAMBLE ####

# date: 14.11.18
# author: Alexander Pitsch

# Analyzes the data created by running a Double DQN algorihtm on a multi-period
# portfolio selection problem setup with 252 periods, 2 assets with 
# independently but not identically normally distributed log-returns, and an 
# investor with log-utility.


#### 0. SETTINGS ####

# 1. disable scientific notation:
options(scipen=999)

# 2. set working directory to directory with data:
setwd("/~")

# 3. set variables:
horizon_dqn <- 252              # number of periods in an episode
actions_dqn <- 11               # number of actions in action space
r_t_1 <- -0.00007               # mean log return on risky asset in regime 1
r_t_2 <- 0.0001123              # mean log return on risky asset in regime 2
interest_1 <- 0.0               # interest on riskless asset in regime 1
interest_2 <- 0.00008           # interest on riskless asset in regime 2
r_f_1 <- log(1 + interest_1)    # log return on riskless asset in regime 1
r_f_2 <- log(1 + interest_2)    # log return on riskless asset in regime 2
sigma_1 <- 0.015                # sd of log return on risky asset in regime 1
sigma_2 <- 0.009                # sd of log return on risky asset in regime 2
gamma <- 1                      # risk aversion coefficient of investor
crisis <- seq(126, 188, by = 1)   # periods for crisis regime 1


# To evaluate the final Q-values, we need to define a state space over 
# which they are evaluated. In the definition of the RL problem, the state
# consists of a time component and a wealth component. While the time component
# is discrete, the wealth component is continuous. By specifying a start value, 
# an ending value and a step size, we convert it to a discrete scale as well.

eval_w_start <- 0.1      
eval_w_end <- 2.1
eval_w_step <- 0.01

# 4. define helper functions:
computeOptimalRisky <- function(r_t, r_f, sigma, gamma){
  opt <- (r_t - r_f + sigma^2/2)/(gamma * sigma^2)
  return(opt)
}

opt_1 <- computeOptimalRisky(r_t_1, r_f_1, sigma_1, gamma)
opt_2 <- computeOptimalRisky(r_t_2, r_f_2, sigma_2, gamma)

#### I. LIBRARIES ####


library(ggplot2)
library(tidyr)
library(dplyr)
library(zoo)
library(data.table)
library(tseries)
library(scales)
library(stringr)
library(rgl)
library(latex2exp)

# reformat ggplot titles:
theme_update(plot.title = element_text(face = "bold", hjust = 0.5))


#### II. DATA IMPORT ####


actions_train_dqn <- fread("actions_train_dqn.csv", header = TRUE)
rewards_train_dqn <- fread("rewards_train_dqn.csv", header = TRUE)
loss_train_dqn <- fread("loss_train_dqn.csv", header = TRUE)
log_train_dqn <- fread("log_train_dqn.csv", header = TRUE)
alloc_sim_dqn <- fread("alloc_sim_dqn.csv", header = TRUE)
fu_sim_dqn <- fread("fu_sim_dqn.csv", header = TRUE)
euw_vs_uew_dqn <- fread("euw_vs_uew_dqn.csv", header = TRUE)

pred_action_filenames <- list.files(pattern = "pred_action_.*_dqn.csv")
r_names <- str_extract(pred_action_filenames, '.*(?=\\.csv)')
for (i in r_names){
  filepath = file.path(paste(i, ".csv", sep = ""))
  assign(i, fread(filepath, header = TRUE))
}

pred_actions_dqn <- fread("pred_actions_dqn.csv", header = TRUE)
q_val_dqn <- fread("qval_train_dqn.csv", header = TRUE)


#### III. DATA MANIPULATION ####


# 1a. Name columns in dataframes:
colnames(loss_train_dqn) <- c("iteration", "loss")
colnames(fu_sim_dqn) <- c("episode", "safe", "myopic", "risky", "dqn")
colnames(actions_train_dqn) <- c("episode", as.character(seq(1, horizon_dqn, 1)))
colnames(rewards_train_dqn) <- c("episode", as.character(seq(1, horizon_dqn, 1)))
colnames(euw_vs_uew_dqn) <- c("action", "euw", "uew")

# 1b. Correct episode numbering:
actions_train_dqn$episode <- seq(0, length(actions_train_dqn$episode)-1, by = 1)
rewards_train_dqn$episode <- seq(0, length(rewards_train_dqn$episode)-1, by = 1)
loss_train_dqn$iteration <- seq(0, length(loss_train_dqn$iteration)-1, by = 1)

# 2. Clean expected utility of wealth vs. utility of expected wealth data:
euw_vs_uew_dqn$action <- seq(0, 1, length.out = length(euw_vs_uew_dqn$action))
euw_vs_uew_dqn_tidy <- gather(euw_vs_uew_dqn, "type", "value", -action)

# 3. Clean actions during training data:
actions_train_dqn_tidy <- gather(
  actions_train_dqn, "time", "value", -episode)

# 4. Clean rewards during training data:
to_keep = c("episode", as.character(horizon_dqn))
rewards_train_dqn_tidy <- rewards_train_dqn[, ..to_keep]
colnames(rewards_train_dqn_tidy) <- c("episode", "reward")

# 5. Clean final Q-value estimates data:
pred_actions_dqn <- subset(pred_actions_dqn, select = -V1)
pred_actions_dqn$action <- pred_actions_dqn$action / (actions_dqn - 1)
# convert state_time to periods:
pred_actions_dqn$state_time <- pred_actions_dqn$state_time * horizon_dqn
# rename columns:
colnames(pred_actions_dqn) <- c("action", "est", "period", "wealth")

# 6. Clean Q-value estimates during training for state (0,1):
colnames(q_val_dqn)[1] <- c("episode")
q_val_dqn$episode <- seq(0, length(q_val_dqn$episode) -1, by = 1)
q_val_dqn_tidy <- gather(q_val_dqn, action, value, -episode)
q_val_dqn_tidy$action <- factor(as.numeric(q_val_dqn_tidy$action))

# 7. Derive strategy from final Q-value estimates:
est_actions_dqn <- pred_actions_dqn
strat <- matrix(nrow = length(unique(est_actions_dqn$period)),
                ncol = length(unique(est_actions_dqn$wealth)))
opt_strat <- matrix(nrow = length(unique(est_actions_dqn$period)),
                    ncol = length(unique(est_actions_dqn$wealth)))
for (i in unique(est_actions_dqn$period)){
  for (j in unique(est_actions_dqn$wealth)){
    col <- match(j, unique(est_actions_dqn$wealth))
    data <- subset(est_actions_dqn, period == i & wealth == j)
    res <- data$action[match(max(data$est), data$est)]
    strat[i+1, col] <- res
    if (i %in% crisis){
      opt_strat[i+1, col] <- opt_1
    }
    else {
      opt_strat[i+1, col] <- opt_2
    }
  }
}
strat <- as.data.frame(cbind(unique(est_actions_dqn$period), strat))
opt_strat <- as.data.frame(cbind(unique(est_actions_dqn$period), opt_strat))
colnames(strat) <- c("period", seq(eval_w_start, eval_w_end, by = eval_w_step))
colnames(opt_strat) <- c("period", seq(eval_w_start, eval_w_end, by = eval_w_step))
strat_tidy <- gather(strat, "wealth", "action", -period)
opt_strat_tidy <- gather(opt_strat, "wealth", "action", -period)
strat_tidy$wealth <- as.numeric(strat_tidy$wealth)
strat_tidy$period <- as.factor(strat_tidy$period)
opt_strat_tidy$wealth <- as.numeric(opt_strat_tidy$wealth)
opt_strat_tidy$period <- as.factor(opt_strat_tidy$period)

# 8. Tidy simulation data:
fu_sim_dqn_tidy <- gather(fu_sim_dqn, "strategy", "value", -episode)
fu_sim_dqn_tidy$strategy <- as.factor(fu_sim_dqn_tidy$strategy)


#### IV. PLOTS ####


# 1. Plot expected utility of wealth vs. utility of expected wealth:
ggplot(euw_vs_uew_dqn_tidy, aes(x=action, y=value, col=type)) +
  geom_point() +
  scale_x_continuous(breaks = pretty(euw_vs_uew_dqn_tidy$action))

# 2. Plot development of average MSE loss between estimates and targets:
ggplot(loss_train_dqn, aes(x=iteration, y=loss)) +
  geom_point(alpha = 0.01) +
  geom_line(data = loss_train_dqn[0:length(loss_train_dqn$loss),] %>% 
              mutate(loss = cumsum(loss)/iteration)) +
  ylab("average loss") +
  coord_cartesian(ylim = c(0, 0.075))

# 3. Plot running average of MSE loss between estimates and targets:
ggplot(loss_train_dqn, aes(x=iteration, y=loss)) +
  geom_line(data = loss_train_dqn[0:length(loss_train_dqn$loss),] %>% 
              mutate(loss = rollmean(loss, 10000, align = "right", fill = NA))) +
  ylab("running average loss")
 
# 4. Plot average reward during training:
ggplot(rewards_train_dqn_tidy, aes(x=episode, y=reward)) +
  geom_line(data = rewards_train_dqn_tidy %>%
              mutate(reward = cumsum(reward)/episode)) +
  ylab("average reward")
 
# 5. Plot running average reward during training:
ggplot(rewards_train_dqn_tidy, aes(x=episode, y=reward)) +
  geom_line(data = rewards_train_dqn_tidy %>%
              mutate(reward = rollmean(reward, 100, align = "right", fill = NA))) +
  ylab("running average reward")
 
# 6a. Plot final allocation strategy suggested by RL:
ggplot(strat_tidy, aes(x = period, y = wealth, fill=action)) +
  geom_tile() +
  scale_fill_distiller(palette = "Spectral",limits = c(0, 1)) +
  scale_x_discrete(breaks = pretty(as.numeric(unique(strat_tidy$period)))) +
  ylab(TeX("$\\frac{W_t}{W_0}$")) +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.5)) +
  ggtitle("Allocations suggested by Q-value estimator network")

# 6b. Plot optimal strategy:
ggplot(opt_strat_tidy, aes(x = period, y = wealth, fill=action)) +
  geom_tile() +
  scale_fill_distiller(palette = "Spectral", limits = c(0, 1)) +
  scale_x_discrete(breaks = pretty(as.numeric(unique(opt_strat_tidy$period)))) +
  ylab(TeX("$\\frac{W_t}{W_0}$")) +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.5)) +
  ggtitle("Allocations required by optimal allocation strategy")

# 6c. Plot final allocation strategy suggested by RL:
x <- 1:horizon_dqn
y <- seq(eval_w_start, eval_w_end, by = eval_w_step)
open3d()
persp3d(x, y, as.matrix(subset(strat, select = -period)),
        xlab = "period", 
        ylab = "wealth component", 
        zlab = "suggested allocation",
        col = "red")

# 8a. Plot development of estimated Q-value for state (0, 1):
ggplot(q_val_dqn_tidy, aes(x=episode, y=value, col=action)) + 
  geom_line() + 
  ylab("estimated Q-value") +
  ggtitle("Development of Q-value estimate for state (0, 1)")

# 8b. Plot development of estimated Q-value for state (0, 1) with zoom:
ggplot(q_val_dqn_tidy, aes(x=episode, y=value, col=action)) + 
  geom_line() + 
  ylab("estimated Q-value") +
  ggtitle("Development of Q-value estimate for state (0, 1) - ZOOM") +
  coord_cartesian(ylim = c(3, 6))

ggplot(fu_sim_dqn_tidy, aes(x=strategy, y=value)) +
  geom_boxplot() +
  ylab("utility")

ggplot(fu_sim_dqn_tidy, aes(x=value, col = strategy)) + 
  stat_ecdf() +
  xlab("utility") +
  ylab("cumulative probability")


#### V. TESTS ####


# test for differences in the mean utility of terminal wealth between portfolio strategies:
  # H_0: No difference in means
  # H_1: Difference is > 0 (first > second)
t.test(fu_sim_dqn$myopic, fu_sim_dqn$safe, paired = FALSE, alternative = "g")
t.test(fu_sim_dqn$myopic, fu_sim_dqn$dqn, paired = FALSE, alternative = "g")
t.test(fu_sim_dqn$dqn, fu_sim_dqn$safe, paired = FALSE, alternative = "g")
t.test(fu_sim_dqn$dqn, fu_sim_dqn$risky, paired = FALSE, alternative = "g")

# test for differences in cumulative distibutions between utilities from different portfolio strategies:
ks.test(fu_sim_dqn$myopic, fu_sim_dqn$dqn)


#### VI. EXPORT HIGH QUALITY PLOTS ####


# 0. reset working directory to directory where plots should be saved:
setwd("/~")


# 0b. Average MSE loss between Q-value estimates and targets:
png(filename = "avloss_mod3.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(loss_train_dqn, aes(x=iteration, y=loss)) +
  geom_point(alpha = 0.01) +
  geom_line(data = loss_train_dqn[0:length(loss_train_dqn$loss),] %>% 
              mutate(loss = cumsum(loss)/iteration)) +
  ylab("average loss") +
  coord_cartesian(ylim = c(0, 0.075))

dev.off()

# 0c. Running average MSE loss between Q-value estimates and targets:
png(filename = "ravloss_mod3.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(loss_train_dqn, aes(x=iteration, y=loss)) +
  geom_line(data = loss_train_dqn[0:length(loss_train_dqn$loss),] %>% 
              mutate(loss = rollmean(loss, 10000, align = "right", fill = NA))) +
  ylab("running average loss")

dev.off()

# 1a. Development of Q-value estimates for state (0, 1) in training:
png(filename = "Q_val_dev_0_1_mod3.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(q_val_dqn_tidy, aes(x=episode, y=value, col=action)) +
  geom_line() +
  ylab("estimated Q-value")

dev.off()

# 1b. Development of Q-value estimates for state (0, 1) in training with ZOOM:
png(filename = "Q_val_dev_0_1_zoom_mod3.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(q_val_dqn_tidy, aes(x=episode, y=value, col=action)) +
  geom_line() +
  ylab("estimated Q-value") +
  coord_cartesian(ylim = c(0.1, 0.4))

dev.off()

# 3a. Plot policy suggested by Q-value estimator network:
png(filename = "RL_policy_mod3.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(strat_tidy, aes(x = period, y = wealth, fill=action)) +
  geom_tile() +
  scale_fill_distiller(palette = "Spectral", limits = c(0, 1)) +
  scale_x_discrete(breaks = pretty(as.numeric(unique(strat_tidy$period)))) +
  ylab(TeX("$\\frac{W_t}{W_0}$")) +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.5))

dev.off()

# 3b. Plot optimal strategy:
png(filename = "opt_policy_mod3.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(opt_strat_tidy, aes(x = period, y = wealth, fill=action)) +
  geom_tile() +
  scale_fill_distiller(palette = "Spectral", limits = c(0, 1)) +
  scale_x_discrete(breaks = pretty(as.numeric(unique(opt_strat_tidy$period)))) +
  ylab(TeX("$\\frac{W_t}{W_0}$")) +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.5))

dev.off()

# 4a. Plot boxplot of terminal utility of wealth from simulation:
png(filename = "sim_box_mod3.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(fu_sim_dqn_tidy, aes(x=strategy, y=value)) +
  geom_boxplot() +
  ylab("utility")

dev.off()

# 4b. Plot ECDF of terminal utility of wealth from simulation:

png(filename = "sim_ecdf_mod3.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(fu_sim_dqn_tidy, aes(x=value, col = strategy)) + 
  stat_ecdf() +
  xlab("utility") +
  ylab("cumulative probability")

dev.off()


# 5a. Plot average reward development during training:

png(filename = "avrew_mod3.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(rewards_train_dqn_tidy, aes(x=episode, y=reward)) +
  geom_line(data = rewards_train_dqn_tidy %>%
              mutate(reward = cumsum(reward)/episode)) +
  ylab("average reward")

dev.off()

# 5b. Plot running average reward development during training:

png(filename = "ravrew_mod3.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(rewards_train_dqn_tidy, aes(x=episode, y=reward)) +
  geom_line(data = rewards_train_dqn_tidy %>%
              mutate(reward = rollmean(reward, 100, align = "right", fill = NA))) +
  ylab("running average reward")

dev.off()
