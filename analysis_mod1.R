#### PREAMBLE ####

# date: 14.11.18
# author: Alexander Pitsch

# Analyzes the data created by running a Double DQN algorihtm on a multi-period
# portfolio selection problem setup with 2 periods, 2 assets with i.i.d. 
# normally distributed log-returns, and an investor with log-utility.


#### 0. SETTINGS ####


# 1. disable scientific notation:
options(scipen=999)

# 2. set working directory to directory with data:
setwd("/~")

# 3. set variables:
horizon_dqn <- 2           # number of periods in an episode
actions_dqn <- 11          # number of actions in action space
r_t <- 0                   # mean log return on risky asset
interest <- 0              # interest on riskless asset
r_f <- log(1 + interest)   # log return on riskless asset
sigma <- 1                 # standard deviation of log return on risky asset
gamma <- 1                 # risk aversion coefficient of investor

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

# 5.  optimal allocation to risky asset:
opt <- computeOptimalRisky(r_t, r_f, sigma, gamma)


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
colnames(actions_train_dqn) <- c("episode", as.character(seq(1, horizon_dqn, 1)))
colnames(rewards_train_dqn) <- c("episode", as.character(seq(1, horizon_dqn, 1)))
colnames(loss_train_dqn) <- c("iteration", "loss")
colnames(fu_sim_dqn) <- c("episode", "safe", "myopic", "risky", "dqn")
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
for (i in 1:nrow(pred_actions_dqn)){
  if (round(pred_actions_dqn$state_time[i] * horizon_dqn) == 1){
    val = pred_actions_dqn$action[i] * (r_t - r_f) + 
      0.5 * pred_actions_dqn$action[i] * (1 - pred_actions_dqn$action[i]) * sigma^2 + 
      0.5 * (1 - gamma) * pred_actions_dqn$action[i]^2 * sigma^2 + 
      log(pred_actions_dqn$state_wealth[i]+ 1e-10)
  }
  if (round(pred_actions_dqn$state_time[i]*horizon_dqn) == 0){
    val = 2*r_f + (pred_actions_dqn$action[i] + opt) * (r_t - r_f + 0.5 * sigma^2) - 
      gamma/2 * (pred_actions_dqn$action[i]^2 + opt^2) * sigma^2 + 
      log(pred_actions_dqn$state_wealth[i] + 1e-10)
  }
  pred_actions_dqn$exp_port_ret[i] = val
}
# convert state_time to periods:
pred_actions_dqn$state_time <- pred_actions_dqn$state_time * 2
# rename columns:
colnames(pred_actions_dqn) <- c("action", "est", "period", "wealth", "true")
pred_actions_dqn_tidy <- gather(pred_actions_dqn, "type", "value",
                                -action, -period, -wealth)

# 6. Clean Q-value estimates during training for state (0,1):
colnames(q_val_dqn)[1] <- c("episode")
q_val_dqn$episode <- seq(0, length(q_val_dqn$episode)-1, by = 1)
q_val_dqn_tidy <- gather(q_val_dqn, action, value, -episode)
q_val_dqn_tidy$action <- factor(as.numeric(q_val_dqn_tidy$action))

# 7. Derive strategy from final Q-value estimates:
est_actions_dqn <- subset(pred_actions_dqn, select = -true)
strat <- matrix(nrow = length(unique(est_actions_dqn$period)),
                ncol = length(unique(est_actions_dqn$wealth)))
for (i in unique(est_actions_dqn$period)){
  for (j in unique(est_actions_dqn$wealth)){
    data <- subset(est_actions_dqn, period == i & wealth == j)
    res <- data$action[match(max(data$est), data$est)]
    strat[i+1, match(j, est_actions_dqn$wealth)] <- res
  }
}
strat <- as.data.frame(cbind(unique(est_actions_dqn$period), strat))
colnames(strat) <- c("period", seq(eval_w_start, eval_w_end, by = eval_w_step))
strat_tidy <- gather(strat, "wealth", "action", -period)
strat_tidy$wealth <- as.numeric(strat_tidy$wealth)
strat_tidy$period <- as.factor(strat_tidy$period)

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
  ylab("running average loss") #+
  # coord_cartesian(ylim = c(0, 0.5))

# 4a. Plot average reward during training:
ggplot(rewards_train_dqn_tidy, aes(x=episode, y=reward)) +
  geom_line(data = rewards_train_dqn_tidy %>%
              mutate(reward = cumsum(reward)/episode)) +
  ylab("average reward") +
  coord_cartesian(ylim = c(-0.1, 0.1))

# 4b. Plot running average reward during training:
ggplot(rewards_train_dqn_tidy, aes(x=episode, y=reward)) +
  # geom_point(alpha = 0.05) +
  geom_line(data = rewards_train_dqn_tidy %>%
              mutate(reward = rollmean(reward, 1000, align = "right", fill = NA))) +
  ylab("running average reward")

# 5. Plot cumulative reward during training:
ggplot(rewards_train_dqn_tidy, aes(x=episode, y=reward)) +
  geom_line(data = rewards_train_dqn_tidy %>%
              mutate(reward = cumsum(reward))) +
  ylab("cumulative reward")

# 6. Plot running average of allocation to risky asset during training:
ggplot(actions_train_dqn_tidy, aes(x=episode, y=value, col=time)) +
  geom_point(alpha = 0.05) +
  geom_line(data = actions_train_dqn_tidy %>%
              group_by(time) %>%
              mutate(value = rollmean(value, 1000, align = "right", fill = NA))) +
  ylab("allocation to the risky asset") +
  scale_y_continuous(breaks = pretty(seq(0, 1, by = 0.1)), limits = c(0, 1))

# 7. Plot final allocation strategy suggested by RL:
labels <- c("0" = "period 0",
            "1" = "period 1")
ggplot(strat_tidy, aes(x = wealth, y = action)) +
  geom_point() +
  scale_color_distiller(palette = "OrRd", trans = "reverse") +
  facet_grid(. ~ period, labeller = as_labeller(labels)) +
  scale_y_continuous(breaks = pretty(seq(0, 1, by = 0.1)), limits = c(0, 1)) +
  geom_hline(yintercept = opt, alpha = 0.5, col = "green") +
  xlab(TeX("$\\frac{W_t}{W_0}$")) +
  ylab("allocation to risky asset") +
  ggtitle("Allocations suggested by Q-value estimator network")

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
  coord_cartesian(ylim = c(0.1, 0.3)) +
  geom_hline(yintercept = subset(pred_actions_dqn, period == 0 & wealth == 1)$true)

# 10. Plot Q-values for states (0, start_w_eval) to (0, end_w_eval) for actions 0.3 - 0.7:
ggplot(subset(pred_actions_dqn_tidy, period == 0 & action %in% c(0.3, 0.4, 0.5, 0.6, 0.7)), 
       aes(x=wealth, y = value, col = as.factor(action), linetype = as.factor(type))) +
  geom_line(alpha = 0.5) +
  xlab(TeX("$\\frac{W_t}{W_0}$")) +
  ylab("Q-value") +
  ggtitle("Estimated and true Q-values for actions 0.3 to 0.7 in states (0, 0.5) to (0, 1.5)") +
  guides(col=guide_legend("action")) +
  guides(linetype=guide_legend("type"))

# 11. Plot Q-values for states (1, start_w_eval) to (1, end_w_eval) for actions 0.3 - 0.7:
ggplot(subset(pred_actions_dqn_tidy, period == 1 & action %in% c(0.3, 0.4, 0.5, 0.6, 0.7)), 
       aes(x=wealth, y = value, col = as.factor(action), linetype = as.factor(type))) +
  geom_line(alpha = 0.5) +
  xlab(TeX("$\\frac{W_t}{W_0}$")) +
  ylab("Q-value") +
  ggtitle("Estimated and true Q-values for actions 0.3 to 0.7 in states (0.5, 0.5) to (0.5, 1.5)") +
  guides(col=guide_legend("action")) +
  guides(linetype=guide_legend("type"))

# 12. Plot estimated and true Q-values for states (0, 1) and (0.5, 1):
ggplot(subset(pred_actions_dqn_tidy, wealth == 1.0),
       aes(x=action, y=value, shape = as.factor(type))) +
  facet_grid(. ~ period, labeller = as_labeller(labels)) + 
  geom_point() +
  ggtitle("Estimated and true Q-values for states (0, 1) and (0.5, 1)") + 
  ylab("Q-value") +
  xlab("allocation to risky asset") +
  scale_x_continuous(breaks = pretty(pred_actions_dqn_tidy$action)) +
  scale_shape_manual(values = c(1, 20)) +
  guides(shape=guide_legend("type")) +
  geom_point(data = data.frame(action=seq(0, 1, by=0.1), 
                               av_value=apply(q_val_dqn[20000:30000,], 2, mean)[2:12]),
             aes(x=action, y=av_value), inherit.aes = FALSE, col = "red") +
  geom_point(data = data.frame(action=seq(0, 1, by=0.1), 
                               av_value=apply(q_val_dqn[30000:40000,], 2, mean)[2:12]),
             aes(x=action, y=av_value), inherit.aes = FALSE, col = "yellow")

# 13. Plot estimated and true Q-values for states (0, 0.9) and (0.5, 0.9)
ggplot(subset(pred_actions_dqn_tidy, wealth == 0.9),
       aes(x=action, y=value, shape = as.factor(type))) +
  facet_grid(. ~ period, labeller = as_labeller(labels)) +
  geom_point() +
  ggtitle("Estimated and true Q-values for states (0, 0.9) and (0.5, 0.9)") +
  ylab("Q-value") +
  xlab("allocation to risky asset") +
  scale_x_continuous(breaks = pretty(pred_actions_dqn$action)) +
  scale_shape_manual(values = c(1, 20)) +
  guides(shape=guide_legend("type")) +
  guides(col=guide_legend("period"))

# 14a. Plot simulation results (boxplot):
ggplot(fu_sim_dqn_tidy, aes(x=strategy, y=value)) +
  geom_boxplot() +
  ylab("utility")

# 14b. Plot simulation results (histogram):
ggplot(fu_sim_dqn_tidy, aes(x=value, fill=strategy)) +
  geom_histogram(position="dodge")

# 14c. Plot simulation results (ECDF):
ggplot(fu_sim_dqn_tidy, aes(x=value, col = strategy)) + 
  stat_ecdf() +
  xlab("utility") +
  ylab("cumulative probability")


#### V. TESTS ####


# MSE error between final Q-value estimations and true Q-values over evaluation state space:
score <- mean((pred_actions_dqn$est - pred_actions_dqn$true)**2)

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


# 0. reset working directory to directory where to save plots:
setwd("/~")

# 0b. Average MSE loss between Q-value estimates and targets:
png(filename = "avloss_mod1.png",
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
png(filename = "ravloss_mod1.png",
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
png(filename = "Q_val_dev_0_1_mod1.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(q_val_dqn_tidy, aes(x=episode, y=value, col=action)) + 
  geom_line() + 
  ylab("estimated Q-value")

dev.off()

# 1b. Development of Q-value estimates for state (0, 1) in training with ZOOM:
png(filename = "Q_val_dev_0_1_zoom_mod1.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(q_val_dqn_tidy, aes(x=episode, y=value, col=action)) + 
  geom_line() + 
  ylab("estimated Q-value") +
  coord_cartesian(ylim = c(0.1, 0.4))

dev.off()

# 2. Final Q-values for state (0, 1) in training:
png(filename = "Q_val_est_true_0_1_mod1.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(subset(pred_actions_dqn_tidy, wealth == 1.0),
       aes(x=action, y=value, shape = as.factor(type))) +
  facet_grid(. ~ period, labeller = as_labeller(labels)) + 
  geom_point() +
  ylab("Q-value") +
  xlab("allocation to risky asset") +
  scale_x_continuous(breaks = pretty(pred_actions_dqn_tidy$action)) +
  scale_shape_manual(values = c(1, 20)) +
  guides(shape=guide_legend("type"))

dev.off()

# 3. Policy suggested by Q-value estimator network:
png(filename = "RL_policy_mod1.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(strat_tidy, aes(x = wealth, y = action)) +
  geom_point() +
  scale_color_distiller(palette = "OrRd", trans = "reverse") +
  facet_grid(. ~ period, labeller = as_labeller(labels)) +
  scale_y_continuous(breaks = pretty(seq(0, 1, by = 0.1)), limits = c(0, 1)) +
  geom_hline(yintercept = opt, alpha = 0.5, col = "green") +
  xlab(TeX("$\\frac{W_t}{W_0}$")) +
  ylab("allocation to risky asset")

dev.off()

# 4a. Plot boxplot of terminal utility of wealth from simulation:
png(filename = "sim_box_mod1.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(fu_sim_dqn_tidy, aes(x=strategy, y=value)) +
  geom_boxplot() +
  ylab("utility")

dev.off()

# 4b. Plot ECDF of terminal utility of wealth from simulation:

png(filename = "sim_ecdf_mod1.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(fu_sim_dqn_tidy, aes(x=value, col = strategy)) + 
  stat_ecdf() +
  xlab("utility") +
  ylab("cumulative probability")

dev.off()

# 4a. Plot
png(filename = "Q_val_est_tru_mult_0_mod1.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(subset(pred_actions_dqn_tidy, period == 0 & action %in% c(0.3, 0.4, 0.5, 0.6, 0.7)), 
       aes(x=wealth, y = value, col = as.factor(action), linetype = as.factor(type))) +
  geom_line(alpha = 0.5) +
  xlab(TeX("$\\frac{W_t}{W_0}$")) +
  ylab("Q-value") +
  guides(col=guide_legend("action")) +
  guides(linetype=guide_legend("type"))

dev.off()

# 4b. Plot
png(filename = "Q_val_est_tru_mult_1_mod1.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(subset(pred_actions_dqn_tidy, period == 1 & action %in% c(0.3, 0.4, 0.5, 0.6, 0.7)), 
       aes(x=wealth, y = value, col = as.factor(action), linetype = as.factor(type))) +
  geom_line(alpha = 0.5) +
  xlab(TeX("$\\frac{W_t}{W_0}$")) +
  ylab("Q-value") +
  guides(col=guide_legend("action")) +
  guides(linetype=guide_legend("type"))

dev.off()

# 5a. Plot average reward development during training:

png(filename = "avrew_mod1.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(rewards_train_dqn_tidy, aes(x=episode, y=reward)) +
  geom_line(data = rewards_train_dqn_tidy %>%
              mutate(reward = cumsum(reward)/episode)) +
  ylab("average reward") +
  coord_cartesian(ylim = c(-0.1, 0.1))

dev.off()

# 5b. Plot running average reward development during training:

png(filename = "ravrew_mod1.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(rewards_train_dqn_tidy, aes(x=episode, y=reward)) +
  geom_line(data = rewards_train_dqn_tidy %>%
              mutate(reward = rollmean(reward, 50000, align = "right", fill = NA))) +
  ylab("running average reward")

dev.off()
