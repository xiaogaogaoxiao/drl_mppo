#### PREAMBLE ####

# date: 14.11.18
# author: Alexander Pitsch

# Analyzes the data created by running a Double DQN algorihtm on a multi-period
# portfolio selection problem setup with 2 periods, 2 assets with independently
# but not identically normally distributed log-returns, and an investor with 
# log-utility.

#### 0. SETTINGS ####


# 1. disable scientific notation:
options(scipen=999)

# 2. set working directory to directory containing data files:
setwd("/~")

# 3. set variables:
horizon_dqn <- 2            # number of periods in an episode
actions_dqn <- 11           # number of actions in action space
r_t_1 <- 0.07               # mean log return on risky asset at t=1
r_t_2 <- 0.0                # mean log return on risky asset at t=2
interest_1 <- 0.02          # interest on riskless asset at t=1
interest_2 <- 0.0           # interest on riskless asset at t=1
r_f_1 <- log(1 + interest_1)   # log return on riskless asset at t=1
r_f_2 <- log(1 + interest_2)   # log return on riskless asset at t=2
sigma_1 <- 0.5              # standard deviation of log return on risky asset at t=1
sigma_2 <- 1                # standard deviation of log return on risky asset at t=2
gamma <- 1                  # risk aversion coefficient of investor

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
pred_actions_dqn$state_time <- pred_actions_dqn$state_time * 2
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
    data <- subset(est_actions_dqn, period == i & wealth == j)
    res <- data$action[match(max(data$est), data$est)]
    strat[i+1, match(j, est_actions_dqn$wealth)] <- res
    if (i == 0){
      opt_strat[i+1, match(j, est_actions_dqn$wealth)] <- opt_1
    }
    if (i == 1){
      opt_strat[i+1, match(j, est_actions_dqn$wealth)] <- opt_2
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
  # geom_point(alpha = 0.05) +
  geom_line(data = rewards_train_dqn_tidy %>%
              mutate(reward = cumsum(reward)/episode)) +
  ylab("average reward")

# 5. Plot running average reward during training:
ggplot(rewards_train_dqn_tidy, aes(x=episode, y=reward)) +
  geom_line(data = rewards_train_dqn_tidy %>%
              mutate(reward = rollmean(reward, 50000, align = "right", fill = NA))) +
  ylab("running average reward")

# 6. Plot running average of allocation to risky asset during training:
ggplot(actions_train_dqn_tidy, aes(x=episode, y=value, col=time)) +
  geom_point(alpha = 0.05) +
  geom_line(data = actions_train_dqn_tidy %>%
              group_by(time) %>%
              mutate(value = rollmean(value, 10000, align = "right", fill = NA))) +
  ylab("allocation to the risky asset") +
  scale_y_continuous(breaks = pretty(seq(0, 1, by=0.1)), limits = c(0,1))

# 7. Plot final allocation strategy suggested by RL:
labels <- c("0" = "period 0",
            "1" = "period 1")
ggplot(strat_tidy, aes(x = wealth, y = action)) +
  geom_point() +
  scale_color_distiller(palette = "OrRd", trans = "reverse") +
  facet_grid(. ~ period, labeller = as_labeller(labels)) +
  scale_y_continuous(breaks = pretty(seq(0, 1, by = 0.1)), limits = c(0, 1)) +
  geom_segment(data=opt_strat_tidy, aes(x=min(wealth), xend=max(wealth), 
                                        y=action, yend=action), 
               alpha = 0.5, col = "green") +
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
  coord_cartesian(ylim = c(0.3, 0.5))

# 10. Plot Q-values for states (0, start_w_eval) to (0, end_w_eval) for actions 0.3 - 0.7:
ggplot(subset(pred_actions_dqn, period == 0 & action %in% c(0.3, 0.4, 0.5, 0.6, 0.7)), 
       aes(x=wealth, y = est, col = as.factor(action))) +
  geom_line(alpha = 0.5) +
  xlab(TeX("$\\frac{W_t}{W_0}$"))  +
  ylab("Q-value") +
  ggtitle("Estimated Q-values for actions 0.3 to 0.7 in states (0, 0.1) to (0, 2)") +
  guides(col=guide_legend("action"))

# 11. Plot Q-values for states (0.5, start_w_eval) to (0.5, end_w_eval) for actions 0.3 - 0.7:
ggplot(subset(pred_actions_dqn, period == 1 & action %in% c(0.3, 0.4, 0.5, 0.6, 0.7)), 
       aes(x=wealth, y = est, col = as.factor(action))) +
  geom_line(alpha = 0.5) +
  xlab(TeX("$\\frac{W_t}{W_0}$"))  +
  ylab("Q-value") +
  ggtitle("Estimated Q-values for actions 0.3 to 0.7 in states (0.5, 0.1) to (0.5, 2)") +
  guides(col=guide_legend("action"))

# 12. Plot estimated Q-values for states (0, 1) and (0.5, 1):
ggplot(subset(pred_actions_dqn, wealth == 1.0),
       aes(x=action, y=est)) +
  facet_grid(. ~ period, labeller = as_labeller(labels)) + 
  geom_point(shape = 1) +
  ggtitle("Estimated Q-values for states (0, 1) and (0.5, 1)") + 
  ylab("Q-value") +
  xlab("allocation to risky asset") +
  scale_x_continuous(breaks = pretty(pred_actions_dqn$action))

# 13. Plot estimated Q-values for states (0, 0.9) and (0.5, 0.9)
ggplot(subset(pred_actions_dqn, wealth == 0.9),
       aes(x=action, y=est)) +
  facet_grid(. ~ period, labeller = as_labeller(labels)) +
  geom_point(shape = 1) +
  ggtitle("Estimated Q-values for states (0, 0.9) and (0.5, 0.9)") +
  ylab("Q-value") +
  xlab("action") +
  scale_x_continuous(breaks = pretty(pred_actions_dqn$action))

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


# test for differences in the mean utility of terminal wealth between portfolio strategies:
   # H_0: No difference in means
   # H_1: Difference is > 0 (first > second)
t.test(fu_sim_dqn$myopic, fu_sim_dqn$safe, paired = FALSE, alternative = "g")
t.test(fu_sim_dqn$myopic, fu_sim_dqn$dqn, paired = FALSE, alternative = "g")
t.test(fu_sim_dqn$dqn, fu_sim_dqn$risky, paired = FALSE, alternative = "g")
t.test(fu_sim_dqn$dqn, fu_sim_dqn$safe, paired = FALSE, alternative = "g")

# test for differences in cumulative distibutions between utilities from different portfolio strategies:
ks.test(fu_sim_dqn$myopic, fu_sim_dqn$dqn)


#### VI. EXPORT HIGH QUALITY PLOTS ####


# 0. reset working directory:
setwd("/home/alexander/Documents/studies/Semester 12_MBF/Masterarbeit/thesis/")

# 0b. Average MSE loss between Q-value estimates and targets:
png(filename = "avloss_mod2.png",
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
png(filename = "ravloss_mod2.png",
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
png(filename = "Q_val_dev_0_1_mod2.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(q_val_dqn_tidy, aes(x=episode, y=value, col=action)) +
  geom_line() +
  ylab("estimated Q-value")

dev.off()

# 1b. Development of Q-value estimates for state (0, 1) in training with ZOOM:
png(filename = "Q_val_dev_0_1_zoom_mod2.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(q_val_dqn_tidy, aes(x=episode, y=value, col=action)) +
  geom_line() +
  ylab("estimated Q-value") +
  coord_cartesian(ylim = c(0.1, 0.4))

dev.off()

# 2. Final Q-values for states (0, 1) and (0.5, 1) after training:
png(filename = "Q_val_est_0_1_mod2.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(subset(pred_actions_dqn, wealth == 1.0),
       aes(x=action, y=est)) +
  facet_grid(. ~ period, labeller = as_labeller(labels)) + 
  geom_point(shape = 1) +
  ylab("Q-value") +
  xlab("allocation to risky asset") +
  scale_x_continuous(breaks = pretty(pred_actions_dqn$action))

dev.off()

# 3. Policy suggested by Q-value estimator network:
png(filename = "RL_policy_mod2.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(strat_tidy, aes(x = wealth, y = action)) +
  geom_point() +
  scale_color_distiller(palette = "OrRd", trans = "reverse") +
  facet_grid(. ~ period, labeller = as_labeller(labels)) +
  scale_y_continuous(breaks = pretty(seq(0, 1, by = 0.1)), limits = c(0, 1)) +
  geom_segment(data=opt_strat_tidy, aes(x=min(wealth), xend=max(wealth), 
                                        y=action, yend=action), 
               alpha = 0.5, col = "green") +
  xlab(TeX("$\\frac{W_t}{W_0}$")) +
  ylab("allocation to risky asset")

dev.off()

# 4a. Plot boxplot of terminal utility of wealth from simulation:
png(filename = "sim_box_mod2.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(fu_sim_dqn_tidy, aes(x=strategy, y=value)) +
  geom_boxplot() +
  ylab("utility")

dev.off()

# 4b. Plot ECDF of terminal utility of wealth from simulation:

png(filename = "sim_ecdf_mod2.png",
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

png(filename = "avrew_mod2.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(rewards_train_dqn_tidy, aes(x=episode, y=reward)) +
  geom_line(data = rewards_train_dqn_tidy %>%
              mutate(reward = cumsum(reward)/episode)) +
  ylab("average reward") +
  coord_cartesian(ylim = c(-0.05, 0.05))

dev.off()

# 5b. Plot running average reward development during training:

png(filename = "ravrew_mod2.png",
    units = "in",
    width = 8,
    height = 5,
    res = 300)

ggplot(rewards_train_dqn_tidy, aes(x=episode, y=reward)) +
  geom_line(data = rewards_train_dqn_tidy %>%
              mutate(reward = rollmean(reward, 50000, align = "right", fill = NA))) +
  ylab("running average reward")

dev.off()
