library(tidyverse)
library(vroom)
library(tidymodels)
library(embed)
library(discrim)
library(naivebayes)
library(kknn)
library(lightgbm)
library(kernlab)
library(themis)

data <- vroom("./data.csv")
data$shot_made_flag <- factor(data$shot_made_flag)
data$time_remaining <- 48 - data$period * 12 + data$minutes_remaining
data$lastminutes <- ifelse(data$time_remaining <= 180, 1, 0)
data$game_num <- as.numeric(data$game_date)

data$first_team <- ifelse((data$game_num >= 395 & data$game_num <= 474) | 
                            (data$game_num >= 494 & data$game_num <= 575) | 
                            (data$game_num >= 588 & data$game_num <= 651) | 
                            (data$game_num >= 740 & data$game_num <= 819) | 
                            (data$game_num >= 827 & data$game_num <= 903) | 
                            (data$game_num >= 909 & data$game_num <= 990) | 
                            (data$game_num >= 1012 & data$game_num <= 1093) | 
                            (data$game_num >= 1117 & data$game_num <= 1189) | 
                            (data$game_num >= 1213 & data$game_num <= 1294) | 
                            (data$game_num >= 1305 & data$game_num <= 1362) | 
                            (data$game_num >= 1375 & data$game_num <= 1452), 
                          1, 0)
data$scoring_leader <- ifelse((data$game_num >= 740 & data$game_num <= 819) | 
                                (data$game_num >= 827 & data$game_num <= 903), 
                              1, 0)

# numeric_data <- data[, sapply(data, is.numeric)]
# correlation_matrix <- cor(numeric_data)
# correlation_matrix

# home vs. away

# feet to polar coordinates
data <- data %>%
  select(-c(team_name))
data$loc_r <- sqrt((data$loc_x)^2 + (data$loc_y)^2)
data$loc_theta <- atan(data$loc_y/data$loc_x)
data$loc_theta[is.na(data$loc_theta)] <- pi/2

data$home <- as.numeric(grepl("vs.", data$matchup, fixed = TRUE))
data$away <- as.numeric(grepl("@", data$matchup, fixed = TRUE))
data$num_rings <- 0
data[data$game_num >= 311 & data$game_num <= 394,]$num_rings <- 1 
data[data$game_num >= 395 & data$game_num <= 493,]$num_rings <- 2 
data[data$game_num >= 494 & data$game_num <= 1116,]$num_rings <- 3 
data[data$game_num >= 1117 & data$game_num <= 1212,]$num_rings <- 4 
data[data$game_num >= 1213 & data$game_num <= 1559,]$num_rings <- 5
data$mvp <- ifelse(data$game_num >= 909 & data$game_num <= 990, 1, 0)
data$finals_mvp <- ifelse((data$game_num >= 1112 & data$game_num <= 1116) | 
                            (data$game_num >= 1206 & data$game_num <= 1212), 
                          1, 0)
data$postachilles <- ifelse(data$game_num > 1452, 1, 0)

data_train <- data %>%
  select(c("action_type", "shot_type", "shot_zone_area", "playoffs", "period", "time_remaining", "loc_r", "loc_theta", "home", "away", "lastminutes", "game_num", "first_team", "scoring_leader", "num_rings", "mvp", "finals_mvp", "postachilles", "shot_made_flag"))

data_train_final <- data_train %>%
  filter(!is.na(shot_made_flag))

data_test <- data %>%
  filter(is.na(shot_made_flag)) %>%
  select(c("action_type", "shot_type", "shot_zone_area", "playoffs", "period", "time_remaining", "loc_r", "loc_theta", "home", "away", "lastminutes", "game_num", "first_team", "scoring_leader", "num_rings", "mvp", "finals_mvp", "postachilles", "shot_made_flag", "shot_id"))

all_levels <- unique(c(data_train_final$action_type, data_test$action_type))
data_train_final$action_type <- factor(data_train_final$action_type, levels = all_levels)
data_test$action_type <- factor(data_test$action_type, levels = all_levels)

recipe <- recipe(shot_made_flag ~ ., data=data_train_final) %>%
  step_mutate_at(c(playoffs, period), fn=factor) %>%
  step_dummy(all_nominal_predictors())

prep <- prep(recipe)
bake <- bake(prep, new_data=data_train_final)

# rf_model <- rand_forest(mtry = tune(),
#                         min_n = tune(),
#                         trees=1000) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# kobe_wf <- workflow() %>%
#   add_recipe(recipe) %>%
#   add_model(rf_model)
# 
# # Set up grid of tuning values
# tuning_grid <- grid_regular(mtry(range=c(1,(ncol(data_train_final) - 1))),
#                             min_n(),
#                             levels = 5) ## L^2 total tuning possibilities
# 
# # Set up K-fold CV
# folds <- vfold_cv(data_train_final, v = 5, repeats=1)
# 
# CV_results <- kobe_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc))
# 
# # Find best tuning parameters
# bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and predict
# final_wf <- kobe_wf %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=data_train_final)
# 
# predictions <- final_wf %>%
#   predict(new_data = data_test, type="prob")
# 
# predictions$shot_made_flag <- predictions$.pred_1
# predictions$shot_id <- data_test$shot_id
# kobe_final <- predictions %>%
#   select(c(shot_id, shot_made_flag))
# 
# write.csv(kobe_final, "./rfclassification.csv", row.names = F)

## xgboost
boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(boost_model)

# Fit or Tune Model HERE
tuning_grid2 <- grid_regular(tree_depth(),
                             trees(),
                             learn_rate(),
                             levels = 5) ## L^2 total tuning possibilities

folds2 <- vfold_cv(data_train_final, v = 5, repeats=1)

CV_results2 <- boost_wf %>%
  tune_grid(resamples=folds2,
            grid=tuning_grid2,
            metrics=metric_set(roc_auc))

# Predict
bestTune2 <- CV_results2 %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf2 <- boost_wf %>%
  finalize_workflow(bestTune2) %>%
  fit(data=data_train_final)

predictions2 <- final_wf2 %>%
  predict(new_data = data_test, type="prob")

predictions2$shot_made_flag <- predictions2$.pred_1
predictions2$shot_id <- data_test$shot_id
xgboost_final <- predictions2 %>%
  select(c(shot_id, shot_made_flag))

write.csv(xgboost_final, "./xgboostclassification.csv", row.names = F)
