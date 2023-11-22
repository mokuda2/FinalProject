library(tidyverse)
library(vroom)
library(tidymodels)
library(embed)
library(discrim)
library(naivebayes)
library(kknn)
library(kernlab)
library(lightgbm)
library(themis)
library(bonsai)

data <- vroom("./STAT 348/FinalProject/data.csv")
data$shot_made_flag <- factor(data$shot_made_flag)
data$time_remaining <- 48 - data$period + data$minutes_remaining

data_train <- data[c("action_type", "shot_type", "shot_zone_area", "playoffs", "period", "time_remaining", "shot_made_flag")]

data_train_final <- data_train %>%
  filter(!is.na(shot_made_flag))
  
data_test <- data %>%
  filter(is.na(shot_made_flag)) %>%
  select(c("action_type", "shot_type", "shot_zone_area", "playoffs", "period", "time_remaining", "shot_made_flag", "shot_id"))

all_levels <- unique(c(data_train$action_type, data_test$action_type))
data_train_final$action_type <- factor(data_train_final$action_type, levels = all_levels)
data_test$action_type <- factor(data_test$action_type, levels = all_levels)

recipe <- recipe(shot_made_flag ~ ., data=data_train_final) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_dummy(all_nominal_predictors())

prep <- prep(recipe)
bake <- bake(prep, new_data=data_train_final)

rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

kobe_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(rf_model)

# Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range=c(1,(ncol(data_train_final) - 1))),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities

# Set up K-fold CV
folds <- vfold_cv(data_train_final, v = 5, repeats=1)

CV_results <- kobe_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

# Find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf <- kobe_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=data_train_final)

predictions <- final_wf %>%
  predict(new_data = data_test, type="prob")

predictions$shot_made_flag <- predictions$.pred_1
predictions$shot_id <- data_test$shot_id
kobe_final <- predictions %>%
  select(c(shot_id, shot_made_flag))

write.csv(kobe_final, "./STAT\ 348/FinalProject/rfclassification.csv", row.names = F)

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

write.csv(xgboost_final, "./STAT\ 348/FinalProject/xgboostclassification.csv", row.names = F)

