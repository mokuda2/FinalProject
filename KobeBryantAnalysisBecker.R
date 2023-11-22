library(tidyverse)
library(vroom)
library(tidymodels)
library(embed)
library(discrim)
library(naivebayes)
library(kknn)
library(kernlab)
library(themis)

data <- vroom("./data.csv")
data$shot_made_flag <- factor(data$shot_made_flag)
data_clean <- data %>%
  select(-c(team_id, team_name, matchup, season, game_date, playoffs))
data_clean_NA <- data_clean %>%
  filter(!is.na(shot_made_flag))

view(data_clean)
data_test <- data_clean %>%
  filter(is.na(shot_made_flag))

recipe <- recipe(shot_made_flag ~ ., data=data_clean_NA) %>%
  step_mutate_at(all_nominal_predictors(), fn=factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(shot_made_flag))

prep <- prep(recipe)
bake <- bake(prep, new_data=data_clean_NA)

rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Create a workflow with model & recipe
# target_encoding_amazon_recipe <- recipe(ACTION~., data=amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn=factor) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION))
# prep <- prep(target_encoding_amazon_recipe)
# baked_train <- bake(prep, new_data = amazon_train)

kobe_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(rf_model)

# Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range=c(1,(ncol(data_clean_NA) - 1))),
                            min_n(),
                            levels = 3) ## L^2 total tuning possibilities

# Set up K-fold CV
folds <- vfold_cv(data_clean_NA, v = 3, repeats=1)

CV_results <- kobe_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf <- kobe_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=data_clean_NA)

predictions <- final_wf %>%
  predict(new_data = data_test, type="prob")

predictions$shot_made_flag <- predictions$.pred_0
predictions$shot_id <- data_test$shot_id
kobe_final <- predictions %>%
  select(c(shot_id, shot_made_flag))

write.csv(kobe_final, "./STAT\ 348/FinalProject/rfclassification.csv", row.names = F)

