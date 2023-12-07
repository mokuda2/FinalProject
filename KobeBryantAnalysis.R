library(tidyverse)
library(vroom)
library(tidymodels)
library(embed)
library(discrim)
library(naivebayes)
library(kknn)
library(kernlab)
library(parsnip)
library(themis)
library(bonsai)

kobe <- vroom("./STAT 348/FinalProject/data.csv")
dist <- sqrt((kobe$loc_x/10)^2 + (kobe$loc_y/10)^2) 
kobe$shot_distance <- dist
#Creating angle column 
loc_x_zero <- kobe$loc_x == 0
kobe['angle'] <- rep(0,nrow(kobe))
kobe$angle[!loc_x_zero] <- atan(kobe$loc_y[!loc_x_zero] / kobe$loc_x[!loc_x_zero])
kobe$angle[loc_x_zero] <- pi / 2
# Create one time variable 
kobe$time_remaining = (kobe$minutes_remaining*60)+kobe$seconds_remaining
# Home and Away
kobe$matchup = ifelse(str_detect(kobe$matchup, 'vs.'), 'Home', 'Away')
# Season
kobe['season'] <- substr(str_split_fixed(kobe$season, '-',2)[,2],2,2)
### period into a factor
kobe$period <- as.factor(kobe$period)
# delete columns
kobe <- kobe %>%
  select(-c('shot_id', 'team_id', 'team_name', 'shot_zone_range', 'lon', 'lat', 
            'seconds_remaining', 'minutes_remaining', 'game_event_id', 
            'game_id', 'game_date','shot_zone_area',
            'shot_zone_basic', 'loc_x', 'loc_y'))
# Train
train <- kobe %>%
  filter(!is.na(shot_made_flag))
# Test 
test <- kobe %>% 
  filter(is.na(shot_made_flag))
## Make the response variable into a factor 
train$shot_made_flag <- as.factor(train$shot_made_flag)
recipe <- recipe(shot_made_flag ~ ., data = train) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors())

prep <- prep(recipe)
bake <- bake(prep, new_data=train)

## random forest
rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees=800) %>%
  set_engine("ranger") %>%
  set_mode("classification")

kobe_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(rf_model)

# Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range=c(1,(ncol(train) - 1))),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities

# Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats=1)

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
  fit(data=train)

predictions <- final_wf %>%
  predict(new_data = test, type="prob")

predictions$shot_made_flag <- predictions$.pred_1
predictions$shot_id <- test$shot_id
kobe_final <- predictions %>%
  select(c(shot_id, shot_made_flag))

write.csv(kobe_final, "./STAT\ 348/FinalProject/rfclassification.csv", row.names = F)

## knn
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(knn_model)

# Fit or Tune Model HERE
tuning_grid3 <- grid_regular(neighbors(),
                            levels = 5) ## L^2 total tuning possibilities

folds3 <- vfold_cv(train, v = 5, repeats=1)

CV_results3 <- knn_wf %>%
  tune_grid(resamples=folds3,
            grid=tuning_grid3,
            metrics=metric_set(roc_auc))

# Predict
bestTune3 <- CV_results3 %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf3 <- knn_wf %>%
  finalize_workflow(bestTune3) %>%
  fit(data=train)

predictions3 <- final_wf3 %>%
  predict(new_data = test, type="prob")

predictions3$shot_made_flag <- predictions3$.pred_1
predictions3$shot_id <- test$shot_id
knn_final <- predictions3 %>%
  select(c(shot_id, shot_made_flag))

write.csv(knn_final, "./STAT\ 348/FinalProject/knn.csv", row.names = F)

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

folds2 <- vfold_cv(train, v = 5, repeats=1)

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
  fit(data=train)

predictions2 <- final_wf2 %>%
  predict(new_data = test, type="prob")

predictions2$shot_made_flag <- predictions2$.pred_1
predictions2$shot_id <- test$shot_id
xgboost_final <- predictions2 %>%
  select(c(shot_id, shot_made_flag))

write.csv(xgboost_final, "./STAT\ 348/FinalProject/xgboostclassification.csv", row.names = F)

