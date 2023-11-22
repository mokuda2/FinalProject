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
data <- data %>%
  select(-c(team_id, team_name, matchup))

view(data)
data_test <- data %>%
  filter(is.na(shot_made_flag))

recipe <- recipe(shot_made_flag ~ ., data=data) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_dummy(all_nominal_predictors())

prep <- prep(recipe)
bake <- bake(prep, new_data=data)

