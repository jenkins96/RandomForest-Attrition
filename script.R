
# LIBRARIES ---------------------------------------------------------------
listOfPackages <-c("tidyverse","magrittr","tidymodels",
                   "ROCR","vip", "ranger","recipes","randomForest","caret")

for (i in listOfPackages){
  if(! i %in% installed.packages()){
    install.packages(i, dependencies = TRUE)
  }
  lapply(i, require, character.only = TRUE)
  rm(i)
}


# IMPORTING DATA ----------------------------------------------------------
data <- read_csv("dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
attrition <- data
attach(attrition)


# VIEWING DATA -------------------------------------------------------------
sapply(attrition, function(x) sum(is.na(x)))
summary(attrition)
str(attrition)


# DATA TRANSFORMATION -----------------------------------------------------
attrition %<>%
  select(-c(EmployeeCount, EmployeeNumber, NumCompaniesWorked, 
            Over18, PercentSalaryHike, PerformanceRating, 
            StandardHours, StockOptionLevel, 
            TrainingTimesLastYear, TotalWorkingYears, 
            YearsInCurrentRole, YearsSinceLastPromotion, 
            YearsWithCurrManager, DailyRate))
attrition$Attrition <- factor(attrition$Attrition, ordered = TRUE, 
                              levels = c("No", "Yes"))
attrition %<>% 
  mutate_if(is.character,as.factor)


# DATA PARTIONING ---------------------------------------------------------
set.seed(1)
attrition_split <- initial_split(attrition, strata = Attrition)
attrition_train <- training(attrition_split)
attrition_test <- testing(attrition_split)
attrition_split



# RANDOM FOREST -----------------------------------------------------------
modelo_rf <- randomForest(Attrition ~., data = attrition_train,
                          importance = TRUE, proximity = TRUE)
modelo_rf

# Feature Importance
varImpPlot(modelo_rf, main="Feature Importance")
feature_importance_filter <- importance(modelo_rf)
feature_importance_filter <- as.data.frame(feature_importance_filter) 
my_order <- order( feature_importance_filter$MeanDecreaseGini, decreasing = T)
feature_importance_filter[my_order,] %>%
  head()

# Error Rate per number of trees
plot(modelo_rf, main = "Error  Per Number Of Trees")

# Prediction
pred_rf <- predict(modelo_rf, attrition_test, type = "class")
my_table <- table( attrition_test$Attrition, pred_rf,  dnn = c("Actual", "Prediction")) 
my_table
round(prop.table(my_table, 1) * 100,2) # per row
round(prop.table(my_table, 2) * 100,2) # per column  

# Evaluation
TP <- my_table[2,2]
TN <- my_table[1,1]
FN <- my_table[2,1]
FP <- my_table[1,2]
# Accuracy
  # = TP + TN / TP+FP+FN+TN
sprintf("Accuracy: %f", sum(TP,TN)/sum(TP,TN,FN,FP))  

# Precision
  # = TP / TP + FP
precision <-  TP/ sum(TP,FP)
sprintf("Precision: %f", precision)

# Recall / Sensitivity
  # = TP/ TP + FN 
recall<- TP/(sum(TP, FN))
sprintf("Recall: %f", recall)

# SPECIFICITY
  # = TN / TN + FP 
specificity <- TN/(sum(TN,FP))
sprintf("Specificity: %f", specificity)

# F1
F1 <- 2 * precision * recall / (precision + recall)
sprintf("F1 score: %f", F1)

# ROC & AUC
probs <- predict(modelo_rf, attrition_test, type = "prob")
head(probs)
pred_for_ROCR <- prediction(probs[,2], attrition_test$Attrition)
perf <- performance(pred_for_ROCR, "tpr", "fpr")
plot(perf)
lines(par()$usr[1:2], par()$usr[3:4])
# AUC
auc.tmp <- performance(pred_for_ROCR,"auc")
auc <- as.numeric(auc.tmp@y.values)
auc

prob.cuts.1 <- data.frame(cut = perf@alpha.values[[1]], 
                          fpr = perf@x.values[[1]],
                          tpr = perf@y.values[[1]])
head(prob.cuts.1)
tail(prob.cuts.1)

head(prob.cuts.1[prob.cuts.1$tpr >= 0.7,])


# OPTIMIZING --------------------------------------------------------------

# Recipe
tree_recipe <- recipe(Attrition ~. , data = attrition_train)
tree_prep <- prep(tree_recipe)
juiced <- juice(tree_prep)
tune_spec <- rand_forest(
  mtry = tune(),
  trees = 501,
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")
# Workflow
tune_wf <- workflow() %>%
  add_recipe(tree_recipe) %>%
  add_model(tune_spec)
# Cross Validation
set.seed(2)
trees_fold <- vfold_cv(attrition_train, v = 4)
trees_fold  
