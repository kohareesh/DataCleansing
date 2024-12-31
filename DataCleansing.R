install.packages("titanic")
install.packages("RANN")

library(titanic)
library(vtreat)
library(caret)
library(randomForest)

data <- titanic::titanic_train

# Convert Survived to a factor
data$Survived <- as.factor(data$Survived)

# Define target and variable names
target <- "Survived"
vars <- setdiff(names(data), c(target, "PassengerId", "Name", "Ticket", "Cabin"))

# Prepare vtreat design
treatment_plan <- vtreat::designTreatmentsC(
  dframe = data,
  varlist = vars,
  outcomename = target,
  outcometarget = 1
)

# Apply treatment
treated_data <- vtreat::prepare(treatment_plan, data)

# Summary of results
vtreat_summary <- treatment_plan$scoreFrame


# Define preprocessing parameters
preprocess_params <- preProcess(
  data[, vars],
  method = c("center", "scale", "knnImpute")
)

# Apply preprocessing using caret
caret_data <- predict(preprocess_params, data[, vars])



# Train the random forest model with vtreat data
rf_vtreat <- randomForest(Survived ~ ., data = treated_data)

# Predictions for vtreat
predicted_vtreat <- factor(predict(rf_vtreat, treated_data), levels = levels(data$Survived))
true_labels_vtreat <- factor(data$Survived, levels = levels(data$Survived))

# Confusion matrix for vtreat
rf_vtreat_perf <- confusionMatrix(predicted_vtreat, true_labels_vtreat)

# Train the random forest model with caret data
rf_caret <- randomForest(Survived ~ ., data = cbind(caret_data, Survived = data$Survived))

# Predictions for caret
predicted_caret <- factor(predict(rf_caret, caret_data), levels = levels(data$Survived))
true_labels_caret <- factor(data$Survived, levels = levels(data$Survived))

# Confusion matrix for caret
rf_caret_perf <- confusionMatrix(predicted_caret, true_labels_caret)

# Print results
print("Performance for vtreat model:")
print(rf_vtreat_perf)

print("Performance for caret model:")
print(rf_caret_perf)
