# Read raw data
setwd("/Users/lancelin/Desktop/R_project")
df = read.csv("adult.csv")

# Rename the columns
colnames(df)[5] ="education_num"
colnames(df)[6] ="marital_status"
colnames(df)[11] ="capital_gain"
colnames(df)[12] ="capital_loss "
colnames(df)[13] ="hours_per_week"
colnames(df)[14] ="native_country"


# Deal with the values of the “?” sign (replace with N/A)
data_prep <- function(x) {
  if (x == '?') {
    return(NA)
  } else{
    return(x)
  }
}

# Apply Function data_prep to all columns
for (i in 1:ncol(df)) {
  df[, i] <- sapply(df[, i], data_prep)
}

# Remove all NA rows
df <- na.omit(df)

# Add new columns named over50k which is depended on column "income"
df$over50k <- as.factor(ifelse(df$income == ">50K", 1, 0))

# Convert categorical columns into factor data type
df$workclass <- as.factor(df$workclass)
df$education <- as.factor(df$education)
df$marital_status <- as.factor(df$marital_status)
df$relationship <- as.factor(df$relationship)
df$race <- as.factor(df$race)
df$sex <- as.factor(df$sex)
df$occupation <- as.factor(df$occupation)
df$native_country <- as.factor(df$native_country)
df$income <- as.factor(df$income)

# Check Correlation

## Grab only numeric columns
num.cols <- sapply(df, is.numeric)
## Filter to numeric columns for correlation
cor.data <- cor(df[, num.cols])
cor.data

# Split data into testing and training dataset with ratio = 0.8
library(caTools)
set.seed(1)
split = sample.split(df$over50k, SplitRatio = 0.8)
train = subset(df, split == TRUE)
test = subset(df, split == FALSE)


# Check IV value (Just keep variables with IV >= 0.3)
# https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html#id-75e7fa
# install.packages("Information")
library(Information)
train$over50k=as.numeric(train$over50k)-1
IV <- create_infotables(data=train, y="over50k", bins=10, parallel=FALSE)
train$over50k=as.factor(train$over50k)
IV_Value = data.frame(IV$Summary)
IV_Value

# Because education and education_num represent the same information, we pick education due to higher IV
# So, we pick relationship + marital_status + age + occupation + education + hours_per_week + capital_gain as our predicted variables
 
########################################
# Logistic Regression

# Fit Logistic Regression model
logistic <-
  glm(
    over50k ~ relationship + marital_status + age + occupation + education + hours_per_week + capital_gain,
    data = train,
    family = "binomial"(link = 'logit')
  )

## VIF check (if >= 4, keep higher one)
install.packages("car")
library(car)
vif(logistic)

## We decide drop marital_status

## Fit model again
logistic <-
  glm(
    over50k ~ relationship + age + occupation + education + hours_per_week + capital_gain,
    data = train,
    family = "binomial"(link = 'logit')
  )

## VIF check again
vif(logistic)

## Training data Accuracy
train_prob <- predict(logistic, newdata = train, type = 'response')
train_results <- ifelse(train_prob > 0.5, 1, 0)
train_confmatrix <-
  table(Actual_value = train$over50k,
        Predicted_value = train_results > 0.5)
train_confmatrix

## Training Accuracy
print('Logistic Regression Training Accuracy:')
(train_confmatrix[[1, 1]] + train_confmatrix[[2, 2]]) / sum(train_confmatrix)


## Testing data Accuracy
test_prob <- predict(logistic, newdata = test, type = 'response')
test_results <- ifelse(test_prob > 0.5, 1, 0)
test_confmatrix <-
  table(Actual_value = test$over50k,
        Predicted_value = test_results > 0.5)
test_confmatrix
## Testing Accuracy
print('Logistic Regression Testing Accuracy:')
(test_confmatrix[[1, 1]] + test_confmatrix[[2, 2]]) / sum(test_confmatrix)

########################################
# C5.0 Decision Tree
## install.packages("C50")
library("C50")

# Fit C5.0 Decision Tree model
predictors <-
  c(
    'relationship',
    'marital_status',
    'age',
    'occupation',
    'education',
    'hours_per_week',
    'capital_gain'
  )
model <- C5.0(x = train[, predictors], y = train$over50k)
summary(model)

## Training data Accuracy
train_results <- predict(model, train)
train_confmatrix <- table(train$over50k, train_results)
train_confmatrix
## Training Accuracy
print('C5.0 Training Accuracy:')
(train_confmatrix[[1, 1]] + train_confmatrix[[2, 2]]) / sum(train_confmatrix)


## Testing data Accuracy
testing_results <- predict(model, test)
test_confmatrix <- table(test$over50k, testing_results)
test_confmatrix
## Testing Accuracy
print('C5.0 Testing Accuracy:')
(test_confmatrix[[1, 1]] + test_confmatrix[[2, 2]]) / sum(test_confmatrix)

########################################
# SVM
install.packages('e1071',repos = 'http://cran.us.r-project.org')
library(e1071)

# Fit SVM model
model <- svm(over50k ~ relationship + age + occupation + education + hours_per_week + capital_gain, data=train)

## Training data Accuracy
predicted.values <- predict(model,train)
train_confmatrix <- table(train$over50k, predicted.values)
train_confmatrix
print('SVM training Accuracy:')
(train_confmatrix[[1, 1]] + train_confmatrix[[2, 2]]) / sum(train_confmatrix)

## Testing data Accuracy
predicted.values <- predict(model, test)
test_confmatrix <- table(test$over50k, predicted.values)
test_confmatrix
print('SVM Testing Accuracy:')
(test_confmatrix[[1, 1]] + test_confmatrix[[2, 2]]) / sum(test_confmatrix)

# We get the primary testing Acc for these three models:
## Logistic Testing Acc = 83.63%
## C5.0 Testing Acc = 85.26% (Best one)
## SVM  Testing Acc = 84.22%


######## Use K-fold Validation to compare three models ######## 

library(caret)
library(C50)
install.packages('svmLinear')

k <- 10
train_control <- trainControl(method = "cv", number = k)

# For logistic regression
k_fold_log_model <- train(over50k ~ relationship + age + occupation + education + hours_per_week + capital_gain, data = df, method = "glm", family = binomial(link = 'logit'), trControl = train_control)
k_fold_log_model
# Acc = 84.33%

# For C5.0 Decision tree
k_fold_c50_model <- train(over50k ~ relationship + age + occupation + education + hours_per_week + capital_gain, data = df, method = "C5.0", trControl = train_control)
k_fold_c50_model
# Acc = 85.72%

# For SVM Decision tree
k_fold_svm_model <- train(over50k ~ relationship + age + occupation + education + hours_per_week + capital_gain, data = df, method = "svmLinear", trControl = train_control)
k_fold_svm_model
# Acc = 83.48%

# By K-fold validation, we know that C5.0 Decision tree has the greatest acc!


