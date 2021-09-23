library(faraway)
library(readr)
library(mlbench)
library(corrplot)
dreadtest <- read_csv("dreaddit-test.csv")
View(dreadtest)
dim(dreadtest)
head(dreadtest, n = 40)
sapply(dreadtest, class)
summary(dreadtest)
sum(is.na(dreadtest))

#Dig deep into the dataset#
library(car)
scatterplotMatrix(dreadtest[7:10])
cor(dreadtest[,7:10])
hist(dreadtest$confidence, main = names(dreadtest$confidence))
hist(dreadtest$social_timestamp, main = names(dreadtest$social_timestamp))
hist(dreadtest$social_karma, main = names(dreadtest$social_karma))
hist(dreadtest$syntax_ari, main = names(dreadtest$syntax_ari))
plot(density(dreadtest$confidence), main = names(dreadtest$confidence))
plot(density(dreadtest$social_timestamp), main = names(dreadtest$social_timestamp))
plot(density(dreadtest$social_karma), main = names(dreadtest$social_karma))
plot(density(dreadtest$syntax_ari), main = names(dreadtest$syntax_ari))
boxplot(dreadtest$confidence, main = names(dreadtest$confidence))
boxplot(dreadtest$social_timestamp, main = names(dreadtest$social_timestamp))
boxplot(dreadtest$social_karma, main = names(dreadtest$social_karma))
boxplot(dreadtest$syntax_ari, main = names(dreadtest$syntax_ari))

#Running the algorithms using 5-fold cross validation#
library(caret)
trainControl = trainControl(method = "repeatedcv", number = 5, repeats = 2)
metric <- "RMSE"

#LM#
set.seed(3)
fit.lm <- train(confidence ~ ., data = dreadtest, method = "lm", metric = metric, preProc = c("center","scale"), trControl = trainControl)

#GLM#
set.seed(3)
fit.glm <- train(confidence ~ ., data = dreadtest, method = "glm", metric = metric, preProc = c("center","scale"), trControl = trainControl)

#GLMNET#
set.seed(3)
fit.glmnet <- train(confidence ~ ., data = dreadtest, method = "glmnet", metric = metric, preProc = c("center","scale"), trControl = trainControl)

#SVM#
set.seed(3)
fit.svm <- train(confidence ~ ., data = dreadtest, method = "svmRadial", metric = metric, preProc = c("center","scale"), trControl = trainControl)

#RF#
set.seed(3)
fit.rf <- train(confidence ~ ., data = dreadtest, method = "rf", metric = metric, preProc = c("center","scale"), trControl = trainControl)

#KNN#
set.seed(3)
fit.knn <- train(confidence ~ ., data = dreadtest, method = "knn", metric = metric, preProc = c("center","scale"), trControl = trainControl)

#Comparing the algorithms#
results <- resamples(list(LM = fit.lm, GLM = fit.glm, GLMNET = fit.glmnet, SVM = fit.svm, RF = fit.rf, KNN = fit.knn))
summary(results)
dotplot(results)

#Remove the correlated attributes#
#Find the attributes that are highly correlated#
set.seed(3)
cutoff <- .7
correlations <- cor(dreadtest[,7:10])
highlyCorrelated <- findCorrelation(correlations, cutoff = cutoff)
for (value in highlyCorrelated) {
  print(names(dreadtest)[value])
}

#Create a new dataset without highly correlated features#
datasetFeatures <- dreadtest[, -highlyCorrelated]
dim(datasetFeatures)
#No attributes that are highly correlated#

#Running the algorithms again using 5-fold cross validation & Box-Cox#
trainControl = trainControl(method = "repeatedcv", number = 5, repeats = 2)
metric <- "RMSE"

#LM#
set.seed(3)
fit.lm <- train(confidence ~ ., data = dreadtest, method = "lm", metric = metric, preProc = c("center","scale","BoxCox"), trControl = trainControl)

#GLM#
set.seed(3)
fit.glm <- train(confidence ~ ., data = dreadtest, method = "glm", metric = metric, preProc = c("center","scale", "BoxCox"), trControl = trainControl)

#GLMNET#
set.seed(3)
fit.glmnet <- train(confidence ~ ., data = dreadtest, method = "glmnet", metric = metric, preProc = c("center","scale","BoxCox"), trControl = trainControl)

#SVM#
set.seed(3)
fit.svm <- train(confidence ~ ., data = dreadtest, method = "svmRadial", metric = metric, preProc = c("center","scale","BoxCox"), trControl = trainControl)

#RF#
set.seed(3)
fit.rf <- train(confidence ~ ., data = dreadtest, method = "rf", metric = metric, preProc = c("center","scale","BoxCox"), trControl = trainControl)

#KNN#
set.seed(3)
fit.knn <- train(confidence ~ ., data = dreadtest, method = "knn", metric = metric, preProc = c("center","scale","BoxCox"), trControl = trainControl)

#Comparing the algorithms#
results <- resamples(list(LM = fit.lm, GLM = fit.glm, GLMNET = fit.glmnet, SVM = fit.svm, RF = fit.rf, KNN = fit.knn))
summary(results)
dotplot(results)

print(fit.svm)
print(fit.lm)
print(fit.glm)
print(fit.glmnet)
print(fit.rf)
print(fit.knn)

#Tune SVM sigma & C parameters#
set.seed(3)
grid <- expand.grid(.sigma = c(.025,.05,.1,.15), .C = seq(1,10, by = 1))
fit.svm <- train(confidence ~ ., data = dreadtest, method = "svmRadial", metric = metric, tuneGrid = grid, preProc = c("BoxCox"), trControl = trainControl)
print(fit.svm)
plot(fit.svm)

#Stochastic Gradient Boosting (GBM)#
set.seed(3)
fit.gbm <- train(confidence ~ ., data = dreadtest, method = "gbm", metric = metric, preProc = c("BoxCox"), trControl = trainControl, verbose = FALSE)

print(fit.gbm)
plot(fit.gbm)

#SVM wins#

#Let's try for Cubist to experiment further#
#Setting up the Final Model#
set.seed(3)
validationIndex <- createDataPartition(dreadtest$confidence, p = .8, list = FALSE)
validation <- dreadtest[-validationIndex,]
dataset <- dreadtest[validationIndex,]
par(mfrow = c(2,7))
for (i in 1:13) {
  boxplot(dataset[,i], main = names(dataset)[i])
}
x <- dataset[,1:13]
y <- dataset[,14]
preprocessParams <- preProcess(x)
transX <- sample(1:nrow(dataset), floor(.8*nrow(dataset)))
predictors <- c("confidence", "social_karma", "social_timestamp","syntax_ari")
transXpred <- dataset[transX, predictors]
transXresp <- dataset$confidence[transX]

#Train the final model#
library(Cubist)
finalModel <- cubist(x = transXpred, y = transXresp)
summary(finalModel)

#Transform the validation dataset#
set.seed(3)
predictions <- predict(finalModel, transXpred)

#calculate the RMSE#
rmse <- sqrt(mean((predictions - transXresp)^2))
r2 <- cor(predictions, transXresp)^2
print(rmse)
print(r2)

#Citations#
citation(package = "faraway")
citation(package = "readr")
citation(package = "mlbench")
citation(package = "corrplot")
citation(package = "car")
citation(package = "caret")
citation(package = "Cubist")
