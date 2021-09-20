library(faraway)
library(readr)
library(mlbench)
library(corrplot)
dreadtrain <- read_csv("dreaddit-train.csv")
View(dreadtrain)
dim(dreadtrain)
head(dreadtrain, n = 40)
sapply(dreadtrain, class)
summary(dreadtrain)
sum(is.na(dreadtrain))

library(car)
scatterplotMatrix(dreadtrain[7:10])
correlations <- cor(dreadtrain[,7:10])

#Linear Model#
g <- lm(confidence ~ ., data = dreadtrain)
summary(g)
p <- predict(g, type = "response")

#General Linear Model#
g2 <- glm(confidence ~ ., data = dreadtrain, family = poisson())
summary(g2)
p2 <- predict(g2, type = "response")

library(klaR)
library(psych)
library(ggord)
library(devtools)
library(caret)
library(MASS)

#Linear Discriminant Analysis#
g3 <- lda(confidence ~ ., data = dreadtrain)
summary(g3)
p3 <- predict(g3, type = "response")

#Classification & Regression Trees#
library(rpart)
g4 <- rpart(confidence ~ ., data = dreadtrain, method = "anova", control = rpart.control(minsplit = 10, cp = .01))
summary(g4)
p4 <- predict(g4, type = "response")

#Random Forest#
library(randomForest)
g5 <- randomForest(confidence ~ ., data = dreadtrain)
summary(g5)
p5 <- predict(g5, type = "response")

#k-Nearest Neighbors#
g6 <- knnreg(confidence ~ ., data = dreadtrain)
summary(g6)
p6 <- predict(g6, type ="response")

#Support Vector Machines w/a radial kernel#
library(e1071)
g7 <- svm(confidence ~ ., data = dreadtrain)
summary(g7)
p7 <- predict(g7, type = "response")

#Compare the algorithms#
rmse <- function(x,y) sqrt(mean(x-y)^2)
summary(p)
summary(p2)
summary(p3)
summary(p4)
summary(p5)
summary(p6)
summary(p7) #SVM best fit#

rmse(g$fitted.values,dreadtrain$confidence)
rmse(g2$fitted.values,dreadtrain$confidence)
rmse(g3$fitted.values,dreadtrain$confidence)
rmse(g4$fitted.values,dreadtrain$confidence)
rmse(g5$fitted.values,dreadtrain$confidence)
rmse(g6$fitted.values,dreadtrain$confidence)
rmse(g7$fitted.values,dreadtrain$confidence)

print(p7)
plot(fitted(g7), residuals(g7), xlab = "Fitted", ylab = "Residuals")

#Citations#
citation(package = "faraway")
citation(package = "readr")
citation(package = "mlbench")
citation(package = "corrplot")
citation(package = "klaR")
citation(package = "psych")
citation(package = "ggord")
citation(package = "devtools")
citation(package = "caret")
citation(package = "MASS")
citation(package = "rpart")
citation(package = "randomForest")
citation(package = "e1071")