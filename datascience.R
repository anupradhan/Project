rm(list = ls())
setwd("~/Documents/project")

if (!file.exists("data")) dir.create("data")

if (!file.exists("data/winequality-red.csv")) {
  fileURL <- "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
  download.file(fileURL, destfile = "data/winequality-red.csv", method="curl")
  list.files("data")
}

# Reading input file 
winedata <- read.table("data/winequality-red.csv", header=TRUE, sep = ";", dec=".")

# To simply classification task, winequality > 3 is classified as 1 (good) and < 3 is classifed as 0 (bad)
winedata$quality <- ifelse(winedata$quality < 6,0,1)
set.seed(123)

#sampling 2/3 of the input data
samp <- sample(1:nrow(winedata), as.integer(nrow(winedata)*0.66))
X = winedata[,1:11]
Y = winedata[,12]

# decision tree
library(rpart)
wine.tree.infogain <- rpart(quality ~ ., data=winedata, parms = list(split="information"), cp=0.002, method="class")
wine.tree.gini <- rpart(quality ~ ., data=winedata, parms = list(split="gini"), cp=0.002, method="class")

#prunning tree
wine.tree2.infogain <- prune(wine.tree.infogain, cp=0.01)
wine.tree2.gini <- prune(wine.tree.gini, cp=0.01)

#plotting prunned trees for better visualization
plot(wine.tree2.infogain, uniform=T)
text(wine.tree2.infogain, digits=2)

plot(wine.tree2.gini, uniform=T)
text(wine.tree2.gini, digits=2)

#cross validation error plot
plot(wine.tree.infogain$cptable[,2],0.46529*wine.tree.infogain$cptable[,4], type="l",col="red", lwd=5, xlab="Number of Splits", ylab="10-fold Cross Validation Error")
lines(wine.tree.gini$cptable[,2],0.46529*wine.tree.gini$cptable[,4], type="l", col="blue", pch=4, lty = 2, lwd=5, xlab="Number of Splits", ylab="10-fold Cross Validation Error")
legend("topright", inset=.05, c("Information Gain","Gini"), col=c("red","blue"), lty = c(0, 2), lwd = c(5, 5), pch = c(3, 4), horiz=FALSE)

# support vector machine algorithm
library(class)
library(e1071)
wine.svm.linear.train <- svm(quality~., data= winedata[samp,],scale=TRUE, type="C", kernel="linear", probability=TRUE)
wine.svm.linear.predict <- predict(wine.svm.linear.train, X[-samp,], decision.values=FALSE, probability=FALSE)
svm.correct <- table(wine.svm.linear.predict == as.factor(Y[-samp]))["TRUE"]
svm.error <- 1 - (svm.correct/length(Y[-samp]))
print(svm.error)

# Adaboost algorithm
library(ada)
train <- winedata[samp,]
test <- winedata[-samp]
wine.adaboost.train <- ada(quality~.,data=train,type="discrete",loss="exponential", nu=1,iter=200)
wine.adaboost.test <- addtest(wine.adaboost.train, X[-samp,], Y[-samp])
plot(wine.adaboost.test, TRUE, TRUE)
varplot(wine.adaboost.test)

#knn algorithm
library(class)
n <- 30
knn.error <- rep(NA,n)
for (i in 1:n){
  wine.knn <- knn(scale(X[samp,]), scale(X[-samp,]), as.factor(Y[samp]), k=i, prob=TRUE)
  incorrect <- table(wine.knn == as.factor(Y[-samp]))["FALSE"]
  knn.error[i] <- (incorrect/length(Y[-samp]))
}
plot(1:n, knn.error, type="l",col="red", lwd=5, xlab="Different Values of K", ylab="Test Error Rate")
print(knn.error)  

#neural network
library(nnet)
#training neural network
wine.NN.train <- nnet(scale(X[samp,]), Y[samp], size = 5, rang = 0.2, decay = 5e-4, maxit = 50, HESS = FALSE, method="class")
# testing neural network
wine.NN.test <- round(predict(wine.NN.train, scale(X[-samp,])))
#classification error
NN.error <- table(wine.NN.test == as.factor(Y[-samp]))["FALSE"]/length(Y[-samp])
print(NN.error)