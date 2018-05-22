
# Title: "Practical Machine Learning Project"
# Author: "Pradeep Pal"
# Date: "May 22, 2018"

#I. Overview

The main goal of the project is to predict the manner in which 6 participants performed some exercise as described below. This is the “classe” variable in the training set. The machine learning algorithm described here is applied to the 20 test cases available in the test data and the predictions are submitted in appropriate format to the Course Project Prediction Quiz for automated grading.

#II. Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3xsbS5bVX

#III. Data Loading and Exploratory Analysis

##a) Dataset Overview
The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from http://groupware.les.inf.puc-rio.br/har. 

Full source:

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. “Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human ’13)”. Stuttgart, Germany: ACM SIGCHI, 2013.

My special thanks to the above mentioned authors for being so generous in allowing their data to be used for this kind of assignment.

A short description of the datasets content from the authors’ website:

“Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg)."

##b) Environment Preparation
  Here is the R libraries that are necessary for the complete analysis.

  *   library(knitr)
  *   library(caret)
  *   library(rpart)
  *   library(rpart.plot)
  *   library(rattle)
  *   library(randomForest)
  *   library(corrplot)
  *   set.seed(12345)
  
##c) Data Loading and Cleaning
The next step is loading the dataset from the URL provided above. The training dataset is then partinioned in 2 to create a Training set (70% of the data) for the modeling process and a Test set (with the remaining 30%) for the validations. The testing dataset is not changed and will only be used for the quiz results generation.



```{r}
#Upload the R libraries

library(knitr)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)
library(tidyr)
library(xgboost)
library(Rtsne)
library(stats)
library(ggplot2)
library(data.table)
library(curl)
set.seed(12345)
```

```{r}
# Setting path of training and testing dataset.
train_data = read.csv("C:/Users/Pradeep/Documents/Coursera/ML_Project/pml-training.csv")
test_data = read.csv("C:/Users/Pradeep/Documents/Coursera/ML_Project/pml-testing.csv")

# create a partition with the training dataset 
inTrain  <- createDataPartition(train_data$classe, p=0.7, list=FALSE)
TrainSet <- train_data[inTrain, ]
TestSet  <- train_data[-inTrain, ]
dim(TrainSet)
dim(TestSet)

names(TrainSet)
```
```{r}
# remove variables with Nearly Zero Variance
NZV <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[, -NZV]
TestSet  <- TestSet[, -NZV]
dim(TrainSet)
dim(TestSet)

# remove variables that are mostly NA
AllNA    <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
TrainSet <- TrainSet[, AllNA==FALSE]
TestSet  <- TestSet[, AllNA==FALSE]
dim(TrainSet)
dim(TestSet)

# remove identification only variables (columns 1 to 5)
TrainSet <- TrainSet[, -(1:5)]
TestSet  <- TestSet[, -(1:5)]
dim(TrainSet)
dim(TestSet)

# correlation analysis
corMatrix <- cor(TrainSet[, -54])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

```{r}
# model fit
set.seed(12345)
modFitA1 <- rpart(classe ~ ., data=TrainSet, method="class")
fancyRpartPlot(modFitA1)

predictionsA1 <- predict(modFitA1, TestSet, type = "class")
cmtree <- confusionMatrix(predictionsA1, TestSet$classe)
cmtree

plot(cmtree$table, col = cmtree$byClass, main = paste("Decision Tree Confusion Matrix: Accuracy =", round(cmtree$overall['Accuracy'], 4)))
```

```{r}
#Prediction with Random Forests
set.seed(12345)
modFitB1 <- randomForest(classe ~ ., data=TrainSet)
predictionB1 <- predict(modFitB1, TestSet, type = "class")
cmrf <- confusionMatrix(predictionB1, TestSet$classe)
cmrf

plot(cmrf$table, col = cmtree$byClass, main = paste("Random Forest Confusion Matrix: Accuracy =", round(cmrf$overall['Accuracy'], 4)))

```

#IV. Applying the Selected Model to the Test Data
The accuracy of the 2 regression modeling methods above are:

a. Decision Tree : 0.7368
b. Random Forest : 0.9952

#Predicting Results on the Test Data
Random Forests gave an Accuracy in the myTesting dataset of 99.52%, which was more accurate that what I got from the Decision Trees . The expected out-of-sample error is 100-99.52 = 0.48%.

In that case, the Random Forest model will be applied to predict the 20 quiz results (testing dataset) as shown below.


