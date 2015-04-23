# Predicting exercice quality through accelerometers data

## Introduction

In this project we will analyse the accelerometers measurements on the belt, forearm, arm, and dumbell of 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways.
We will try to fit a model to predict activity from measurements, using cross validation on a training dataset.
Then, calculating the out of sample error comparing real and predicted activity will give us the model accuracy.

## Exploratory data analysis

The training dataset is loaded from the "pml-training.csv" file.


```r
# Cleaning the environment
rm(list = ls(all = TRUE))

# Setting the working directory
setwd("E:\\BIG DATA\\Coursera - Data Scientist\\08 - Practical Machine Learning\\PA")

# Initialising the data file name containing the data (it must be in the working directory)
trainingFileName = "pml-training.csv"

# Reading the data and putting it in a dataframe
trainingdataset = read.csv(trainingFileName, header = TRUE, sep = ",", stringsAsFactors = FALSE)
```

This dataset has to be cleaned. Indeed, the first 8 columns are metadata, they are not needed in the model.
Moreover, a few measurements have only NA values or only empty values.


```r
# Function to calculate data percentage in each column
countPercentData = function(v) {
	count = 0
	for (i in 1:length(v)) {
		if (is.na(v[i]) | v[i] == "") {
			count = count + 1
		}
	}
	100 - count / length(v) *100
}

# Getting data percentage in each column
dataPercentColumn = apply(trainingdataset, 2, function(v) countPercentData(v))

# Removing colmns with less than 100% data and with metadata
columnNames = names((dataPercentColumn[dataPercentColumn == 100]))
columnNames = columnNames[8:length(columnNames)]
```

We will only conserve columns with all values filled.


```r
# Selecting columns in the training dataset
trainingdataset = trainingdataset[, columnNames]

# Setting the outcome as a factor
trainingdataset$classe = factor(trainingdataset$classe)
```

## Model analysis

We will use the caret package to perform cross validation. From the training dataset, we will create training (60%) and testing (40%) partitions.


```r
# Loading the needed libraries
library(caret)
library(randomForest)

# Creating training and testing partitions (cross validation)
inTrain = createDataPartition(y = trainingdataset$classe, p = 0.6, list = FALSE)
training = trainingdataset[inTrain, ]
testing  = trainingdataset[-inTrain, ]
```

We will use the training partition to fit a random forest model with the "classe" variable as outcome and all other variables as predictors.


```r
# Fitting a random forest model on the training partition
modFit = randomForest(classe ~ ., data = training)
```

We will predict the outcome with the model on the testing partition.


```r
# Applying the model on the testing partition
predictions = predict(modFit, testing)
```

Printing a confusion matrix between real and predicted data will give us out of sample error.


```r
# Estimating the model accuracy comparing real and predicted data
confusionMatrix(predictions, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    8    0    0    0
##          B    1 1505   23    0    0
##          C    0    5 1345   10    0
##          D    0    0    0 1276   14
##          E    0    0    0    0 1428
## 
## Overall Statistics
##                                        
##                Accuracy : 0.9922       
##                  95% CI : (0.99, 0.994)
##     No Information Rate : 0.2845       
##     P-Value [Acc > NIR] : < 2.2e-16    
##                                        
##                   Kappa : 0.9902       
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9914   0.9832   0.9922   0.9903
## Specificity            0.9986   0.9962   0.9977   0.9979   1.0000
## Pos Pred Value         0.9964   0.9843   0.9890   0.9891   1.0000
## Neg Pred Value         0.9998   0.9979   0.9965   0.9985   0.9978
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1918   0.1714   0.1626   0.1820
## Detection Prevalence   0.2854   0.1949   0.1733   0.1644   0.1820
## Balanced Accuracy      0.9991   0.9938   0.9904   0.9950   0.9951
```

## Conclusion
A random forest model is particularly accurate on this case (> 99%).
We can conclude that activity quality can be accurately predicted from accelerometers measurements.


