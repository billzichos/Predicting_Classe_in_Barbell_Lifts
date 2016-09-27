# Predicting Classe in Barbell Lifts
Bill Zichos  
September 25, 2016  

I developed a model that with >99% accuracy, predicts the classe for different barbell lifts.  The best model used gradient boosting and 40 predictors including some features engineered from the date.  The rest of this document is organized in the following sections.

- Setup

- Analysis

- Model and Feature Selection

- Prediction

#Setup


```r
library("caret")
library("Metrics")
library("e1071")
library("lubridate")

setwd("~/Coursera-Data-Science/08 Practical Machine Learning")

set.seed(10)
```

##Gather the data.


```r
train <- read.csv("pml-training.csv", stringsAsFactors = FALSE)
test <- read.csv("pml-testing.csv", stringsAsFactors = FALSE)

inSplit <- createDataPartition(train$classe, p = .7, list = FALSE)
```

##Prepare the data.

*X* is a unique identifier, so we want to make sure it is excluded from model fitting.


```r
exclude.predictors <- which(names(train)=="X")
```

Drop predictors that have more than 19000 NULLS.  This is way too sparse to add any value.


```r
toomanynulls <- which(sapply(train, function(x) {length(x[is.na(x)])}) > 19000)

train <- train[,-toomanynulls]
test <- test[,-toomanynulls]
```

*cvtd_timestamp* needs to be formatted as a datetime.


```r
train$cvtd_timestamp <- as.POSIXct(train$cvtd_timestamp, format = "%d/%m/%Y %R")
train$cvtd_timestamp.year <- year(train$cvtd_timestamp)
train$cvtd_timestamp.month <- month(train$cvtd_timestamp)
train$cvtd_timestamp.day <- day(train$cvtd_timestamp)
train$cvtd_timestamp.hour <- hour(train$cvtd_timestamp)
train$cvtd_timestamp.minute <- minute(train$cvtd_timestamp)
train$cvtd_timestamp <- NULL

test$cvtd_timestamp <- as.POSIXct(test$cvtd_timestamp, format = "%d/%m/%Y %R")
test$cvtd_timestamp.year <- year(test$cvtd_timestamp)
test$cvtd_timestamp.month <- month(test$cvtd_timestamp)
test$cvtd_timestamp.day <- day(test$cvtd_timestamp)
test$cvtd_timestamp.hour <- hour(test$cvtd_timestamp)
test$cvtd_timestamp.minute <- minute(test$cvtd_timestamp)
test$cvtd_timestamp <- NULL
```

Center and scale the numeric predictors.


```r
train$roll_belt <- scale(train$roll_belt, center = TRUE, scale = TRUE)
test$roll_belt <- scale(test$roll_belt, center = TRUE, scale = TRUE)
train$pitch_belt <- scale(train$pitch_belt, center = TRUE, scale = TRUE)
test$pitch_belt <- scale(test$pitch_belt, center = TRUE, scale = TRUE)
train$yaw_belt <- scale(train$yaw_belt, center = TRUE, scale = TRUE)
test$yaw_belt <- scale(test$yaw_belt, center = TRUE, scale = TRUE)
train$gyros_belt_x <- scale(train$gyros_belt_x, center = TRUE, scale = TRUE)
test$gyros_belt_x <- scale(test$gyros_belt_x, center = TRUE, scale = TRUE)
train$gyros_belt_y <- scale(train$gyros_belt_y, center = TRUE, scale = TRUE)
test$gyros_belt_y <- scale(test$gyros_belt_y, center = TRUE, scale = TRUE)
train$gyros_belt_z <- scale(train$gyros_belt_z, center = TRUE, scale = TRUE)
test$gyros_belt_z <- scale(test$gyros_belt_z, center = TRUE, scale = TRUE)
train$roll_arm <- scale(train$roll_arm, center = TRUE, scale = TRUE)
test$roll_arm <- scale(test$roll_arm, center = TRUE, scale = TRUE)
train$pitch_arm <- scale(train$pitch_arm, center = TRUE, scale = TRUE)
test$pitch_arm <- scale(test$pitch_arm, center = TRUE, scale = TRUE)
train$yaw_arm <- scale(train$yaw_arm, center = TRUE, scale = TRUE)
test$yaw_arm <- scale(test$yaw_arm, center = TRUE, scale = TRUE)
train$gyros_arm_x <- scale(train$gyros_arm_x, center = TRUE, scale = TRUE)
test$gyros_arm_x <- scale(test$gyros_arm_x, center = TRUE, scale = TRUE)
train$gyros_arm_y <- scale(train$gyros_arm_y, center = TRUE, scale = TRUE)
test$gyros_arm_y <- scale(test$gyros_arm_y, center = TRUE, scale = TRUE)
train$gyros_arm_z <- scale(train$gyros_arm_z, center = TRUE, scale = TRUE)
test$gyros_arm_z <- scale(test$gyros_arm_z, center = TRUE, scale = TRUE)
train$roll_dumbbell <- scale(train$roll_dumbbell, center = TRUE, scale = TRUE)
test$roll_dumbbell <- scale(test$roll_dumbbell, center = TRUE, scale = TRUE)
train$pitch_dumbbell <- scale(train$pitch_dumbbell, center = TRUE, scale = TRUE)
test$pitch_dumbbell <- scale(test$pitch_dumbbell, center = TRUE, scale = TRUE)
train$yaw_dumbbell <- scale(train$yaw_dumbbell, center = TRUE, scale = TRUE)
test$yaw_dumbbell <- scale(test$yaw_dumbbell, center = TRUE, scale = TRUE)
train$gyros_dumbbell_x <- scale(train$gyros_dumbbell_x, center = TRUE, scale = TRUE)
test$gyros_dumbbell_x <- scale(test$gyros_dumbbell_x, center = TRUE, scale = TRUE)
train$gyros_dumbbell_y <- scale(train$gyros_dumbbell_y, center = TRUE, scale = TRUE)
test$gyros_dumbbell_y <- scale(test$gyros_dumbbell_y, center = TRUE, scale = TRUE)
train$gyros_dumbbell_z <- scale(train$gyros_dumbbell_z, center = TRUE, scale = TRUE)
test$gyros_dumbbell_z <- scale(test$gyros_dumbbell_z, center = TRUE, scale = TRUE)
train$magnet_dumbbell_z <- scale(train$magnet_dumbbell_z, center = TRUE, scale = TRUE)
test$magnet_dumbbell_z <- scale(test$magnet_dumbbell_z, center = TRUE, scale = TRUE)
train$roll_forearm <- scale(train$roll_forearm, center = TRUE, scale = TRUE)
test$roll_forearm <- scale(test$roll_forearm, center = TRUE, scale = TRUE)
train$pitch_forearm <- scale(train$pitch_forearm, center = TRUE, scale = TRUE)
test$pitch_forearm <- scale(test$pitch_forearm, center = TRUE, scale = TRUE)
train$yaw_forearm <- scale(train$yaw_forearm, center = TRUE, scale = TRUE)
test$yaw_forearm <- scale(test$yaw_forearm, center = TRUE, scale = TRUE)
train$gyros_forearm_x <- scale(train$gyros_forearm_x, center = TRUE, scale = TRUE)
test$gyros_forearm_x <- scale(test$gyros_forearm_x, center = TRUE, scale = TRUE)
train$gyros_forearm_y <- scale(train$gyros_forearm_y, center = TRUE, scale = TRUE)
test$gyros_forearm_y <- scale(test$gyros_forearm_y, center = TRUE, scale = TRUE)
train$gyros_forearm_z <- scale(train$gyros_forearm_z, center = TRUE, scale = TRUE)
test$gyros_forearm_z <- scale(test$gyros_forearm_z, center = TRUE, scale = TRUE)
train$magnet_forearm_y <- scale(train$magnet_forearm_y, center = TRUE, scale = TRUE)
test$magnet_forearm_y <- scale(test$magnet_forearm_y, center = TRUE, scale = TRUE)
train$magnet_forearm_z <- scale(train$magnet_forearm_z, center = TRUE, scale = TRUE)
test$magnet_forearm_z <- scale(test$magnet_forearm_z, center = TRUE, scale = TRUE)
train$cvtd_timestamp.year <- scale(train$cvtd_timestamp.year, center = TRUE, scale = TRUE)
test$cvtd_timestamp.year <- scale(test$cvtd_timestamp.year, center = TRUE, scale = TRUE)
train$cvtd_timestamp.month <- scale(train$cvtd_timestamp.month, center = TRUE, scale = TRUE)
test$cvtd_timestamp.month <- scale(test$cvtd_timestamp.month, center = TRUE, scale = TRUE)

train$kurtosis_roll_belt <- scale(as.numeric(train$kurtosis_roll_belt), center = TRUE, scale = TRUE)
```

```
## Warning in scale(as.numeric(train$kurtosis_roll_belt), center = TRUE,
## scale = TRUE): NAs introduced by coercion
```

```r
test$kurtosis_roll_belt <- scale(as.numeric(test$kurtosis_roll_belt), center = TRUE, scale = TRUE)

train$kurtosis_picth_belt <- scale(as.numeric(train$kurtosis_picth_belt), center = TRUE, scale = TRUE)
```

```
## Warning in scale(as.numeric(train$kurtosis_picth_belt), center = TRUE,
## scale = TRUE): NAs introduced by coercion
```

```r
test$kurtosis_picth_belt <- scale(as.numeric(test$kurtosis_picth_belt), center = TRUE, scale = TRUE)

train$skewness_roll_belt <- scale(as.numeric(train$skewness_roll_belt), center = TRUE, scale = TRUE)
```

```
## Warning in scale(as.numeric(train$skewness_roll_belt), center = TRUE,
## scale = TRUE): NAs introduced by coercion
```

```r
test$skewness_roll_belt <- scale(as.numeric(test$skewness_roll_belt), center = TRUE, scale = TRUE)

train$skewness_roll_belt.1 <- scale(as.numeric(train$skewness_roll_belt.1), center = TRUE, scale = TRUE)
```

```
## Warning in scale(as.numeric(train$skewness_roll_belt.1), center = TRUE, :
## NAs introduced by coercion
```

```r
test$skewness_roll_belt.1 <- scale(as.numeric(test$skewness_roll_belt.1), center = TRUE, scale = TRUE)

train$skewness_yaw_belt <- scale(as.numeric(train$skewness_yaw_belt), center = TRUE, scale = TRUE)
```

```
## Warning in scale(as.numeric(train$skewness_yaw_belt), center = TRUE, scale
## = TRUE): NAs introduced by coercion
```

```r
test$skewness_yaw_belt <- scale(as.numeric(test$skewness_yaw_belt), center = TRUE, scale = TRUE)
```

Center and scale the integer predictors.


```r
train$raw_timestamp_part_1 <- scale(train$raw_timestamp_part_1, center = TRUE, scale = TRUE)
test$raw_timestamp_part_1 <- scale(test$raw_timestamp_part_1, center = TRUE, scale = TRUE)
train$raw_timestamp_part_2 <- scale(train$raw_timestamp_part_2, center = TRUE, scale = TRUE)
test$raw_timestamp_part_2 <- scale(test$raw_timestamp_part_2, center = TRUE, scale = TRUE)
train$num_window <- scale(train$num_window, center = TRUE, scale = TRUE)
test$num_window <- scale(test$num_window, center = TRUE, scale = TRUE)
train$total_accel_belt <- scale(train$total_accel_belt, center = TRUE, scale = TRUE)
test$total_accel_belt <- scale(test$total_accel_belt, center = TRUE, scale = TRUE)
train$accel_belt_x <- scale(train$accel_belt_x, center = TRUE, scale = TRUE)
test$accel_belt_x <- scale(test$accel_belt_x, center = TRUE, scale = TRUE)
train$accel_belt_y <- scale(train$accel_belt_y, center = TRUE, scale = TRUE)
test$accel_belt_y <- scale(test$accel_belt_y, center = TRUE, scale = TRUE)
train$accel_belt_z <- scale(train$accel_belt_z, center = TRUE, scale = TRUE)
test$accel_belt_z <- scale(test$accel_belt_z, center = TRUE, scale = TRUE)
train$magnet_belt_x <- scale(train$magnet_belt_x, center = TRUE, scale = TRUE)
test$magnet_belt_x <- scale(test$magnet_belt_x, center = TRUE, scale = TRUE)
train$magnet_belt_y <- scale(train$magnet_belt_y, center = TRUE, scale = TRUE)
test$magnet_belt_y <- scale(test$magnet_belt_y, center = TRUE, scale = TRUE)
train$magnet_belt_z <- scale(train$magnet_belt_z, center = TRUE, scale = TRUE)
test$magnet_belt_z <- scale(test$magnet_belt_z, center = TRUE, scale = TRUE)
train$total_accel_arm <- scale(train$total_accel_arm, center = TRUE, scale = TRUE)
test$total_accel_arm <- scale(test$total_accel_arm, center = TRUE, scale = TRUE)
train$accel_arm_x <- scale(train$accel_arm_x, center = TRUE, scale = TRUE)
test$accel_arm_x <- scale(test$accel_arm_x, center = TRUE, scale = TRUE)
train$accel_arm_y <- scale(train$accel_arm_y, center = TRUE, scale = TRUE)
test$accel_arm_y <- scale(test$accel_arm_y, center = TRUE, scale = TRUE)
train$accel_arm_z <- scale(train$accel_arm_z, center = TRUE, scale = TRUE)
test$accel_arm_z <- scale(test$accel_arm_z, center = TRUE, scale = TRUE)
train$magnet_arm_x <- scale(train$magnet_arm_x, center = TRUE, scale = TRUE)
test$magnet_arm_x <- scale(test$magnet_arm_x, center = TRUE, scale = TRUE)
train$magnet_arm_y <- scale(train$magnet_arm_y, center = TRUE, scale = TRUE)
test$magnet_arm_y <- scale(test$magnet_arm_y, center = TRUE, scale = TRUE)
train$magnet_arm_z <- scale(train$magnet_arm_z, center = TRUE, scale = TRUE)
test$magnet_arm_z <- scale(test$magnet_arm_z, center = TRUE, scale = TRUE)
train$total_accel_dumbbell <- scale(train$total_accel_dumbbell, center = TRUE, scale = TRUE)
test$total_accel_dumbbell <- scale(test$total_accel_dumbbell, center = TRUE, scale = TRUE)
train$accel_dumbbell_x <- scale(train$accel_dumbbell_x, center = TRUE, scale = TRUE)
test$accel_dumbbell_x <- scale(test$accel_dumbbell_x, center = TRUE, scale = TRUE)
train$accel_dumbbell_y <- scale(train$accel_dumbbell_y, center = TRUE, scale = TRUE)
test$accel_dumbbell_y <- scale(test$accel_dumbbell_y, center = TRUE, scale = TRUE)
train$accel_dumbbell_z <- scale(train$accel_dumbbell_z, center = TRUE, scale = TRUE)
test$accel_dumbbell_z <- scale(test$accel_dumbbell_z, center = TRUE, scale = TRUE)
train$magnet_dumbbell_x <- scale(train$magnet_dumbbell_x, center = TRUE, scale = TRUE)
test$magnet_dumbbell_x <- scale(test$magnet_dumbbell_x, center = TRUE, scale = TRUE)
train$magnet_dumbbell_y <- scale(train$magnet_dumbbell_y, center = TRUE, scale = TRUE)
test$magnet_dumbbell_y <- scale(test$magnet_dumbbell_y, center = TRUE, scale = TRUE)
train$total_accel_forearm <- scale(train$total_accel_forearm, center = TRUE, scale = TRUE)
test$total_accel_forearm <- scale(test$total_accel_forearm, center = TRUE, scale = TRUE)
train$accel_forearm_x <- scale(train$accel_forearm_x, center = TRUE, scale = TRUE)
test$accel_forearm_x <- scale(test$accel_forearm_x, center = TRUE, scale = TRUE)
train$accel_forearm_y <- scale(train$accel_forearm_y, center = TRUE, scale = TRUE)
test$accel_forearm_y <- scale(test$accel_forearm_y, center = TRUE, scale = TRUE)
train$accel_forearm_z <- scale(train$accel_forearm_z, center = TRUE, scale = TRUE)
test$accel_forearm_z <- scale(test$accel_forearm_z, center = TRUE, scale = TRUE)
train$magnet_forearm_x <- scale(train$magnet_forearm_x, center = TRUE, scale = TRUE)
test$magnet_forearm_x <- scale(test$magnet_forearm_x, center = TRUE, scale = TRUE)
train$cvtd_timestamp.day <- scale(train$cvtd_timestamp.day, center = TRUE, scale = TRUE)
test$cvtd_timestamp.day <- scale(test$cvtd_timestamp.day, center = TRUE, scale = TRUE)
train$cvtd_timestamp.hour <- scale(train$cvtd_timestamp.hour, center = TRUE, scale = TRUE)
test$cvtd_timestamp.hour <- scale(test$cvtd_timestamp.hour, center = TRUE, scale = TRUE)
train$cvtd_timestamp.minute <- scale(train$cvtd_timestamp.minute, center = TRUE, scale = TRUE)
test$cvtd_timestamp.minute <- scale(test$cvtd_timestamp.minute, center = TRUE, scale = TRUE)
```

Format factors.


```r
train$user_name <- as.factor(train$user_name)
train$classe <- as.factor(train$classe)
train$new_window <- as.factor(train$new_window)
train$kurtosis_yaw_belt <- as.factor(train$kurtosis_yaw_belt)

test$user_name <- as.factor(test$user_name)
#test$classe <- as.factor(test$classe)
test$new_window <- as.factor(test$new_window)
test$kurtosis_yaw_belt <- as.factor(test$kurtosis_yaw_belt)
```

Impute missing values.


```r
train$kurtosis_roll_belt[is.na(train$kurtosis_roll_belt)] <- median(train$kurtosis_roll_belt[inSplit], na.rm = TRUE)
test$kurtosis_roll_belt[is.na(test$kurtosis_roll_belt)] <- median(test$kurtosis_roll_belt[inSplit], na.rm = TRUE)

train$kurtosis_picth_belt[is.na(train$kurtosis_picth_belt)] <- median(train$kurtosis_picth_belt[inSplit], na.rm = TRUE)
test$kurtosis_picth_belt[is.na(test$kurtosis_picth_belt)] <- median(test$kurtosis_picth_belt[inSplit], na.rm = TRUE)

train$skewness_roll_belt[is.na(train$skewness_roll_belt)] <- median(train$skewness_roll_belt[inSplit], na.rm = TRUE)
test$skewness_roll_belt[is.na(test$skewness_roll_belt)] <- median(test$skewness_roll_belt[inSplit], na.rm = TRUE)

train$skewness_roll_belt.1[is.na(train$skewness_roll_belt.1)] <- median(train$skewness_roll_belt.1[inSplit], na.rm = TRUE)
test$skewness_roll_belt.1[is.na(test$skewness_roll_belt.1)] <- median(test$skewness_roll_belt.1[inSplit], na.rm = TRUE)
```

Split the training data for cross validation.


```r
train1 <- train[inSplit, -exclude.predictors]
train2 <- train[-inSplit, -exclude.predictors]
```

Remove zero- and near zero-variance predictors.


```r
nzv <- nearZeroVar(train1)

train1 <- train1[, -nzv]
train2 <- train2[, -nzv]
```


#Analysis


```r
summary(train$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```


```r
fit.rpart <- train(classe~., data = train1, method = "rpart")
fit.gbm <- train(classe ~ ., data = train1, method = "gbm")
#fit.svm <- svm(classe ~ ., train1)
#fit.rf <- train(classe ~ ., data = train1, method = "rf")
fit.lda <- train(classe ~ ., data = train1, method = "lda")
```


```r
pred.rpart <- predict(fit.rpart, train2)
pred.gbm <- predict(fit.gbm, train2)
#pred.svm <- predict(fit.svm, train2)
#pred.rf <- predict(fit.rf, train2)
pred.lda <- predict(fit.lda, train2)
```


```r
confusionMatrix(pred.rpart, train2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1489  291  160  174   40
##          B   40  544   33  165   81
##          C  140  304  833  584  297
##          D    0    0    0    0    0
##          E    5    0    0   41  664
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5998          
##                  95% CI : (0.5872, 0.6124)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4879          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8895  0.47761   0.8119   0.0000   0.6137
## Specificity            0.8421  0.93279   0.7273   1.0000   0.9904
## Pos Pred Value         0.6913  0.63036   0.3860      NaN   0.9352
## Neg Pred Value         0.9504  0.88152   0.9482   0.8362   0.9192
## Prevalence             0.2845  0.19354   0.1743   0.1638   0.1839
## Detection Rate         0.2530  0.09244   0.1415   0.0000   0.1128
## Detection Prevalence   0.3660  0.14664   0.3667   0.0000   0.1206
## Balanced Accuracy      0.8658  0.70520   0.7696   0.5000   0.8021
```

```r
confusionMatrix(pred.gbm, train2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    2    0    0    0
##          B    3 1128    2    0    0
##          C    0    9 1019    0    0
##          D    0    0    5  961    3
##          E    0    0    0    3 1079
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9954         
##                  95% CI : (0.9933, 0.997)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9942         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9903   0.9932   0.9969   0.9972
## Specificity            0.9995   0.9989   0.9981   0.9984   0.9994
## Pos Pred Value         0.9988   0.9956   0.9912   0.9917   0.9972
## Neg Pred Value         0.9993   0.9977   0.9986   0.9994   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2839   0.1917   0.1732   0.1633   0.1833
## Detection Prevalence   0.2843   0.1925   0.1747   0.1647   0.1839
## Balanced Accuracy      0.9989   0.9946   0.9957   0.9976   0.9983
```

```r
#confusionMatrix(pred.svm, train2$classe)
#confusionMatrix(pred.rf, train2$classe)
confusionMatrix(pred.lda, train2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1524  156   13    0    0
##          B  115  800   92    4    0
##          C   35  176  885   92   26
##          D    0    6   36  845  159
##          E    0    1    0   23  897
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8413          
##                  95% CI : (0.8317, 0.8505)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7994          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9104   0.7024   0.8626   0.8766   0.8290
## Specificity            0.9599   0.9555   0.9323   0.9592   0.9950
## Pos Pred Value         0.9002   0.7913   0.7290   0.8078   0.9739
## Neg Pred Value         0.9642   0.9304   0.9698   0.9754   0.9627
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2590   0.1359   0.1504   0.1436   0.1524
## Detection Prevalence   0.2877   0.1718   0.2063   0.1777   0.1565
## Balanced Accuracy      0.9351   0.8290   0.8974   0.9179   0.9120
```

#Model and Feature Selection

Recursive Partitioning/Decision Trees Accuracy: **0.5998**

Gradient Boosting (GBM) Accuracy: **0.9954**

Random Forest Accuracy: **NA**

Linear Discriminant Analysis (LDA) Accuracy: **0.8413**

At 99.54% Accuracy, we will use the GBM model for predicting new observations.

The best predictors for this dataset are below.


```r
predictors(fit.gbm)
```

```
##  [1] "raw_timestamp_part_1"  "num_window"           
##  [3] "roll_belt"             "pitch_belt"           
##  [5] "yaw_belt"              "gyros_belt_y"         
##  [7] "gyros_belt_z"          "accel_belt_z"         
##  [9] "magnet_belt_x"         "magnet_belt_y"        
## [11] "magnet_belt_z"         "roll_arm"             
## [13] "yaw_arm"               "gyros_arm_y"          
## [15] "accel_arm_x"           "magnet_arm_x"         
## [17] "magnet_arm_y"          "magnet_arm_z"         
## [19] "roll_dumbbell"         "pitch_dumbbell"       
## [21] "yaw_dumbbell"          "gyros_dumbbell_x"     
## [23] "gyros_dumbbell_y"      "accel_dumbbell_x"     
## [25] "accel_dumbbell_y"      "accel_dumbbell_z"     
## [27] "magnet_dumbbell_x"     "magnet_dumbbell_y"    
## [29] "magnet_dumbbell_z"     "roll_forearm"         
## [31] "pitch_forearm"         "gyros_forearm_z"      
## [33] "accel_forearm_x"       "accel_forearm_z"      
## [35] "magnet_forearm_x"      "magnet_forearm_y"     
## [37] "magnet_forearm_z"      "cvtd_timestamp.month" 
## [39] "cvtd_timestamp.day"    "cvtd_timestamp.minute"
```

#Prediction


```r
test$prediction <- predict(fit.gbm, test)
test[,which(names(test) %in% c("problem_id", "prediction"))]
```

```
##    problem_id prediction
## 1           1          E
## 2           2          A
## 3           3          A
## 4           4          E
## 5           5          C
## 6           6          B
## 7           7          D
## 8           8          D
## 9           9          E
## 10         10          E
## 11         11          E
## 12         12          E
## 13         13          E
## 14         14          A
## 15         15          E
## 16         16          E
## 17         17          E
## 18         18          E
## 19         19          E
## 20         20          E
```

#Appendix


```r
#str(train)

summary(train)
```

```
##        X            user_name    raw_timestamp_part_1.V1
##  Min.   :    1   adelmo  :3892   Min.   :-1.6469921     
##  1st Qu.: 4906   carlitos:3112   1st Qu.:-0.7515835     
##  Median : 9812   charles :3536   Median : 0.0283062     
##  Mean   : 9812   eurico  :3070   Mean   : 0.0000000     
##  3rd Qu.:14717   jeremy  :3402   3rd Qu.: 1.2548072     
##  Max.   :19622   pedro   :2610   Max.   : 1.3075917     
##                                                         
##  raw_timestamp_part_2.V1 new_window     num_window.V1    
##  Min.   :-1.7360251      no :19216   Min.   :-1.7330516  
##  1st Qu.:-0.8595567      yes:  406   1st Qu.:-0.8415974  
##  Median :-0.0148362                  Median :-0.0267842  
##  Mean   : 0.0000000                  Mean   : 0.0000000  
##  3rd Qu.: 0.8716678                  3rd Qu.: 0.8606363  
##  Max.   : 1.7283321                  Max.   : 1.7480567  
##                                                          
##      roll_belt.V1        pitch_belt.V1         yaw_belt.V1     
##  Min.   :-1.4869612   Min.   :-2.5101640   Min.   :-1.7731692  
##  1st Qu.:-1.0088755   1st Qu.: 0.0650844   1st Qu.:-0.8098725  
##  Median : 0.7743842   Median : 0.2225701   Median :-0.0188556  
##  Mean   : 0.0000000   Mean   : 0.0000000   Mean   : 0.0000000  
##  3rd Qu.: 0.9337461   3rd Qu.: 0.6529712   3rd Qu.: 0.2532206  
##  Max.   : 1.5552575   Max.   : 2.6841782   Max.   : 1.9980798  
##                                                                
##  total_accel_belt.V1  kurtosis_roll_belt.V1 kurtosis_picth_belt.V1
##  Min.   :-1.4611414   Min.   :-0.650003     Min.   :-0.563307     
##  1st Qu.:-1.0736601   1st Qu.:-0.224189     1st Qu.:-0.387225     
##  Median : 0.7345860   Median :-0.224189     Median :-0.387225     
##  Mean   : 0.0000000   Mean   :-0.219665     Mean   :-0.379844     
##  3rd Qu.: 0.8637464   3rd Qu.:-0.224189     3rd Qu.:-0.387225     
##  Max.   : 2.2845112   Max.   :11.359108     Max.   : 4.633227     
##                                                                   
##  kurtosis_yaw_belt skewness_roll_belt.V1 skewness_roll_belt.1.V1
##         :19216     Min.   :-6.240261     Min.   :-3.240476      
##  #DIV/0!:  406     1st Qu.: 0.028478     1st Qu.: 0.094467      
##                    Median : 0.028478     Median : 0.094467      
##                    Mean   : 0.027901     Mean   : 0.092667      
##                    3rd Qu.: 0.028478     3rd Qu.: 0.094467      
##                    Max.   : 3.951914     Max.   : 3.384175      
##                                                                 
##  skewness_yaw_belt.V1 max_yaw_belt       min_yaw_belt      
##  Min.   : NA          Length:19622       Length:19622      
##  1st Qu.: NA          Class :character   Class :character  
##  Median : NA          Mode  :character   Mode  :character  
##  Mean   :NaN                                               
##  3rd Qu.: NA                                               
##  Max.   : NA                                               
##  NA's   :19622                                             
##  amplitude_yaw_belt   gyros_belt_x.V1     gyros_belt_y.V1  
##  Length:19622       Min.   :-4.989209   Min.   :-8.686434  
##  Class :character   1st Qu.:-0.117725   1st Qu.:-0.506007  
##  Mode  :character   Median : 0.171670   Median :-0.250368  
##                     Mean   : 0.000000   Mean   : 0.000000  
##                     3rd Qu.: 0.557530   3rd Qu.: 0.900004  
##                     Max.   :10.734590   Max.   : 7.674420  
##                                                            
##    gyros_belt_z.V1     accel_belt_x.V1     accel_belt_y.V1  
##  Min.   :-5.509099   Min.   :-3.859205   Min.   :-3.469643  
##  1st Qu.:-0.287842   1st Qu.:-0.519663   1st Qu.:-0.950208  
##  Median : 0.126544   Median :-0.317267   Median : 0.169541  
##  Mean   : 0.000000   Mean   : 0.000000   Mean   : 0.000000  
##  3rd Qu.: 0.458052   3rd Qu.: 0.020061   3rd Qu.: 1.079337  
##  Max.   : 7.253974   Max.   : 3.056008   Max.   : 4.683529  
##                                                             
##    accel_belt_z.V1     magnet_belt_x.V1    magnet_belt_y.V1  
##  Min.   :-2.0150617   Min.   :-1.676567   Min.   :-6.717244  
##  1st Qu.:-0.8900874   1st Qu.:-0.726104   1st Qu.:-0.355288  
##  Median :-0.7905322   Median :-0.320989   Median : 0.205236  
##  Mean   : 0.0000000   Mean   : 0.000000   Mean   : 0.000000  
##  3rd Qu.: 0.9915067   3rd Qu.: 0.052963   3rd Qu.: 0.457472  
##  Max.   : 1.7680377   Max.   : 6.690619   Max.   : 2.223125  
##                                                              
##   magnet_belt_z.V1       roll_arm.V1          pitch_arm.V1    
##  Min.   :-4.255806   Min.   :-2.7195375   Min.   :-2.7438956  
##  1st Qu.:-0.452682   1st Qu.:-0.6819139   1st Qu.:-0.6938305  
##  Median : 0.390753   Median :-0.2451085   Median : 0.1503139  
##  Mean   : 0.000000   Mean   : 0.0000000   Mean   : 0.0000000  
##  3rd Qu.: 0.605446   3rd Qu.: 0.8175213   3rd Qu.: 0.5153494  
##  Max.   : 9.791217   Max.   : 2.2293206   Max.   : 3.0347458  
##                                                               
##       yaw_arm.V1      total_accel_arm.V1    gyros_arm_x.V1   
##  Min.   :-2.5136315   Min.   :-2.329208   Min.   :-3.216685  
##  1st Qu.:-0.5952807   1st Qu.:-0.808624   1st Qu.:-0.688591  
##  Median : 0.0086704   Median : 0.141741   Median : 0.018674  
##  Mean   : 0.0000000   Mean   : 0.000000   Mean   : 0.000000  
##  3rd Qu.: 0.6515071   3rd Qu.: 0.711960   3rd Qu.: 0.766066  
##  Max.   : 2.5309723   Max.   : 3.848165   Max.   : 2.421366  
##                                                              
##    gyros_arm_y.V1      gyros_arm_z.V1       accel_arm_x.V1   
##  Min.   :-3.738495   Min.   :-4.699253   Min.   :-1.8883243  
##  1st Qu.:-0.637685   1st Qu.:-0.613701   1st Qu.:-0.9984339  
##  Median : 0.020062   Median :-0.071371   Median : 0.0892098  
##  Mean   : 0.000000   Mean   : 0.000000   Mean   : 0.0000000  
##  3rd Qu.: 0.466391   3rd Qu.: 0.814435   3rd Qu.: 0.7923330  
##  Max.   : 3.637673   Max.   : 4.972297   Max.   : 2.7314150  
##                                                              
##    accel_arm_y.V1      accel_arm_z.V1      magnet_arm_x.V1   
##  Min.   :-3.191133   Min.   :-4.194155   Min.   :-1.7485284  
##  1st Qu.:-0.788206   1st Qu.:-0.532888   1st Qu.:-1.1083745  
##  Median :-0.169271   Median : 0.180056   Median : 0.2192686  
##  Mean   : 0.000000   Mean   : 0.000000   Mean   : 0.0000000  
##  3rd Qu.: 0.968479   3rd Qu.: 0.699912   3rd Qu.: 1.0036825  
##  Max.   : 2.506716   Max.   : 2.697641   Max.   : 1.3305216  
##                                                              
##    magnet_arm_y.V1      magnet_arm_z.V1    kurtosis_roll_arm 
##  Min.   :-2.7171016   Min.   :-2.7661899   Length:19622      
##  1st Qu.:-0.8202214   1st Qu.:-0.5365298   Class :character  
##  Median : 0.2247961   Median : 0.4210070   Mode  :character  
##  Mean   : 0.0000000   Mean   : 0.0000000                     
##  3rd Qu.: 0.8240716   3rd Qu.: 0.7302356                     
##  Max.   : 2.1117709   Max.   : 1.1864242                     
##                                                              
##  kurtosis_picth_arm kurtosis_yaw_arm   skewness_roll_arm 
##  Length:19622       Length:19622       Length:19622      
##  Class :character   Class :character   Class :character  
##  Mode  :character   Mode  :character   Mode  :character  
##                                                          
##                                                          
##                                                          
##                                                          
##  skewness_pitch_arm skewness_yaw_arm     roll_dumbbell.V1  
##  Length:19622       Length:19622       Min.   :-2.5389800  
##  Class :character   Class :character   1st Qu.:-0.6053753  
##  Mode  :character   Mode  :character   Median : 0.3479498  
##                                        Mean   : 0.0000000  
##                                        3rd Qu.: 0.6259319  
##                                        Max.   : 1.8547243  
##                                                            
##   pitch_dumbbell.V1    yaw_dumbbell.V1    kurtosis_roll_dumbbell
##  Min.   :-3.752394   Min.   :-1.8486513   Length:19622          
##  1st Qu.:-0.813858   1st Qu.:-0.9612369   Class :character      
##  Median :-0.275137   Median :-0.0605708   Mode  :character      
##  Mean   : 0.000000   Mean   : 0.0000000                         
##  3rd Qu.: 0.764422   3rd Qu.: 0.9448757                         
##  Max.   : 4.329999   Max.   : 1.8575273                         
##                                                                 
##  kurtosis_picth_dumbbell kurtosis_yaw_dumbbell skewness_roll_dumbbell
##  Length:19622            Length:19622          Length:19622          
##  Class :character        Class :character      Class :character      
##  Mode  :character        Mode  :character      Mode  :character      
##                                                                      
##                                                                      
##                                                                      
##                                                                      
##  skewness_pitch_dumbbell skewness_yaw_dumbbell max_yaw_dumbbell  
##  Length:19622            Length:19622          Length:19622      
##  Class :character        Class :character      Class :character  
##  Mode  :character        Mode  :character      Mode  :character  
##                                                                  
##                                                                  
##                                                                  
##                                                                  
##  min_yaw_dumbbell   amplitude_yaw_dumbbell total_accel_dumbbell.V1
##  Length:19622       Length:19622           Min.   :-1.340545      
##  Class :character   Class :character       1st Qu.:-0.949669      
##  Mode  :character   Mode  :character       Median :-0.363356      
##                                            Mean   : 0.000000      
##                                            3rd Qu.: 0.516114      
##                                            Max.   : 4.327151      
##                                                                   
##  gyros_dumbbell_x.V1  gyros_dumbbell_y.V1 gyros_dumbbell_z.V1
##  Min.   :-135.33567   Min.   :-3.51818    Min.   : -0.98444  
##  1st Qu.:  -0.12667   1st Qu.:-0.30502    1st Qu.: -0.07916  
##  Median :  -0.02061   Median :-0.02632    Median : -0.00044  
##  Mean   :   0.00000   Mean   : 0.00000    Mean   :  0.00000  
##  3rd Qu.:   0.12523   3rd Qu.: 0.26876    3rd Qu.:  0.06953  
##  Max.   :   1.36483   Max.   :85.17175    Max.   :138.69025  
##                                                              
##  accel_dumbbell_x.V1 accel_dumbbell_y.V1 accel_dumbbell_z.V1
##  Min.   :-5.799214   Min.   :-2.992338   Min.   :-2.701030  
##  1st Qu.:-0.317667   1st Qu.:-0.750870   1st Qu.:-0.947100  
##  Median : 0.306249   Median :-0.137872   Median : 0.340942  
##  Mean   : 0.000000   Mean   : 0.000000   Mean   : 0.000000  
##  3rd Qu.: 0.588497   3rd Qu.: 0.722803   3rd Qu.: 0.697209  
##  Max.   : 3.916048   Max.   : 3.249098   Max.   : 3.255023  
##                                                             
##  magnet_dumbbell_x.V1 magnet_dumbbell_y.V1 magnet_dumbbell_z.V1
##  Min.   :-0.9258074   Min.   :-11.689700   Min.   :-2.2009532  
##  1st Qu.:-0.6079025   1st Qu.:  0.030700   1st Qu.:-0.6505519  
##  Median :-0.4430629   Median :  0.275449   Median :-0.2361589  
##  Mean   : 0.0000000   Mean   :  0.000000   Mean   : 0.0000000  
##  3rd Qu.: 0.0720608   3rd Qu.:  0.517138   3rd Qu.: 0.3497070  
##  Max.   : 2.7094941   Max.   :  1.260562   Max.   : 2.9003672  
##                                                                
##    roll_forearm.V1      pitch_forearm.V1      yaw_forearm.V1   
##  Min.   :-1.9792181   Min.   :-2.9562768   Min.   :-1.9299483  
##  1st Qu.:-0.3199311   1st Qu.:-0.3803797   1st Qu.:-0.8506912  
##  Median :-0.1122455   Median :-0.0520860   Median :-0.1860859  
##  Mean   : 0.0000000   Mean   : 0.0000000   Mean   : 0.0000000  
##  3rd Qu.: 0.9827613   3rd Qu.: 0.6286614   3rd Qu.: 0.8796077  
##  Max.   : 1.3530087   Max.   : 2.8101798   Max.   : 1.5577764  
##                                                                
##  kurtosis_roll_forearm kurtosis_picth_forearm kurtosis_yaw_forearm
##  Length:19622          Length:19622           Length:19622        
##  Class :character      Class :character       Class :character    
##  Mode  :character      Mode  :character       Mode  :character    
##                                                                   
##                                                                   
##                                                                   
##                                                                   
##  skewness_roll_forearm skewness_pitch_forearm skewness_yaw_forearm
##  Length:19622          Length:19622           Length:19622        
##  Class :character      Class :character       Class :character    
##  Mode  :character      Mode  :character       Mode  :character    
##                                                                   
##                                                                   
##                                                                   
##                                                                   
##  max_yaw_forearm    min_yaw_forearm    amplitude_yaw_forearm
##  Length:19622       Length:19622       Length:19622         
##  Class :character   Class :character   Class :character     
##  Mode  :character   Mode  :character   Mode  :character     
##                                                             
##                                                             
##                                                             
##                                                             
##  total_accel_forearm.V1 gyros_forearm_x.V1  gyros_forearm_y.V1 
##  Min.   :-3.452215      Min.   :-34.16180   Min.   : -2.28823  
##  1st Qu.:-0.568352      1st Qu.: -0.58270   1st Qu.: -0.49510  
##  Median : 0.127753      Median : -0.16643   Median : -0.01457  
##  Mean   : 0.000000      Mean   :  0.00000   Mean   :  0.00000  
##  3rd Qu.: 0.624971      3rd Qu.:  0.61985   3rd Qu.:  0.49821  
##  Max.   : 7.287689      Max.   :  5.87719   Max.   :100.27489  
##                                                                
##  gyros_forearm_z.V1   accel_forearm_x.V1  accel_forearm_y.V1 
##  Min.   : -4.69725   Min.   :-2.4161873   Min.   :-3.975694  
##  1st Qu.: -0.18880   1st Qu.:-0.6442539   1st Qu.:-0.532933  
##  Median : -0.04061   Median : 0.0257585   Median : 0.186599  
##  Mean   :  0.00000   Mean   : 0.0000000   Mean   : 0.000000  
##  3rd Qu.:  0.19308   3rd Qu.: 0.7622183   3rd Qu.: 0.741238  
##  Max.   :131.57654   Max.   : 2.9826725   Max.   : 3.794253  
##                                                              
##   accel_forearm_z.V1  magnet_forearm_x.V1  magnet_forearm_y.V1 
##  Min.   :-2.8230976   Min.   :-2.7882994   Min.   :-2.5052655  
##  1st Qu.:-0.9155410   1st Qu.:-0.8745257   1st Qu.:-0.7423163  
##  Median : 0.1177188   Median :-0.1885647   Median : 0.4140055  
##  Mean   : 0.0000000   Mean   : 0.0000000   Mean   : 0.0000000  
##  3rd Qu.: 0.5873823   3rd Qu.: 0.6905030   3rd Qu.: 0.7006320  
##  Max.   : 2.5021644   Max.   : 2.8377340   Max.   : 2.1592859  
##                                                                
##  magnet_forearm_z.V1 classe   cvtd_timestamp.year.V1
##  Min.   :-3.700865   A:5580   Min.   : NA           
##  1st Qu.:-0.548689   B:3797   1st Qu.: NA           
##  Median : 0.317888   C:3422   Median : NA           
##  Mean   : 0.000000   D:3216   Mean   :NaN           
##  3rd Qu.: 0.702432   E:3607   3rd Qu.: NA           
##  Max.   : 1.885852            Max.   : NA           
##                               NA's   :19622         
##  cvtd_timestamp.month.V1 cvtd_timestamp.day.V1 cvtd_timestamp.hour.V1
##  Min.   :-1.4253862      Min.   :-0.8044125    Min.   :-1.6003460    
##  1st Qu.:-1.4253862      1st Qu.:-0.8044125    1st Qu.:-0.4757135    
##  Median : 0.7015285      Median :-0.5580951    Median : 0.0866028    
##  Mean   : 0.0000000      Mean   : 0.0000000    Mean   : 0.0000000    
##  3rd Qu.: 0.7015285      3rd Qu.: 1.3303382    3rd Qu.: 0.0866028    
##  Max.   : 0.7015285      Max.   : 1.4945498    Max.   : 1.7735516    
##                                                                      
##  cvtd_timestamp.minute.V1
##  Min.   :-1.1463479      
##  1st Qu.:-0.8939209      
##  Median :-0.2628536      
##  Mean   : 0.0000000      
##  3rd Qu.: 0.3682137      
##  Max.   : 1.9458821      
## 
```
