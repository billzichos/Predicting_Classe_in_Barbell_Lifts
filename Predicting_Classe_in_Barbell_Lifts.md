# Predicting Classe in Barbell Lifts
Bill Zichos  
September 25, 2016  

I developed a model that with >99% accuracy, predicts the classe for different barbell lifts.  The best model used gradient boosting and 40 predictors including some features engineers from the date.  The rest of this document is organized in the following sections.

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
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1327
##      2        1.5208             nan     0.1000    0.0868
##      3        1.4621             nan     0.1000    0.0695
##      4        1.4167             nan     0.1000    0.0536
##      5        1.3802             nan     0.1000    0.0508
##      6        1.3475             nan     0.1000    0.0477
##      7        1.3170             nan     0.1000    0.0402
##      8        1.2921             nan     0.1000    0.0426
##      9        1.2635             nan     0.1000    0.0368
##     10        1.2408             nan     0.1000    0.0336
##     20        1.0687             nan     0.1000    0.0198
##     40        0.8680             nan     0.1000    0.0113
##     60        0.7335             nan     0.1000    0.0102
##     80        0.6313             nan     0.1000    0.0047
##    100        0.5511             nan     0.1000    0.0046
##    120        0.4871             nan     0.1000    0.0044
##    140        0.4335             nan     0.1000    0.0043
##    150        0.4101             nan     0.1000    0.0036
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1945
##      2        1.4832             nan     0.1000    0.1397
##      3        1.3918             nan     0.1000    0.1094
##      4        1.3211             nan     0.1000    0.0924
##      5        1.2615             nan     0.1000    0.0723
##      6        1.2137             nan     0.1000    0.0733
##      7        1.1678             nan     0.1000    0.0801
##      8        1.1191             nan     0.1000    0.0623
##      9        1.0806             nan     0.1000    0.0498
##     10        1.0496             nan     0.1000    0.0527
##     20        0.7831             nan     0.1000    0.0387
##     40        0.5006             nan     0.1000    0.0167
##     60        0.3410             nan     0.1000    0.0102
##     80        0.2401             nan     0.1000    0.0032
##    100        0.1808             nan     0.1000    0.0045
##    120        0.1365             nan     0.1000    0.0020
##    140        0.1059             nan     0.1000    0.0026
##    150        0.0947             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2494
##      2        1.4502             nan     0.1000    0.1702
##      3        1.3415             nan     0.1000    0.1452
##      4        1.2508             nan     0.1000    0.0991
##      5        1.1866             nan     0.1000    0.1116
##      6        1.1189             nan     0.1000    0.0943
##      7        1.0617             nan     0.1000    0.0808
##      8        1.0119             nan     0.1000    0.0876
##      9        0.9598             nan     0.1000    0.0747
##     10        0.9147             nan     0.1000    0.0571
##     20        0.6039             nan     0.1000    0.0345
##     40        0.3317             nan     0.1000    0.0141
##     60        0.1908             nan     0.1000    0.0091
##     80        0.1187             nan     0.1000    0.0028
##    100        0.0818             nan     0.1000    0.0023
##    120        0.0592             nan     0.1000    0.0013
##    140        0.0448             nan     0.1000    0.0011
##    150        0.0386             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1311
##      2        1.5234             nan     0.1000    0.0862
##      3        1.4663             nan     0.1000    0.0701
##      4        1.4211             nan     0.1000    0.0567
##      5        1.3847             nan     0.1000    0.0465
##      6        1.3545             nan     0.1000    0.0443
##      7        1.3259             nan     0.1000    0.0436
##      8        1.2988             nan     0.1000    0.0349
##      9        1.2768             nan     0.1000    0.0399
##     10        1.2496             nan     0.1000    0.0325
##     20        1.0788             nan     0.1000    0.0196
##     40        0.8720             nan     0.1000    0.0108
##     60        0.7372             nan     0.1000    0.0082
##     80        0.6324             nan     0.1000    0.0044
##    100        0.5550             nan     0.1000    0.0052
##    120        0.4883             nan     0.1000    0.0043
##    140        0.4332             nan     0.1000    0.0041
##    150        0.4089             nan     0.1000    0.0041
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1926
##      2        1.4858             nan     0.1000    0.1293
##      3        1.4019             nan     0.1000    0.1096
##      4        1.3331             nan     0.1000    0.0869
##      5        1.2780             nan     0.1000    0.0803
##      6        1.2273             nan     0.1000    0.0765
##      7        1.1800             nan     0.1000    0.0682
##      8        1.1367             nan     0.1000    0.0563
##      9        1.1015             nan     0.1000    0.0523
##     10        1.0682             nan     0.1000    0.0623
##     20        0.7941             nan     0.1000    0.0297
##     40        0.5219             nan     0.1000    0.0173
##     60        0.3468             nan     0.1000    0.0079
##     80        0.2494             nan     0.1000    0.0101
##    100        0.1845             nan     0.1000    0.0049
##    120        0.1384             nan     0.1000    0.0024
##    140        0.1082             nan     0.1000    0.0030
##    150        0.0959             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2371
##      2        1.4574             nan     0.1000    0.1697
##      3        1.3511             nan     0.1000    0.1317
##      4        1.2674             nan     0.1000    0.1226
##      5        1.1916             nan     0.1000    0.0963
##      6        1.1309             nan     0.1000    0.0858
##      7        1.0780             nan     0.1000    0.1029
##      8        1.0169             nan     0.1000    0.0806
##      9        0.9682             nan     0.1000    0.0693
##     10        0.9262             nan     0.1000    0.0641
##     20        0.6144             nan     0.1000    0.0375
##     40        0.3213             nan     0.1000    0.0132
##     60        0.1913             nan     0.1000    0.0074
##     80        0.1233             nan     0.1000    0.0054
##    100        0.0841             nan     0.1000    0.0021
##    120        0.0612             nan     0.1000    0.0007
##    140        0.0451             nan     0.1000    0.0009
##    150        0.0392             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1266
##      2        1.5245             nan     0.1000    0.0829
##      3        1.4693             nan     0.1000    0.0663
##      4        1.4262             nan     0.1000    0.0519
##      5        1.3918             nan     0.1000    0.0470
##      6        1.3613             nan     0.1000    0.0450
##      7        1.3323             nan     0.1000    0.0357
##      8        1.3078             nan     0.1000    0.0414
##      9        1.2802             nan     0.1000    0.0323
##     10        1.2585             nan     0.1000    0.0315
##     20        1.0863             nan     0.1000    0.0223
##     40        0.8836             nan     0.1000    0.0120
##     60        0.7464             nan     0.1000    0.0072
##     80        0.6438             nan     0.1000    0.0058
##    100        0.5610             nan     0.1000    0.0047
##    120        0.4944             nan     0.1000    0.0028
##    140        0.4381             nan     0.1000    0.0023
##    150        0.4159             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1923
##      2        1.4852             nan     0.1000    0.1285
##      3        1.4016             nan     0.1000    0.1032
##      4        1.3350             nan     0.1000    0.0945
##      5        1.2748             nan     0.1000    0.0713
##      6        1.2284             nan     0.1000    0.0717
##      7        1.1828             nan     0.1000    0.0681
##      8        1.1405             nan     0.1000    0.0629
##      9        1.1020             nan     0.1000    0.0488
##     10        1.0716             nan     0.1000    0.0459
##     20        0.7899             nan     0.1000    0.0297
##     40        0.5089             nan     0.1000    0.0207
##     60        0.3441             nan     0.1000    0.0111
##     80        0.2405             nan     0.1000    0.0039
##    100        0.1762             nan     0.1000    0.0049
##    120        0.1333             nan     0.1000    0.0028
##    140        0.1029             nan     0.1000    0.0011
##    150        0.0907             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2435
##      2        1.4567             nan     0.1000    0.1682
##      3        1.3514             nan     0.1000    0.1361
##      4        1.2648             nan     0.1000    0.1070
##      5        1.1977             nan     0.1000    0.0883
##      6        1.1425             nan     0.1000    0.0989
##      7        1.0822             nan     0.1000    0.0934
##      8        1.0248             nan     0.1000    0.0779
##      9        0.9777             nan     0.1000    0.0793
##     10        0.9294             nan     0.1000    0.0681
##     20        0.6210             nan     0.1000    0.0314
##     40        0.3235             nan     0.1000    0.0181
##     60        0.1901             nan     0.1000    0.0062
##     80        0.1197             nan     0.1000    0.0026
##    100        0.0832             nan     0.1000    0.0024
##    120        0.0591             nan     0.1000    0.0021
##    140        0.0422             nan     0.1000    0.0008
##    150        0.0371             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1263
##      2        1.5240             nan     0.1000    0.0898
##      3        1.4651             nan     0.1000    0.0668
##      4        1.4215             nan     0.1000    0.0545
##      5        1.3865             nan     0.1000    0.0476
##      6        1.3551             nan     0.1000    0.0477
##      7        1.3253             nan     0.1000    0.0371
##      8        1.3007             nan     0.1000    0.0380
##      9        1.2741             nan     0.1000    0.0368
##     10        1.2494             nan     0.1000    0.0355
##     20        1.0736             nan     0.1000    0.0196
##     40        0.8735             nan     0.1000    0.0107
##     60        0.7407             nan     0.1000    0.0091
##     80        0.6369             nan     0.1000    0.0063
##    100        0.5573             nan     0.1000    0.0055
##    120        0.4932             nan     0.1000    0.0033
##    140        0.4390             nan     0.1000    0.0030
##    150        0.4151             nan     0.1000    0.0034
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1947
##      2        1.4830             nan     0.1000    0.1325
##      3        1.3958             nan     0.1000    0.1105
##      4        1.3261             nan     0.1000    0.0884
##      5        1.2696             nan     0.1000    0.0746
##      6        1.2207             nan     0.1000    0.0655
##      7        1.1777             nan     0.1000    0.0704
##      8        1.1336             nan     0.1000    0.0583
##      9        1.0976             nan     0.1000    0.0550
##     10        1.0636             nan     0.1000    0.0477
##     20        0.7934             nan     0.1000    0.0304
##     40        0.5057             nan     0.1000    0.0140
##     60        0.3348             nan     0.1000    0.0098
##     80        0.2413             nan     0.1000    0.0073
##    100        0.1757             nan     0.1000    0.0034
##    120        0.1336             nan     0.1000    0.0019
##    140        0.1051             nan     0.1000    0.0020
##    150        0.0921             nan     0.1000    0.0027
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2395
##      2        1.4564             nan     0.1000    0.1796
##      3        1.3433             nan     0.1000    0.1326
##      4        1.2604             nan     0.1000    0.1144
##      5        1.1894             nan     0.1000    0.0989
##      6        1.1283             nan     0.1000    0.0956
##      7        1.0704             nan     0.1000    0.0873
##      8        1.0171             nan     0.1000    0.0772
##      9        0.9692             nan     0.1000    0.0593
##     10        0.9318             nan     0.1000    0.0789
##     20        0.6156             nan     0.1000    0.0405
##     40        0.3201             nan     0.1000    0.0141
##     60        0.1874             nan     0.1000    0.0055
##     80        0.1218             nan     0.1000    0.0051
##    100        0.0824             nan     0.1000    0.0018
##    120        0.0589             nan     0.1000    0.0013
##    140        0.0437             nan     0.1000    0.0004
##    150        0.0386             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1300
##      2        1.5210             nan     0.1000    0.0892
##      3        1.4614             nan     0.1000    0.0693
##      4        1.4164             nan     0.1000    0.0587
##      5        1.3780             nan     0.1000    0.0494
##      6        1.3461             nan     0.1000    0.0475
##      7        1.3158             nan     0.1000    0.0386
##      8        1.2906             nan     0.1000    0.0386
##      9        1.2669             nan     0.1000    0.0352
##     10        1.2442             nan     0.1000    0.0346
##     20        1.0717             nan     0.1000    0.0215
##     40        0.8682             nan     0.1000    0.0143
##     60        0.7329             nan     0.1000    0.0082
##     80        0.6322             nan     0.1000    0.0059
##    100        0.5548             nan     0.1000    0.0043
##    120        0.4905             nan     0.1000    0.0034
##    140        0.4350             nan     0.1000    0.0037
##    150        0.4123             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1906
##      2        1.4870             nan     0.1000    0.1426
##      3        1.3964             nan     0.1000    0.1069
##      4        1.3269             nan     0.1000    0.0859
##      5        1.2717             nan     0.1000    0.0716
##      6        1.2253             nan     0.1000    0.0753
##      7        1.1776             nan     0.1000    0.0648
##      8        1.1375             nan     0.1000    0.0566
##      9        1.1023             nan     0.1000    0.0507
##     10        1.0700             nan     0.1000    0.0560
##     20        0.8002             nan     0.1000    0.0333
##     40        0.5092             nan     0.1000    0.0148
##     60        0.3470             nan     0.1000    0.0116
##     80        0.2514             nan     0.1000    0.0070
##    100        0.1843             nan     0.1000    0.0039
##    120        0.1397             nan     0.1000    0.0033
##    140        0.1053             nan     0.1000    0.0018
##    150        0.0947             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2513
##      2        1.4524             nan     0.1000    0.1673
##      3        1.3472             nan     0.1000    0.1392
##      4        1.2623             nan     0.1000    0.1115
##      5        1.1928             nan     0.1000    0.0966
##      6        1.1321             nan     0.1000    0.0873
##      7        1.0776             nan     0.1000    0.0985
##      8        1.0182             nan     0.1000    0.0883
##      9        0.9602             nan     0.1000    0.0579
##     10        0.9235             nan     0.1000    0.0814
##     20        0.5986             nan     0.1000    0.0286
##     40        0.3177             nan     0.1000    0.0195
##     60        0.1926             nan     0.1000    0.0104
##     80        0.1205             nan     0.1000    0.0028
##    100        0.0821             nan     0.1000    0.0016
##    120        0.0585             nan     0.1000    0.0010
##    140        0.0428             nan     0.1000    0.0013
##    150        0.0369             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1334
##      2        1.5228             nan     0.1000    0.0871
##      3        1.4650             nan     0.1000    0.0684
##      4        1.4211             nan     0.1000    0.0554
##      5        1.3852             nan     0.1000    0.0490
##      6        1.3541             nan     0.1000    0.0443
##      7        1.3251             nan     0.1000    0.0427
##      8        1.2981             nan     0.1000    0.0364
##      9        1.2736             nan     0.1000    0.0340
##     10        1.2521             nan     0.1000    0.0372
##     20        1.0797             nan     0.1000    0.0226
##     40        0.8748             nan     0.1000    0.0122
##     60        0.7384             nan     0.1000    0.0119
##     80        0.6362             nan     0.1000    0.0073
##    100        0.5527             nan     0.1000    0.0048
##    120        0.4884             nan     0.1000    0.0050
##    140        0.4330             nan     0.1000    0.0033
##    150        0.4098             nan     0.1000    0.0027
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1883
##      2        1.4868             nan     0.1000    0.1370
##      3        1.4008             nan     0.1000    0.1129
##      4        1.3277             nan     0.1000    0.0888
##      5        1.2715             nan     0.1000    0.0698
##      6        1.2254             nan     0.1000    0.0805
##      7        1.1758             nan     0.1000    0.0695
##      8        1.1334             nan     0.1000    0.0627
##      9        1.0958             nan     0.1000    0.0600
##     10        1.0594             nan     0.1000    0.0435
##     20        0.8062             nan     0.1000    0.0365
##     40        0.5123             nan     0.1000    0.0120
##     60        0.3496             nan     0.1000    0.0137
##     80        0.2489             nan     0.1000    0.0067
##    100        0.1776             nan     0.1000    0.0044
##    120        0.1390             nan     0.1000    0.0023
##    140        0.1046             nan     0.1000    0.0014
##    150        0.0945             nan     0.1000    0.0024
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2394
##      2        1.4577             nan     0.1000    0.1722
##      3        1.3483             nan     0.1000    0.1408
##      4        1.2619             nan     0.1000    0.1155
##      5        1.1895             nan     0.1000    0.0891
##      6        1.1325             nan     0.1000    0.0888
##      7        1.0774             nan     0.1000    0.0987
##      8        1.0172             nan     0.1000    0.0735
##      9        0.9717             nan     0.1000    0.0646
##     10        0.9317             nan     0.1000    0.0765
##     20        0.6136             nan     0.1000    0.0321
##     40        0.3208             nan     0.1000    0.0130
##     60        0.1912             nan     0.1000    0.0077
##     80        0.1256             nan     0.1000    0.0034
##    100        0.0841             nan     0.1000    0.0021
##    120        0.0588             nan     0.1000    0.0013
##    140        0.0431             nan     0.1000    0.0010
##    150        0.0370             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1271
##      2        1.5229             nan     0.1000    0.0865
##      3        1.4654             nan     0.1000    0.0667
##      4        1.4211             nan     0.1000    0.0554
##      5        1.3848             nan     0.1000    0.0518
##      6        1.3517             nan     0.1000    0.0385
##      7        1.3259             nan     0.1000    0.0440
##      8        1.2967             nan     0.1000    0.0406
##      9        1.2713             nan     0.1000    0.0344
##     10        1.2486             nan     0.1000    0.0332
##     20        1.0763             nan     0.1000    0.0231
##     40        0.8731             nan     0.1000    0.0104
##     60        0.7404             nan     0.1000    0.0071
##     80        0.6382             nan     0.1000    0.0076
##    100        0.5543             nan     0.1000    0.0041
##    120        0.4920             nan     0.1000    0.0048
##    140        0.4368             nan     0.1000    0.0034
##    150        0.4119             nan     0.1000    0.0034
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1940
##      2        1.4831             nan     0.1000    0.1290
##      3        1.3979             nan     0.1000    0.1114
##      4        1.3278             nan     0.1000    0.0847
##      5        1.2718             nan     0.1000    0.0805
##      6        1.2206             nan     0.1000    0.0743
##      7        1.1740             nan     0.1000    0.0669
##      8        1.1336             nan     0.1000    0.0660
##      9        1.0939             nan     0.1000    0.0603
##     10        1.0569             nan     0.1000    0.0488
##     20        0.7907             nan     0.1000    0.0312
##     40        0.5076             nan     0.1000    0.0121
##     60        0.3433             nan     0.1000    0.0069
##     80        0.2445             nan     0.1000    0.0064
##    100        0.1832             nan     0.1000    0.0047
##    120        0.1378             nan     0.1000    0.0040
##    140        0.1056             nan     0.1000    0.0025
##    150        0.0918             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2423
##      2        1.4545             nan     0.1000    0.1698
##      3        1.3479             nan     0.1000    0.1315
##      4        1.2646             nan     0.1000    0.1119
##      5        1.1932             nan     0.1000    0.1003
##      6        1.1311             nan     0.1000    0.0785
##      7        1.0811             nan     0.1000    0.1062
##      8        1.0177             nan     0.1000    0.0849
##      9        0.9660             nan     0.1000    0.0698
##     10        0.9239             nan     0.1000    0.0751
##     20        0.5945             nan     0.1000    0.0378
##     40        0.3265             nan     0.1000    0.0115
##     60        0.1929             nan     0.1000    0.0090
##     80        0.1196             nan     0.1000    0.0039
##    100        0.0805             nan     0.1000    0.0025
##    120        0.0559             nan     0.1000    0.0015
##    140        0.0404             nan     0.1000    0.0007
##    150        0.0353             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1262
##      2        1.5261             nan     0.1000    0.0820
##      3        1.4709             nan     0.1000    0.0655
##      4        1.4274             nan     0.1000    0.0529
##      5        1.3930             nan     0.1000    0.0508
##      6        1.3604             nan     0.1000    0.0433
##      7        1.3327             nan     0.1000    0.0395
##      8        1.3076             nan     0.1000    0.0400
##      9        1.2795             nan     0.1000    0.0343
##     10        1.2584             nan     0.1000    0.0367
##     20        1.0860             nan     0.1000    0.0168
##     40        0.8859             nan     0.1000    0.0131
##     60        0.7497             nan     0.1000    0.0088
##     80        0.6478             nan     0.1000    0.0074
##    100        0.5671             nan     0.1000    0.0057
##    120        0.4987             nan     0.1000    0.0042
##    140        0.4426             nan     0.1000    0.0040
##    150        0.4184             nan     0.1000    0.0033
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1910
##      2        1.4850             nan     0.1000    0.1314
##      3        1.3997             nan     0.1000    0.1022
##      4        1.3344             nan     0.1000    0.0909
##      5        1.2764             nan     0.1000    0.0695
##      6        1.2315             nan     0.1000    0.0796
##      7        1.1824             nan     0.1000    0.0550
##      8        1.1469             nan     0.1000    0.0621
##      9        1.1096             nan     0.1000    0.0566
##     10        1.0754             nan     0.1000    0.0552
##     20        0.8067             nan     0.1000    0.0323
##     40        0.4966             nan     0.1000    0.0167
##     60        0.3420             nan     0.1000    0.0077
##     80        0.2447             nan     0.1000    0.0068
##    100        0.1828             nan     0.1000    0.0027
##    120        0.1377             nan     0.1000    0.0019
##    140        0.1065             nan     0.1000    0.0025
##    150        0.0919             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2333
##      2        1.4582             nan     0.1000    0.1725
##      3        1.3489             nan     0.1000    0.1302
##      4        1.2664             nan     0.1000    0.1008
##      5        1.2029             nan     0.1000    0.0973
##      6        1.1421             nan     0.1000    0.0986
##      7        1.0819             nan     0.1000    0.1030
##      8        1.0190             nan     0.1000    0.0764
##      9        0.9717             nan     0.1000    0.0731
##     10        0.9280             nan     0.1000    0.0580
##     20        0.6172             nan     0.1000    0.0454
##     40        0.3259             nan     0.1000    0.0173
##     60        0.1944             nan     0.1000    0.0075
##     80        0.1208             nan     0.1000    0.0044
##    100        0.0832             nan     0.1000    0.0028
##    120        0.0593             nan     0.1000    0.0007
##    140        0.0428             nan     0.1000    0.0023
##    150        0.0359             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1280
##      2        1.5238             nan     0.1000    0.0856
##      3        1.4670             nan     0.1000    0.0684
##      4        1.4219             nan     0.1000    0.0521
##      5        1.3872             nan     0.1000    0.0495
##      6        1.3549             nan     0.1000    0.0459
##      7        1.3265             nan     0.1000    0.0416
##      8        1.3008             nan     0.1000    0.0342
##      9        1.2789             nan     0.1000    0.0383
##     10        1.2527             nan     0.1000    0.0307
##     20        1.0801             nan     0.1000    0.0199
##     40        0.8772             nan     0.1000    0.0106
##     60        0.7443             nan     0.1000    0.0101
##     80        0.6391             nan     0.1000    0.0074
##    100        0.5585             nan     0.1000    0.0044
##    120        0.4922             nan     0.1000    0.0040
##    140        0.4400             nan     0.1000    0.0040
##    150        0.4151             nan     0.1000    0.0025
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1819
##      2        1.4902             nan     0.1000    0.1408
##      3        1.4013             nan     0.1000    0.1085
##      4        1.3317             nan     0.1000    0.0851
##      5        1.2780             nan     0.1000    0.0802
##      6        1.2261             nan     0.1000    0.0721
##      7        1.1818             nan     0.1000    0.0654
##      8        1.1410             nan     0.1000    0.0676
##      9        1.1000             nan     0.1000    0.0573
##     10        1.0653             nan     0.1000    0.0502
##     20        0.8046             nan     0.1000    0.0322
##     40        0.5100             nan     0.1000    0.0169
##     60        0.3482             nan     0.1000    0.0111
##     80        0.2500             nan     0.1000    0.0047
##    100        0.1828             nan     0.1000    0.0042
##    120        0.1389             nan     0.1000    0.0041
##    140        0.1061             nan     0.1000    0.0030
##    150        0.0931             nan     0.1000    0.0024
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2411
##      2        1.4562             nan     0.1000    0.1718
##      3        1.3481             nan     0.1000    0.1331
##      4        1.2645             nan     0.1000    0.1143
##      5        1.1931             nan     0.1000    0.0936
##      6        1.1341             nan     0.1000    0.0894
##      7        1.0791             nan     0.1000    0.0816
##      8        1.0287             nan     0.1000    0.0789
##      9        0.9802             nan     0.1000    0.0758
##     10        0.9348             nan     0.1000    0.0642
##     20        0.6156             nan     0.1000    0.0330
##     40        0.3165             nan     0.1000    0.0111
##     60        0.1923             nan     0.1000    0.0081
##     80        0.1253             nan     0.1000    0.0057
##    100        0.0847             nan     0.1000    0.0023
##    120        0.0603             nan     0.1000    0.0014
##    140        0.0444             nan     0.1000    0.0012
##    150        0.0379             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1270
##      2        1.5259             nan     0.1000    0.0860
##      3        1.4690             nan     0.1000    0.0639
##      4        1.4261             nan     0.1000    0.0526
##      5        1.3920             nan     0.1000    0.0467
##      6        1.3620             nan     0.1000    0.0475
##      7        1.3325             nan     0.1000    0.0419
##      8        1.3067             nan     0.1000    0.0320
##      9        1.2851             nan     0.1000    0.0413
##     10        1.2577             nan     0.1000    0.0359
##     20        1.0859             nan     0.1000    0.0236
##     40        0.8791             nan     0.1000    0.0135
##     60        0.7443             nan     0.1000    0.0090
##     80        0.6396             nan     0.1000    0.0073
##    100        0.5586             nan     0.1000    0.0042
##    120        0.4932             nan     0.1000    0.0049
##    140        0.4385             nan     0.1000    0.0036
##    150        0.4146             nan     0.1000    0.0027
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1771
##      2        1.4923             nan     0.1000    0.1362
##      3        1.4057             nan     0.1000    0.1168
##      4        1.3316             nan     0.1000    0.0960
##      5        1.2716             nan     0.1000    0.0720
##      6        1.2246             nan     0.1000    0.0777
##      7        1.1769             nan     0.1000    0.0678
##      8        1.1343             nan     0.1000    0.0597
##      9        1.0968             nan     0.1000    0.0595
##     10        1.0601             nan     0.1000    0.0471
##     20        0.7983             nan     0.1000    0.0288
##     40        0.5063             nan     0.1000    0.0119
##     60        0.3429             nan     0.1000    0.0105
##     80        0.2547             nan     0.1000    0.0046
##    100        0.1897             nan     0.1000    0.0083
##    120        0.1408             nan     0.1000    0.0020
##    140        0.1093             nan     0.1000    0.0014
##    150        0.0984             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2302
##      2        1.4625             nan     0.1000    0.1689
##      3        1.3556             nan     0.1000    0.1356
##      4        1.2697             nan     0.1000    0.1194
##      5        1.1951             nan     0.1000    0.0877
##      6        1.1392             nan     0.1000    0.1011
##      7        1.0775             nan     0.1000    0.0856
##      8        1.0244             nan     0.1000    0.0750
##      9        0.9772             nan     0.1000    0.0607
##     10        0.9400             nan     0.1000    0.0724
##     20        0.6073             nan     0.1000    0.0378
##     40        0.3306             nan     0.1000    0.0159
##     60        0.1969             nan     0.1000    0.0087
##     80        0.1241             nan     0.1000    0.0030
##    100        0.0838             nan     0.1000    0.0018
##    120        0.0612             nan     0.1000    0.0012
##    140        0.0453             nan     0.1000    0.0008
##    150        0.0386             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1309
##      2        1.5237             nan     0.1000    0.0896
##      3        1.4636             nan     0.1000    0.0682
##      4        1.4187             nan     0.1000    0.0535
##      5        1.3836             nan     0.1000    0.0454
##      6        1.3533             nan     0.1000    0.0466
##      7        1.3245             nan     0.1000    0.0445
##      8        1.2967             nan     0.1000    0.0423
##      9        1.2690             nan     0.1000    0.0335
##     10        1.2459             nan     0.1000    0.0349
##     20        1.0704             nan     0.1000    0.0219
##     40        0.8704             nan     0.1000    0.0124
##     60        0.7343             nan     0.1000    0.0090
##     80        0.6321             nan     0.1000    0.0056
##    100        0.5536             nan     0.1000    0.0052
##    120        0.4891             nan     0.1000    0.0057
##    140        0.4346             nan     0.1000    0.0035
##    150        0.4104             nan     0.1000    0.0025
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1897
##      2        1.4853             nan     0.1000    0.1399
##      3        1.3960             nan     0.1000    0.1124
##      4        1.3252             nan     0.1000    0.0876
##      5        1.2692             nan     0.1000    0.0776
##      6        1.2201             nan     0.1000    0.0787
##      7        1.1716             nan     0.1000    0.0707
##      8        1.1288             nan     0.1000    0.0649
##      9        1.0881             nan     0.1000    0.0600
##     10        1.0524             nan     0.1000    0.0445
##     20        0.7918             nan     0.1000    0.0272
##     40        0.5023             nan     0.1000    0.0204
##     60        0.3423             nan     0.1000    0.0089
##     80        0.2461             nan     0.1000    0.0082
##    100        0.1826             nan     0.1000    0.0029
##    120        0.1382             nan     0.1000    0.0015
##    140        0.1044             nan     0.1000    0.0016
##    150        0.0924             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2422
##      2        1.4541             nan     0.1000    0.1801
##      3        1.3422             nan     0.1000    0.1371
##      4        1.2577             nan     0.1000    0.1164
##      5        1.1855             nan     0.1000    0.0958
##      6        1.1261             nan     0.1000    0.0823
##      7        1.0744             nan     0.1000    0.0924
##      8        1.0186             nan     0.1000    0.0789
##      9        0.9709             nan     0.1000    0.0751
##     10        0.9265             nan     0.1000    0.0662
##     20        0.6186             nan     0.1000    0.0290
##     40        0.3213             nan     0.1000    0.0181
##     60        0.1915             nan     0.1000    0.0070
##     80        0.1197             nan     0.1000    0.0037
##    100        0.0829             nan     0.1000    0.0029
##    120        0.0582             nan     0.1000    0.0007
##    140        0.0441             nan     0.1000    0.0016
##    150        0.0380             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1275
##      2        1.5243             nan     0.1000    0.0853
##      3        1.4663             nan     0.1000    0.0669
##      4        1.4232             nan     0.1000    0.0540
##      5        1.3870             nan     0.1000    0.0450
##      6        1.3579             nan     0.1000    0.0490
##      7        1.3275             nan     0.1000    0.0435
##      8        1.3015             nan     0.1000    0.0425
##      9        1.2732             nan     0.1000    0.0290
##     10        1.2534             nan     0.1000    0.0317
##     20        1.0838             nan     0.1000    0.0218
##     40        0.8771             nan     0.1000    0.0155
##     60        0.7406             nan     0.1000    0.0073
##     80        0.6383             nan     0.1000    0.0071
##    100        0.5582             nan     0.1000    0.0040
##    120        0.4930             nan     0.1000    0.0050
##    140        0.4371             nan     0.1000    0.0031
##    150        0.4134             nan     0.1000    0.0039
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1961
##      2        1.4826             nan     0.1000    0.1298
##      3        1.3978             nan     0.1000    0.1089
##      4        1.3299             nan     0.1000    0.0917
##      5        1.2721             nan     0.1000    0.0746
##      6        1.2249             nan     0.1000    0.0850
##      7        1.1738             nan     0.1000    0.0646
##      8        1.1341             nan     0.1000    0.0694
##      9        1.0917             nan     0.1000    0.0513
##     10        1.0597             nan     0.1000    0.0483
##     20        0.7941             nan     0.1000    0.0386
##     40        0.5090             nan     0.1000    0.0125
##     60        0.3503             nan     0.1000    0.0132
##     80        0.2478             nan     0.1000    0.0079
##    100        0.1832             nan     0.1000    0.0029
##    120        0.1410             nan     0.1000    0.0034
##    140        0.1082             nan     0.1000    0.0017
##    150        0.0962             nan     0.1000    0.0032
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2453
##      2        1.4538             nan     0.1000    0.1681
##      3        1.3499             nan     0.1000    0.1276
##      4        1.2693             nan     0.1000    0.1205
##      5        1.1943             nan     0.1000    0.0928
##      6        1.1360             nan     0.1000    0.1083
##      7        1.0704             nan     0.1000    0.0976
##      8        1.0110             nan     0.1000    0.0760
##      9        0.9643             nan     0.1000    0.0797
##     10        0.9111             nan     0.1000    0.0702
##     20        0.6102             nan     0.1000    0.0375
##     40        0.3193             nan     0.1000    0.0119
##     60        0.1920             nan     0.1000    0.0106
##     80        0.1190             nan     0.1000    0.0043
##    100        0.0825             nan     0.1000    0.0031
##    120        0.0592             nan     0.1000    0.0012
##    140        0.0432             nan     0.1000    0.0004
##    150        0.0376             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1269
##      2        1.5258             nan     0.1000    0.0860
##      3        1.4693             nan     0.1000    0.0664
##      4        1.4263             nan     0.1000    0.0555
##      5        1.3906             nan     0.1000    0.0487
##      6        1.3595             nan     0.1000    0.0479
##      7        1.3299             nan     0.1000    0.0464
##      8        1.2995             nan     0.1000    0.0367
##      9        1.2760             nan     0.1000    0.0352
##     10        1.2525             nan     0.1000    0.0323
##     20        1.0794             nan     0.1000    0.0213
##     40        0.8765             nan     0.1000    0.0139
##     60        0.7412             nan     0.1000    0.0103
##     80        0.6344             nan     0.1000    0.0063
##    100        0.5525             nan     0.1000    0.0047
##    120        0.4867             nan     0.1000    0.0056
##    140        0.4331             nan     0.1000    0.0040
##    150        0.4093             nan     0.1000    0.0025
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1879
##      2        1.4885             nan     0.1000    0.1439
##      3        1.3981             nan     0.1000    0.1056
##      4        1.3312             nan     0.1000    0.0899
##      5        1.2730             nan     0.1000    0.0860
##      6        1.2198             nan     0.1000    0.0798
##      7        1.1709             nan     0.1000    0.0716
##      8        1.1266             nan     0.1000    0.0664
##      9        1.0853             nan     0.1000    0.0472
##     10        1.0556             nan     0.1000    0.0455
##     20        0.7938             nan     0.1000    0.0303
##     40        0.5072             nan     0.1000    0.0148
##     60        0.3537             nan     0.1000    0.0120
##     80        0.2503             nan     0.1000    0.0053
##    100        0.1852             nan     0.1000    0.0040
##    120        0.1423             nan     0.1000    0.0041
##    140        0.1120             nan     0.1000    0.0020
##    150        0.0977             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2391
##      2        1.4572             nan     0.1000    0.1749
##      3        1.3464             nan     0.1000    0.1280
##      4        1.2639             nan     0.1000    0.1198
##      5        1.1904             nan     0.1000    0.0979
##      6        1.1290             nan     0.1000    0.0894
##      7        1.0730             nan     0.1000    0.0816
##      8        1.0223             nan     0.1000    0.0781
##      9        0.9747             nan     0.1000    0.0756
##     10        0.9283             nan     0.1000    0.0645
##     20        0.6061             nan     0.1000    0.0277
##     40        0.3228             nan     0.1000    0.0109
##     60        0.1927             nan     0.1000    0.0065
##     80        0.1218             nan     0.1000    0.0030
##    100        0.0840             nan     0.1000    0.0037
##    120        0.0606             nan     0.1000    0.0024
##    140        0.0443             nan     0.1000    0.0014
##    150        0.0384             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1314
##      2        1.5215             nan     0.1000    0.0859
##      3        1.4642             nan     0.1000    0.0676
##      4        1.4200             nan     0.1000    0.0539
##      5        1.3847             nan     0.1000    0.0516
##      6        1.3515             nan     0.1000    0.0468
##      7        1.3218             nan     0.1000    0.0377
##      8        1.2952             nan     0.1000    0.0365
##      9        1.2719             nan     0.1000    0.0387
##     10        1.2479             nan     0.1000    0.0363
##     20        1.0744             nan     0.1000    0.0198
##     40        0.8752             nan     0.1000    0.0106
##     60        0.7402             nan     0.1000    0.0095
##     80        0.6357             nan     0.1000    0.0061
##    100        0.5551             nan     0.1000    0.0059
##    120        0.4914             nan     0.1000    0.0027
##    140        0.4385             nan     0.1000    0.0034
##    150        0.4133             nan     0.1000    0.0025
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1921
##      2        1.4849             nan     0.1000    0.1397
##      3        1.3951             nan     0.1000    0.1088
##      4        1.3254             nan     0.1000    0.0970
##      5        1.2650             nan     0.1000    0.0781
##      6        1.2156             nan     0.1000    0.0655
##      7        1.1737             nan     0.1000    0.0663
##      8        1.1319             nan     0.1000    0.0617
##      9        1.0936             nan     0.1000    0.0568
##     10        1.0588             nan     0.1000    0.0526
##     20        0.7945             nan     0.1000    0.0273
##     40        0.5065             nan     0.1000    0.0207
##     60        0.3437             nan     0.1000    0.0100
##     80        0.2520             nan     0.1000    0.0070
##    100        0.1844             nan     0.1000    0.0031
##    120        0.1370             nan     0.1000    0.0022
##    140        0.1067             nan     0.1000    0.0023
##    150        0.0927             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2352
##      2        1.4572             nan     0.1000    0.1777
##      3        1.3462             nan     0.1000    0.1342
##      4        1.2623             nan     0.1000    0.1130
##      5        1.1908             nan     0.1000    0.1018
##      6        1.1266             nan     0.1000    0.1086
##      7        1.0616             nan     0.1000    0.0831
##      8        1.0104             nan     0.1000    0.0668
##      9        0.9686             nan     0.1000    0.0825
##     10        0.9192             nan     0.1000    0.0686
##     20        0.6102             nan     0.1000    0.0304
##     40        0.3194             nan     0.1000    0.0130
##     60        0.1893             nan     0.1000    0.0044
##     80        0.1224             nan     0.1000    0.0024
##    100        0.0863             nan     0.1000    0.0020
##    120        0.0638             nan     0.1000    0.0020
##    140        0.0470             nan     0.1000    0.0007
##    150        0.0406             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1333
##      2        1.5215             nan     0.1000    0.0845
##      3        1.4639             nan     0.1000    0.0721
##      4        1.4171             nan     0.1000    0.0548
##      5        1.3810             nan     0.1000    0.0512
##      6        1.3490             nan     0.1000    0.0450
##      7        1.3198             nan     0.1000    0.0369
##      8        1.2961             nan     0.1000    0.0411
##      9        1.2683             nan     0.1000    0.0395
##     10        1.2425             nan     0.1000    0.0328
##     20        1.0715             nan     0.1000    0.0213
##     40        0.8672             nan     0.1000    0.0110
##     60        0.7333             nan     0.1000    0.0082
##     80        0.6302             nan     0.1000    0.0074
##    100        0.5501             nan     0.1000    0.0059
##    120        0.4857             nan     0.1000    0.0040
##    140        0.4318             nan     0.1000    0.0047
##    150        0.4072             nan     0.1000    0.0033
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1990
##      2        1.4813             nan     0.1000    0.1385
##      3        1.3937             nan     0.1000    0.1078
##      4        1.3249             nan     0.1000    0.0905
##      5        1.2655             nan     0.1000    0.0717
##      6        1.2190             nan     0.1000    0.0745
##      7        1.1718             nan     0.1000    0.0744
##      8        1.1272             nan     0.1000    0.0590
##      9        1.0912             nan     0.1000    0.0537
##     10        1.0583             nan     0.1000    0.0517
##     20        0.7893             nan     0.1000    0.0327
##     40        0.5146             nan     0.1000    0.0234
##     60        0.3486             nan     0.1000    0.0073
##     80        0.2491             nan     0.1000    0.0084
##    100        0.1829             nan     0.1000    0.0046
##    120        0.1380             nan     0.1000    0.0035
##    140        0.1076             nan     0.1000    0.0019
##    150        0.0932             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2454
##      2        1.4521             nan     0.1000    0.1678
##      3        1.3460             nan     0.1000    0.1442
##      4        1.2565             nan     0.1000    0.1105
##      5        1.1869             nan     0.1000    0.1041
##      6        1.1223             nan     0.1000    0.0925
##      7        1.0661             nan     0.1000    0.0811
##      8        1.0156             nan     0.1000    0.0796
##      9        0.9669             nan     0.1000    0.0773
##     10        0.9193             nan     0.1000    0.0643
##     20        0.6099             nan     0.1000    0.0298
##     40        0.3293             nan     0.1000    0.0136
##     60        0.1972             nan     0.1000    0.0096
##     80        0.1242             nan     0.1000    0.0032
##    100        0.0835             nan     0.1000    0.0035
##    120        0.0587             nan     0.1000    0.0014
##    140        0.0430             nan     0.1000    0.0013
##    150        0.0374             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1227
##      2        1.5262             nan     0.1000    0.0860
##      3        1.4693             nan     0.1000    0.0631
##      4        1.4276             nan     0.1000    0.0519
##      5        1.3931             nan     0.1000    0.0509
##      6        1.3606             nan     0.1000    0.0456
##      7        1.3316             nan     0.1000    0.0401
##      8        1.3061             nan     0.1000    0.0396
##      9        1.2794             nan     0.1000    0.0314
##     10        1.2576             nan     0.1000    0.0373
##     20        1.0821             nan     0.1000    0.0213
##     40        0.8783             nan     0.1000    0.0105
##     60        0.7394             nan     0.1000    0.0087
##     80        0.6357             nan     0.1000    0.0075
##    100        0.5519             nan     0.1000    0.0059
##    120        0.4878             nan     0.1000    0.0045
##    140        0.4339             nan     0.1000    0.0031
##    150        0.4113             nan     0.1000    0.0034
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1918
##      2        1.4871             nan     0.1000    0.1325
##      3        1.3995             nan     0.1000    0.1048
##      4        1.3310             nan     0.1000    0.0904
##      5        1.2734             nan     0.1000    0.0762
##      6        1.2249             nan     0.1000    0.0764
##      7        1.1770             nan     0.1000    0.0755
##      8        1.1306             nan     0.1000    0.0569
##      9        1.0947             nan     0.1000    0.0518
##     10        1.0625             nan     0.1000    0.0597
##     20        0.7888             nan     0.1000    0.0256
##     40        0.5149             nan     0.1000    0.0195
##     60        0.3463             nan     0.1000    0.0128
##     80        0.2395             nan     0.1000    0.0074
##    100        0.1785             nan     0.1000    0.0041
##    120        0.1311             nan     0.1000    0.0028
##    140        0.1011             nan     0.1000    0.0014
##    150        0.0901             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2334
##      2        1.4582             nan     0.1000    0.1707
##      3        1.3525             nan     0.1000    0.1382
##      4        1.2658             nan     0.1000    0.1076
##      5        1.1961             nan     0.1000    0.1000
##      6        1.1339             nan     0.1000    0.0842
##      7        1.0815             nan     0.1000    0.0965
##      8        1.0230             nan     0.1000    0.0838
##      9        0.9726             nan     0.1000    0.0702
##     10        0.9294             nan     0.1000    0.0737
##     20        0.6080             nan     0.1000    0.0378
##     40        0.3272             nan     0.1000    0.0212
##     60        0.1895             nan     0.1000    0.0058
##     80        0.1229             nan     0.1000    0.0045
##    100        0.0830             nan     0.1000    0.0009
##    120        0.0587             nan     0.1000    0.0016
##    140        0.0428             nan     0.1000    0.0010
##    150        0.0373             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1284
##      2        1.5253             nan     0.1000    0.0857
##      3        1.4682             nan     0.1000    0.0653
##      4        1.4251             nan     0.1000    0.0545
##      5        1.3897             nan     0.1000    0.0518
##      6        1.3569             nan     0.1000    0.0440
##      7        1.3292             nan     0.1000    0.0420
##      8        1.3031             nan     0.1000    0.0388
##      9        1.2763             nan     0.1000    0.0339
##     10        1.2540             nan     0.1000    0.0273
##     20        1.0821             nan     0.1000    0.0189
##     40        0.8768             nan     0.1000    0.0119
##     60        0.7382             nan     0.1000    0.0078
##     80        0.6356             nan     0.1000    0.0072
##    100        0.5533             nan     0.1000    0.0049
##    120        0.4890             nan     0.1000    0.0033
##    140        0.4348             nan     0.1000    0.0043
##    150        0.4109             nan     0.1000    0.0033
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1917
##      2        1.4856             nan     0.1000    0.1402
##      3        1.3981             nan     0.1000    0.1137
##      4        1.3256             nan     0.1000    0.0833
##      5        1.2717             nan     0.1000    0.0871
##      6        1.2179             nan     0.1000    0.0777
##      7        1.1702             nan     0.1000    0.0619
##      8        1.1312             nan     0.1000    0.0570
##      9        1.0961             nan     0.1000    0.0591
##     10        1.0607             nan     0.1000    0.0525
##     20        0.8028             nan     0.1000    0.0391
##     40        0.5069             nan     0.1000    0.0207
##     60        0.3448             nan     0.1000    0.0074
##     80        0.2465             nan     0.1000    0.0078
##    100        0.1798             nan     0.1000    0.0073
##    120        0.1367             nan     0.1000    0.0031
##    140        0.1037             nan     0.1000    0.0019
##    150        0.0931             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2424
##      2        1.4568             nan     0.1000    0.1723
##      3        1.3491             nan     0.1000    0.1377
##      4        1.2633             nan     0.1000    0.1117
##      5        1.1935             nan     0.1000    0.0954
##      6        1.1334             nan     0.1000    0.1051
##      7        1.0695             nan     0.1000    0.0875
##      8        1.0153             nan     0.1000    0.0688
##      9        0.9724             nan     0.1000    0.0836
##     10        0.9222             nan     0.1000    0.0674
##     20        0.6123             nan     0.1000    0.0301
##     40        0.3174             nan     0.1000    0.0118
##     60        0.1794             nan     0.1000    0.0047
##     80        0.1171             nan     0.1000    0.0039
##    100        0.0781             nan     0.1000    0.0025
##    120        0.0558             nan     0.1000    0.0017
##    140        0.0411             nan     0.1000    0.0009
##    150        0.0357             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1345
##      2        1.5221             nan     0.1000    0.0888
##      3        1.4632             nan     0.1000    0.0670
##      4        1.4185             nan     0.1000    0.0543
##      5        1.3828             nan     0.1000    0.0500
##      6        1.3505             nan     0.1000    0.0398
##      7        1.3244             nan     0.1000    0.0398
##      8        1.2993             nan     0.1000    0.0423
##      9        1.2708             nan     0.1000    0.0365
##     10        1.2479             nan     0.1000    0.0341
##     20        1.0741             nan     0.1000    0.0191
##     40        0.8693             nan     0.1000    0.0103
##     60        0.7348             nan     0.1000    0.0095
##     80        0.6326             nan     0.1000    0.0066
##    100        0.5530             nan     0.1000    0.0053
##    120        0.4892             nan     0.1000    0.0046
##    140        0.4340             nan     0.1000    0.0039
##    150        0.4098             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1857
##      2        1.4884             nan     0.1000    0.1445
##      3        1.3978             nan     0.1000    0.1107
##      4        1.3274             nan     0.1000    0.0905
##      5        1.2696             nan     0.1000    0.0832
##      6        1.2174             nan     0.1000    0.0716
##      7        1.1722             nan     0.1000    0.0671
##      8        1.1312             nan     0.1000    0.0576
##      9        1.0948             nan     0.1000    0.0529
##     10        1.0622             nan     0.1000    0.0467
##     20        0.7992             nan     0.1000    0.0401
##     40        0.5171             nan     0.1000    0.0188
##     60        0.3530             nan     0.1000    0.0170
##     80        0.2461             nan     0.1000    0.0085
##    100        0.1790             nan     0.1000    0.0036
##    120        0.1345             nan     0.1000    0.0030
##    140        0.1032             nan     0.1000    0.0016
##    150        0.0917             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2363
##      2        1.4578             nan     0.1000    0.1729
##      3        1.3510             nan     0.1000    0.1349
##      4        1.2647             nan     0.1000    0.1071
##      5        1.1969             nan     0.1000    0.0979
##      6        1.1349             nan     0.1000    0.1078
##      7        1.0692             nan     0.1000    0.0912
##      8        1.0155             nan     0.1000    0.0846
##      9        0.9635             nan     0.1000    0.0675
##     10        0.9214             nan     0.1000    0.0645
##     20        0.6075             nan     0.1000    0.0318
##     40        0.3258             nan     0.1000    0.0135
##     60        0.1971             nan     0.1000    0.0062
##     80        0.1250             nan     0.1000    0.0046
##    100        0.0838             nan     0.1000    0.0025
##    120        0.0597             nan     0.1000    0.0022
##    140        0.0446             nan     0.1000    0.0005
##    150        0.0390             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1309
##      2        1.5218             nan     0.1000    0.0916
##      3        1.4628             nan     0.1000    0.0698
##      4        1.4174             nan     0.1000    0.0536
##      5        1.3823             nan     0.1000    0.0512
##      6        1.3497             nan     0.1000    0.0477
##      7        1.3203             nan     0.1000    0.0398
##      8        1.2946             nan     0.1000    0.0404
##      9        1.2672             nan     0.1000    0.0354
##     10        1.2430             nan     0.1000    0.0341
##     20        1.0689             nan     0.1000    0.0227
##     40        0.8637             nan     0.1000    0.0125
##     60        0.7260             nan     0.1000    0.0090
##     80        0.6225             nan     0.1000    0.0050
##    100        0.5460             nan     0.1000    0.0051
##    120        0.4800             nan     0.1000    0.0033
##    140        0.4265             nan     0.1000    0.0037
##    150        0.4041             nan     0.1000    0.0029
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1925
##      2        1.4832             nan     0.1000    0.1380
##      3        1.3948             nan     0.1000    0.1176
##      4        1.3206             nan     0.1000    0.0902
##      5        1.2626             nan     0.1000    0.0874
##      6        1.2079             nan     0.1000    0.0746
##      7        1.1620             nan     0.1000    0.0645
##      8        1.1201             nan     0.1000    0.0614
##      9        1.0824             nan     0.1000    0.0509
##     10        1.0510             nan     0.1000    0.0422
##     20        0.7847             nan     0.1000    0.0258
##     40        0.5082             nan     0.1000    0.0143
##     60        0.3364             nan     0.1000    0.0121
##     80        0.2421             nan     0.1000    0.0068
##    100        0.1789             nan     0.1000    0.0039
##    120        0.1354             nan     0.1000    0.0015
##    140        0.1056             nan     0.1000    0.0020
##    150        0.0931             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2465
##      2        1.4540             nan     0.1000    0.1718
##      3        1.3432             nan     0.1000    0.1263
##      4        1.2643             nan     0.1000    0.1114
##      5        1.1941             nan     0.1000    0.0934
##      6        1.1345             nan     0.1000    0.0941
##      7        1.0753             nan     0.1000    0.0936
##      8        1.0179             nan     0.1000    0.0683
##      9        0.9756             nan     0.1000    0.0834
##     10        0.9251             nan     0.1000    0.0640
##     20        0.5988             nan     0.1000    0.0370
##     40        0.3194             nan     0.1000    0.0084
##     60        0.1900             nan     0.1000    0.0060
##     80        0.1211             nan     0.1000    0.0031
##    100        0.0831             nan     0.1000    0.0029
##    120        0.0589             nan     0.1000    0.0004
##    140        0.0445             nan     0.1000    0.0014
##    150        0.0391             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1242
##      2        1.5262             nan     0.1000    0.0870
##      3        1.4694             nan     0.1000    0.0658
##      4        1.4253             nan     0.1000    0.0546
##      5        1.3905             nan     0.1000    0.0509
##      6        1.3586             nan     0.1000    0.0460
##      7        1.3299             nan     0.1000    0.0432
##      8        1.3016             nan     0.1000    0.0363
##      9        1.2782             nan     0.1000    0.0309
##     10        1.2578             nan     0.1000    0.0355
##     20        1.0811             nan     0.1000    0.0183
##     40        0.8824             nan     0.1000    0.0134
##     60        0.7449             nan     0.1000    0.0088
##     80        0.6419             nan     0.1000    0.0071
##    100        0.5620             nan     0.1000    0.0045
##    120        0.4942             nan     0.1000    0.0035
##    140        0.4406             nan     0.1000    0.0031
##    150        0.4157             nan     0.1000    0.0037
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1845
##      2        1.4898             nan     0.1000    0.1397
##      3        1.4003             nan     0.1000    0.1057
##      4        1.3315             nan     0.1000    0.0905
##      5        1.2738             nan     0.1000    0.0782
##      6        1.2238             nan     0.1000    0.0734
##      7        1.1773             nan     0.1000    0.0630
##      8        1.1384             nan     0.1000    0.0574
##      9        1.1024             nan     0.1000    0.0544
##     10        1.0697             nan     0.1000    0.0497
##     20        0.7976             nan     0.1000    0.0414
##     40        0.5003             nan     0.1000    0.0211
##     60        0.3440             nan     0.1000    0.0125
##     80        0.2446             nan     0.1000    0.0051
##    100        0.1844             nan     0.1000    0.0074
##    120        0.1374             nan     0.1000    0.0023
##    140        0.1068             nan     0.1000    0.0017
##    150        0.0957             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2385
##      2        1.4580             nan     0.1000    0.1810
##      3        1.3450             nan     0.1000    0.1317
##      4        1.2635             nan     0.1000    0.1131
##      5        1.1925             nan     0.1000    0.1104
##      6        1.1245             nan     0.1000    0.1090
##      7        1.0597             nan     0.1000    0.0930
##      8        1.0036             nan     0.1000    0.0698
##      9        0.9593             nan     0.1000    0.0741
##     10        0.9143             nan     0.1000    0.0615
##     20        0.6092             nan     0.1000    0.0423
##     40        0.3254             nan     0.1000    0.0174
##     60        0.1920             nan     0.1000    0.0064
##     80        0.1245             nan     0.1000    0.0032
##    100        0.0842             nan     0.1000    0.0028
##    120        0.0591             nan     0.1000    0.0015
##    140        0.0435             nan     0.1000    0.0009
##    150        0.0380             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1267
##      2        1.5249             nan     0.1000    0.0863
##      3        1.4694             nan     0.1000    0.0638
##      4        1.4264             nan     0.1000    0.0533
##      5        1.3916             nan     0.1000    0.0488
##      6        1.3603             nan     0.1000    0.0386
##      7        1.3355             nan     0.1000    0.0503
##      8        1.3029             nan     0.1000    0.0367
##      9        1.2797             nan     0.1000    0.0359
##     10        1.2569             nan     0.1000    0.0338
##     20        1.0833             nan     0.1000    0.0201
##     40        0.8768             nan     0.1000    0.0143
##     60        0.7451             nan     0.1000    0.0099
##     80        0.6391             nan     0.1000    0.0068
##    100        0.5584             nan     0.1000    0.0049
##    120        0.4941             nan     0.1000    0.0049
##    140        0.4387             nan     0.1000    0.0031
##    150        0.4135             nan     0.1000    0.0032
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1921
##      2        1.4862             nan     0.1000    0.1357
##      3        1.3992             nan     0.1000    0.1046
##      4        1.3324             nan     0.1000    0.0915
##      5        1.2750             nan     0.1000    0.0762
##      6        1.2259             nan     0.1000    0.0813
##      7        1.1758             nan     0.1000    0.0623
##      8        1.1377             nan     0.1000    0.0696
##      9        1.0954             nan     0.1000    0.0561
##     10        1.0608             nan     0.1000    0.0502
##     20        0.7961             nan     0.1000    0.0291
##     40        0.4986             nan     0.1000    0.0165
##     60        0.3369             nan     0.1000    0.0082
##     80        0.2415             nan     0.1000    0.0068
##    100        0.1786             nan     0.1000    0.0035
##    120        0.1361             nan     0.1000    0.0035
##    140        0.1048             nan     0.1000    0.0018
##    150        0.0940             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2452
##      2        1.4546             nan     0.1000    0.1721
##      3        1.3471             nan     0.1000    0.1290
##      4        1.2661             nan     0.1000    0.1188
##      5        1.1922             nan     0.1000    0.0959
##      6        1.1329             nan     0.1000    0.0898
##      7        1.0780             nan     0.1000    0.0965
##      8        1.0210             nan     0.1000    0.0890
##      9        0.9678             nan     0.1000    0.0590
##     10        0.9304             nan     0.1000    0.0697
##     20        0.6064             nan     0.1000    0.0361
##     40        0.3273             nan     0.1000    0.0157
##     60        0.1928             nan     0.1000    0.0080
##     80        0.1221             nan     0.1000    0.0038
##    100        0.0821             nan     0.1000    0.0014
##    120        0.0581             nan     0.1000    0.0015
##    140        0.0439             nan     0.1000    0.0007
##    150        0.0380             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1319
##      2        1.5239             nan     0.1000    0.0857
##      3        1.4665             nan     0.1000    0.0669
##      4        1.4230             nan     0.1000    0.0540
##      5        1.3872             nan     0.1000    0.0509
##      6        1.3542             nan     0.1000    0.0472
##      7        1.3245             nan     0.1000    0.0364
##      8        1.3014             nan     0.1000    0.0361
##      9        1.2788             nan     0.1000    0.0294
##     10        1.2590             nan     0.1000    0.0363
##     20        1.0790             nan     0.1000    0.0221
##     40        0.8718             nan     0.1000    0.0126
##     60        0.7383             nan     0.1000    0.0098
##     80        0.6308             nan     0.1000    0.0062
##    100        0.5492             nan     0.1000    0.0064
##    120        0.4847             nan     0.1000    0.0033
##    140        0.4324             nan     0.1000    0.0036
##    150        0.4099             nan     0.1000    0.0042
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1935
##      2        1.4851             nan     0.1000    0.1375
##      3        1.3970             nan     0.1000    0.1063
##      4        1.3284             nan     0.1000    0.0900
##      5        1.2715             nan     0.1000    0.0694
##      6        1.2268             nan     0.1000    0.0797
##      7        1.1772             nan     0.1000    0.0660
##      8        1.1355             nan     0.1000    0.0575
##      9        1.0987             nan     0.1000    0.0570
##     10        1.0639             nan     0.1000    0.0527
##     20        0.7932             nan     0.1000    0.0206
##     40        0.5153             nan     0.1000    0.0152
##     60        0.3555             nan     0.1000    0.0103
##     80        0.2552             nan     0.1000    0.0083
##    100        0.1849             nan     0.1000    0.0053
##    120        0.1373             nan     0.1000    0.0026
##    140        0.1067             nan     0.1000    0.0023
##    150        0.0945             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2439
##      2        1.4551             nan     0.1000    0.1713
##      3        1.3471             nan     0.1000    0.1355
##      4        1.2613             nan     0.1000    0.1078
##      5        1.1905             nan     0.1000    0.0920
##      6        1.1321             nan     0.1000    0.0931
##      7        1.0741             nan     0.1000    0.0681
##      8        1.0308             nan     0.1000    0.0752
##      9        0.9846             nan     0.1000    0.0811
##     10        0.9361             nan     0.1000    0.0632
##     20        0.6044             nan     0.1000    0.0319
##     40        0.3291             nan     0.1000    0.0117
##     60        0.1905             nan     0.1000    0.0051
##     80        0.1227             nan     0.1000    0.0056
##    100        0.0819             nan     0.1000    0.0015
##    120        0.0589             nan     0.1000    0.0013
##    140        0.0418             nan     0.1000    0.0008
##    150        0.0358             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1268
##      2        1.5235             nan     0.1000    0.0855
##      3        1.4659             nan     0.1000    0.0663
##      4        1.4220             nan     0.1000    0.0559
##      5        1.3854             nan     0.1000    0.0493
##      6        1.3534             nan     0.1000    0.0435
##      7        1.3249             nan     0.1000    0.0382
##      8        1.3004             nan     0.1000    0.0410
##      9        1.2732             nan     0.1000    0.0358
##     10        1.2498             nan     0.1000    0.0313
##     20        1.0757             nan     0.1000    0.0242
##     40        0.8751             nan     0.1000    0.0113
##     60        0.7364             nan     0.1000    0.0097
##     80        0.6381             nan     0.1000    0.0091
##    100        0.5587             nan     0.1000    0.0051
##    120        0.4938             nan     0.1000    0.0047
##    140        0.4369             nan     0.1000    0.0025
##    150        0.4140             nan     0.1000    0.0034
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1901
##      2        1.4873             nan     0.1000    0.1380
##      3        1.3988             nan     0.1000    0.1149
##      4        1.3257             nan     0.1000    0.0860
##      5        1.2696             nan     0.1000    0.0720
##      6        1.2220             nan     0.1000    0.0800
##      7        1.1723             nan     0.1000    0.0700
##      8        1.1288             nan     0.1000    0.0632
##      9        1.0898             nan     0.1000    0.0621
##     10        1.0525             nan     0.1000    0.0443
##     20        0.7852             nan     0.1000    0.0332
##     40        0.4949             nan     0.1000    0.0128
##     60        0.3407             nan     0.1000    0.0070
##     80        0.2462             nan     0.1000    0.0056
##    100        0.1805             nan     0.1000    0.0047
##    120        0.1312             nan     0.1000    0.0023
##    140        0.1018             nan     0.1000    0.0015
##    150        0.0901             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2379
##      2        1.4560             nan     0.1000    0.1788
##      3        1.3444             nan     0.1000    0.1374
##      4        1.2595             nan     0.1000    0.1069
##      5        1.1920             nan     0.1000    0.0956
##      6        1.1321             nan     0.1000    0.0954
##      7        1.0737             nan     0.1000    0.0779
##      8        1.0248             nan     0.1000    0.0696
##      9        0.9823             nan     0.1000    0.0785
##     10        0.9346             nan     0.1000    0.0678
##     20        0.5949             nan     0.1000    0.0340
##     40        0.3281             nan     0.1000    0.0133
##     60        0.1961             nan     0.1000    0.0095
##     80        0.1239             nan     0.1000    0.0048
##    100        0.0845             nan     0.1000    0.0036
##    120        0.0591             nan     0.1000    0.0021
##    140        0.0420             nan     0.1000    0.0007
##    150        0.0359             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1272
##      2        1.5234             nan     0.1000    0.0861
##      3        1.4653             nan     0.1000    0.0693
##      4        1.4211             nan     0.1000    0.0534
##      5        1.3852             nan     0.1000    0.0484
##      6        1.3538             nan     0.1000    0.0481
##      7        1.3244             nan     0.1000    0.0402
##      8        1.2993             nan     0.1000    0.0367
##      9        1.2769             nan     0.1000    0.0373
##     10        1.2512             nan     0.1000    0.0355
##     20        1.0779             nan     0.1000    0.0209
##     40        0.8790             nan     0.1000    0.0117
##     60        0.7431             nan     0.1000    0.0079
##     80        0.6409             nan     0.1000    0.0079
##    100        0.5585             nan     0.1000    0.0048
##    120        0.4920             nan     0.1000    0.0042
##    140        0.4370             nan     0.1000    0.0041
##    150        0.4120             nan     0.1000    0.0028
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1886
##      2        1.4865             nan     0.1000    0.1367
##      3        1.3985             nan     0.1000    0.1123
##      4        1.3274             nan     0.1000    0.0880
##      5        1.2707             nan     0.1000    0.0792
##      6        1.2210             nan     0.1000    0.0713
##      7        1.1759             nan     0.1000    0.0674
##      8        1.1342             nan     0.1000    0.0626
##      9        1.0953             nan     0.1000    0.0537
##     10        1.0611             nan     0.1000    0.0508
##     20        0.7962             nan     0.1000    0.0382
##     40        0.5035             nan     0.1000    0.0182
##     60        0.3456             nan     0.1000    0.0082
##     80        0.2462             nan     0.1000    0.0051
##    100        0.1778             nan     0.1000    0.0041
##    120        0.1379             nan     0.1000    0.0038
##    140        0.1037             nan     0.1000    0.0015
##    150        0.0906             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2464
##      2        1.4543             nan     0.1000    0.1726
##      3        1.3470             nan     0.1000    0.1363
##      4        1.2623             nan     0.1000    0.1117
##      5        1.1922             nan     0.1000    0.0961
##      6        1.1320             nan     0.1000    0.1040
##      7        1.0686             nan     0.1000    0.0757
##      8        1.0215             nan     0.1000    0.0879
##      9        0.9677             nan     0.1000    0.0746
##     10        0.9231             nan     0.1000    0.0567
##     20        0.6072             nan     0.1000    0.0321
##     40        0.3240             nan     0.1000    0.0117
##     60        0.1950             nan     0.1000    0.0070
##     80        0.1252             nan     0.1000    0.0023
##    100        0.0838             nan     0.1000    0.0015
##    120        0.0602             nan     0.1000    0.0016
##    140        0.0435             nan     0.1000    0.0014
##    150        0.0368             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1299
##      2        1.5235             nan     0.1000    0.0906
##      3        1.4645             nan     0.1000    0.0663
##      4        1.4208             nan     0.1000    0.0522
##      5        1.3859             nan     0.1000    0.0543
##      6        1.3513             nan     0.1000    0.0422
##      7        1.3236             nan     0.1000    0.0444
##      8        1.2970             nan     0.1000    0.0413
##      9        1.2687             nan     0.1000    0.0338
##     10        1.2470             nan     0.1000    0.0341
##     20        1.0734             nan     0.1000    0.0211
##     40        0.8724             nan     0.1000    0.0138
##     60        0.7389             nan     0.1000    0.0087
##     80        0.6371             nan     0.1000    0.0071
##    100        0.5547             nan     0.1000    0.0070
##    120        0.4885             nan     0.1000    0.0050
##    140        0.4344             nan     0.1000    0.0034
##    150        0.4113             nan     0.1000    0.0029
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1905
##      2        1.4852             nan     0.1000    0.1382
##      3        1.3973             nan     0.1000    0.1079
##      4        1.3282             nan     0.1000    0.0941
##      5        1.2686             nan     0.1000    0.0729
##      6        1.2214             nan     0.1000    0.0805
##      7        1.1712             nan     0.1000    0.0666
##      8        1.1293             nan     0.1000    0.0610
##      9        1.0907             nan     0.1000    0.0479
##     10        1.0599             nan     0.1000    0.0519
##     20        0.7953             nan     0.1000    0.0403
##     40        0.4922             nan     0.1000    0.0149
##     60        0.3411             nan     0.1000    0.0060
##     80        0.2433             nan     0.1000    0.0072
##    100        0.1782             nan     0.1000    0.0041
##    120        0.1334             nan     0.1000    0.0030
##    140        0.1016             nan     0.1000    0.0015
##    150        0.0903             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2360
##      2        1.4582             nan     0.1000    0.1782
##      3        1.3457             nan     0.1000    0.1314
##      4        1.2640             nan     0.1000    0.1179
##      5        1.1913             nan     0.1000    0.0957
##      6        1.1322             nan     0.1000    0.0843
##      7        1.0795             nan     0.1000    0.0795
##      8        1.0303             nan     0.1000    0.0980
##      9        0.9717             nan     0.1000    0.0829
##     10        0.9223             nan     0.1000    0.0716
##     20        0.6091             nan     0.1000    0.0469
##     40        0.3165             nan     0.1000    0.0103
##     60        0.1929             nan     0.1000    0.0064
##     80        0.1251             nan     0.1000    0.0061
##    100        0.0834             nan     0.1000    0.0021
##    120        0.0602             nan     0.1000    0.0013
##    140        0.0438             nan     0.1000    0.0013
##    150        0.0385             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2403
##      2        1.4578             nan     0.1000    0.1654
##      3        1.3530             nan     0.1000    0.1238
##      4        1.2734             nan     0.1000    0.1093
##      5        1.2039             nan     0.1000    0.1020
##      6        1.1392             nan     0.1000    0.0941
##      7        1.0816             nan     0.1000    0.0889
##      8        1.0280             nan     0.1000    0.0896
##      9        0.9736             nan     0.1000    0.0639
##     10        0.9342             nan     0.1000    0.0642
##     20        0.6136             nan     0.1000    0.0358
##     40        0.3226             nan     0.1000    0.0109
##     60        0.1939             nan     0.1000    0.0087
##     80        0.1245             nan     0.1000    0.0050
##    100        0.0838             nan     0.1000    0.0030
##    120        0.0612             nan     0.1000    0.0010
##    140        0.0456             nan     0.1000    0.0009
##    150        0.0398             nan     0.1000    0.0007
```

```r
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
