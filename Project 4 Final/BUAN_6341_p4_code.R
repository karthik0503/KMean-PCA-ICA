
#############
#set Path
#############
setwd("E:/AML - BUAN 6341")

##############################
# load Required Packages
##############################

# install necessary packages if missing
if(!require("DMwR")){
  cat(" \n DMwR package not found.. Hence installing...")
  install.packages("DMwR")
}

if(!require("caret")){
  cat(" \n caret package not found.. Hence installing...")
  install.packages("caret")
}

if(!require("unbalanced")){
  cat(" \n unbalanced package not found.. Hence installing...")
  install.packages("unbalanced")
}

# ROC Curve: requires the ROCR package.
if(!require("ROCR")){
  cat(" \n ROCR package not found.. Hence installing...")
  install.packages("ROCR")
}

#Data partitioning and modeling

if(!require("dplyr")){
  cat(" \n dplyr package not found.. Hence installing...")
  install.packages("dplyr")
}
if(!require("outliers")){
  cat(" \n outliers package not found.. Hence installing...")
  install.packages("outliers")
}

#Plotting Area under curve
if(!require("AUC")){
  cat(" \n AUC package not found.. Hence installing...")
  install.packages("AUC")
}

#Expectation Maximization
if(!require("mclust")){
  cat(" \n mclust package not found.. Hence installing...")
  install.packages("mclust")
}


library(AUC)
library(outliers)
library(dplyr)
library(ROCR)
library(DMwR) #for SMOTE
library(caret)
library(caTools)
library(unbalanced)
library(mclust)

#############
#Clear All existing graphs and variables 
#############

graphics.off()
rm(list=ls(all=TRUE))

# load IBM Attrition dataset
df <- read.csv("Attrition.csv", header = TRUE)

#omit missing value
df <- na.omit(df)

#Dropping categorical variable with just 1 level
df = select(df, -EmployeeCount,-StandardHours,-Over18)

#Converting to categorical variables
df$Education <- as.factor(df$Education) 
df$EnvironmentSatisfaction <- as.factor(df$EnvironmentSatisfaction)
df$JobInvolvement <- as.factor(df$JobInvolvement)
df$JobLevel <- as.factor(df$JobLevel)
df$JobSatisfaction <- as.factor(df$JobSatisfaction)
df$PerformanceRating <- as.factor(df$PerformanceRating)
df$RelationshipSatisfaction <- as.factor(df$RelationshipSatisfaction)
df$StockOptionLevel <- as.factor(df$StockOptionLevel)
df$TrainingTimesLastYear <- as.factor(df$TrainingTimesLastYear)
df$WorkLifeBalance <- as.factor(df$WorkLifeBalance)

#Scaling Continous Features
if(!require("BBmisc")){
  cat(" \n BBmisc package not found.. Hence installing...")
  install.packages("BBmisc")
}
library(BBmisc)
nums <- sapply(df, is.numeric)
numeric <- df[,nums]

#Finding the categorical variables
categ <- sapply(df, is.factor)
categorical <- df[,categ]


#Dummy variable conversion
X <- select(categorical,-Attrition)
dmy <- dummyVars(" ~ . ", data = X ) 
df3 <- data.frame(predict(dmy, newdata = X))
str(df3)


#combining Categorical and Numeric
df5 <- cbind.data.frame(df3, numeric)

#Scaling whole Dataset
scaled_df <- normalize(df5, method = "standardize", range = c(0, 1))

df6 <-  cbind.data.frame(scaled_df, df$Attrition)
colnames(df6)[86] <- "Attrition"

levels(df6$Attrition) <- c(0, 1)
data<-ubBalance(X= df6[1:85], Y=df6$Attrition, type="ubSMOTE", percOver=300, percUnder=150, verbose=TRUE)
dt1_smoted<-cbind(data$X,data$Y)
colnames(dt1_smoted)[86]<-'Attrition'

#levels(dt1_smoted$Attrition) <- c("No", "Yes")

# Split the data into training and test
set.seed(1000)
intrain_dt1 <- createDataPartition(y = dt1_smoted$Attrition, p= 0.5, list = FALSE)

train_dt1 <- dt1_smoted[intrain_dt1,]
test_dt1 <- dt1_smoted[-intrain_dt1,]

# Load Occupancy Dataset

# Occupancy dataset

#Reading in the data
df1 <- read.csv('datatraining.csv')

#Removing the Date column
df1 <- df1[-1]

#Normalizing the Dataset
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# normalize the room data
df_n <- as.data.frame(lapply(df1[1:5], normalize))
df1$Occupancy <- as.factor(df1$Occupancy )
df_n <- cbind(df_n, df1$Occupancy)
colnames(df_n )[6] <- "Occupancy"

# create training and test data for occupancy dataset
train_dt2 <- df_n[1:4071, ]
test_dt2 <- df_n[4072:8143, ]

#################### Task 1 K-Means and EM #############################

# k means on Occupancy dataset
dt2_kmeans_trn <- kmeans(train_dt2, centers = 4, nstart = 25)
#table(dt2_kmeans_trn$cluster,train_dt2$Occupancy)
dt2_kmeans_trn$cluster <- as.factor(dt2_kmeans_trn$cluster)
ggplot(train_dt2, aes(train_dt2$Light, train_dt2$CO2, color = dt2_kmeans_trn$cluster)) + geom_point()

wss <- (nrow(train_dt2)-1)*sum(apply(train_dt2,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(train_dt2,
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares",
     main="Assessing the Optimal Number of Clusters with the Elbow Method",
     pch=20, cex=2)

# Perform K-Means with the optimal number of clusters identified from the Elbow method
set.seed(7)
km2_dt2 = kmeans(train_dt2, 6, nstart=100)

# Examine the result of the clustering algorithm
print(km2_dt2)

plot(train_dt2, col =(km2_dt2$cluster +1) , main="K-Means result with 6 clusters for Occupancy Detection", pch=20, cex=2)


### k means on IBM Attrition ##

dt1_kmeans_trn <- kmeans(train_dt1, centers = 5, nstart = 25)
dt1_kmeans_trn$cluster <- as.factor(dt1_kmeans_trn$cluster)
ggplot(train_dt1, aes(train_dt1$ï..Age, train_dt1$MonthlyIncome, color = dt1_kmeans_trn$cluster)) + geom_point()

wss_1 <- (nrow(train_dt1)-1)*sum(apply(train_dt1,2,var))
for (i in 2:15) wss_1[i] <- sum(kmeans(train_dt1,
                                     centers=i)$withinss)

plot(1:15, wss_1, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares",
     main="Assessing the Optimal Number of Clusters with the Elbow Method",
     pch=20, cex=2)

# Perform K-Means with the optimal number of clusters identified from the Elbow method
set.seed(7)
km2_dt1 = kmeans(train_dt1,7, nstart =100)
km2_dt1$cluster <- as.factor(km2_dt1$cluster)
ggplot(train_dt1, aes(train_dt1$ï..Age, train_dt1$MonthlyIncome, color = km2_dt1$cluster)) + geom_point()



## Expectation Maximization Occupancy Detection ##
#install.packages("mclust")
#library(mclust)

#Remove target variable
head(df_n)
X = df_n[,-6]
head(X)

#Fitting EM clusteting object
fit_occ <- Mclust(X)
fit_occ


summary(fit_occ)

#names of the fitted model
names(fit_occ)

#column with classification values
fit_occ['classification']

#Plot describing components created based on the BIC criterion
plot(fit_occ, what = "BIC")

plot(fit_occ, what = "classification")



## Expectation Maximization IBM Attrition #####

#Remove target variable and only using continous variables for clustering.
head(numeric)
Y = numeric
head(Y)

#Fitting EM clusteting object
fit_attr <- Mclust(Y)
fit_attr


summary(fit_attr)

#names of the fitted model
names(fit_attr)

#column with classification values
fit_attr['classification']

#Plot describing components created based on the BIC criterion
plot(fit_attr, what = "BIC")

plot(fit_attr, what = "classification")

############## Task 2: Dimentionality reduction #################


##### Feature selection ########
if(!require("FSelector")){
  cat(" \n FSelector package not found.. Hence installing...")
  install.packages("FSelector")
}

library(FSelector)
cat("\n Feature selection for Occupancy Detection")
att.scores.dt2 <- random.forest.importance(Occupancy ~ ., train_dt2)
print(att.scores.dt2)
cat("\n The Most important feature: \n", cutoff.biggest.diff(att.scores.dt2))
cat("\n overall features with importance are: \n", cutoff.k.percent(att.scores.dt2, 0.4))
cat("\n Light is very important feature and then CO2 with feature selection uding random forest.")

cat("\n Feature selection for IBM Attirtion")
att.scores.dt1 <- random.forest.importance(Attrition ~ ., train_dt1)
print(att.scores.dt1)
cat("\n The Most important features are: \n", cutoff.biggest.diff(att.scores.dt1))
cat("\n overall features with importance are: \n", cutoff.k.percent(att.scores.dt1, 0.4))

############## PCA ################

cat("\n PCA for Occupancy")
#removing responce variable 
pca.trn.dt2 <- train_dt2[1:5]
pca.dt2 <- prcomp(pca.trn.dt2, scale. = T)
pca.dt2$rotation
pca.dt2.var <- (pca.dt2$sdev^2)
cat("\n variance of the components: ", pca.dt2.var)
cat("\n check the proportion of variance explained by each component by dividing the variance by sum of total variance.")
prop.pca.dt2.var <- pca.dt2.var/sum(pca.dt2.var)
cat("\nThe proportion is ", prop.pca.dt2.var)

plot(prop.pca.dt2.var, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b",main = "For Occupancy")

biplot(pca.dt2, scale = 0)

#cumulative scree plot
 plot(cumsum(prop.pca.dt2.var), xlab = "Principal Component",
       ylab = "Cumulative Proportion of Variance Explained",
       type = "b", main = "Cumulative Plot For Occupancy")
cat(" \n The component 4 and 5 are closed to 98%, So it doesn't show much variance and can be ignored.")

cat("#add a training set with principal components")

train.pca.dt2 <- data.frame(pca.dt2$x)

#we are interested in first 3 PCAs
train.pca.dt2 <- train.pca.dt2[,1:3]

#test dataset for pca 
test.pca.dt2 <- predict(pca.dt2, newdata = test_dt2[1:5])
test.pca.dt2 <- as.data.frame(test.pca.dt2)

#select the first 3 components
test.pca.dt2 <- test.pca.dt2[,1:3]

# New train and test data with PCA forOccupancy are train.pca.dt2 and test.pca.dt2 resp.

cat("\n PCA for IBM Attrition")

#removing responce variable 
pca.trn.dt1 <- train_dt1[1:85]
pca.dt1 <- prcomp(pca.trn.dt1, scale. = T)
pca.dt1.var <- (pca.dt1$sdev^2)
cat("\n variance of the 1st 10 components: ", pca.dt1.var[1:10])
cat("\n check the proportion of variance explained by each component by dividing the variance by sum of total variance.")
prop.pca.dt1.var <- pca.dt1.var/sum(pca.dt1.var)
cat("\nThe proportion is ", prop.pca.dt1.var[1:20])

plot(prop.pca.dt1.var, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b",main = "For IBM Attrition")

biplot(pca.dt1, scale = 0)

#cumulative scree plot
plot(cumsum(prop.pca.dt1.var), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b", main = "Cumulative Plot For IBM Attrition")

cat(" \n The components affter 60 are closed to 98%, So it doesn't show much variance and can be ignored.")

cat("#add a training set with principal components")

train.pca.dt1 <- data.frame(pca.dt1$x)

#we are interested in first 60 PCAs
train.pca.dt1 <- train.pca.dt1[,1:60]

#test dataset for pca 
test.pca.dt1 <- predict(pca.dt1, newdata = test_dt1[1:85])
test.pca.dt1 <- as.data.frame(test.pca.dt1)

#select the first 60 components
test.pca.dt1 <- test.pca.dt1[,1:60]

# New train and test data with PCA for IBM Attirtion are train.pca.dt1 and test.pca.dt1 resp.


############## ICA ################

#Independant Component Analysis for Attrition#
library("fastICA")
dt1_smoted$Attrition<-NULL
tt_1<-as.matrix(dt1_smoted)

for(i in c(5,10,15,20,25)){
  ica_dt1<-fastICA(tt_1, n.comp = i,alg.typ = "parallel",alpha = 1.0,method = c("R","C"), row.norm = FALSE , maxit = 200 ,tol= 1e-04,verbose = FALSE,
                   w.init = NULL)
  
  par(mfrow = c(1, 2))
  plot(ica_dt1$X, main = "Pre-processed data for IBM Attrition",col = "green")
  plot(ica_dt1$S, main = "ICA components for IBM Attrition",col = "blue")
  
}


cat("#Extracting 15 Icas provides the best separation")
ica_dt1_final<-fastICA(tt_1, n.comp = 15,alg.typ = "parallel",alpha = 1.0,method = c("R","C"), row.norm = FALSE , maxit = 200 ,tol= 1e-04,verbose = FALSE,
                       w.init = NULL)
ica_dt1_knn<-ica_dt1_final$S


#Independant Component Analysis for Occupany#

library("fastICA")
df_n$Occupancy<-NULL
tt<-as.matrix(df_n)

for(i in c(2,3,4)){
  
  ica_dt2<-fastICA(tt, n.comp = i,alg.typ = "parallel",alpha = 1.0,method = c("R","C"), row.norm = FALSE , maxit = 200 ,tol= 1e-04,verbose = FALSE,
                   w.init = NULL)
  par(mfrow = c(1, 2))
  plot(ica_dt2$X, main = "Pre-processed data",col = "green")
  plot(ica_dt2$S, main = "ICA components ",col = "blue")
  print(i)
}

cat("#Extracting 3 Icas provides the best separation")
ica_dt2_final<-fastICA(tt, n.comp = 3,alg.typ = "parallel",alpha = 1.0,method = c("R","C"), row.norm = FALSE , maxit = 200 ,tol= 1e-04,verbose = FALSE,w.init = NULL)
ica_dt2_knn<-ica_dt2_final$S


par(mfrow = c(1, 1))

#########  Randomized Projections  #########




########### Task 3:  Clustering after dimensionality reduction  ##############

cat("\n Using PCA for Occupancy Detection")
set.seed(7)
km_dt2_pca = kmeans(train.pca.dt2, 6, nstart=100)
cat("\nExamine the result of the clustering algorithm based on PCA \n")
print(km_dt2_pca)
plot(train.pca.dt2, col =(km_dt2_pca$cluster +1) , main="K-Means:6 clusters: Occupancy Detection: PCA", pch=20, cex=2)

cat("\n Using Features obtained from feature selecction for Occupancy Detection")

km_dt2_fselect <- kmeans(train_dt2[3:4], 6, nstart=100)
cat("\n Examine the result of the clustering algorithm based on Feature Seletion \n")
print(km_dt2_fselect)

plot(train_dt2[3:4], col =(km_dt2_fselect$cluster +1) , main="K-Means:6 clusters: Occupancy Detection: Feature Selection", pch=20, cex=2)


cat("\n Using PCA for IBM Attrition ")
set.seed(7)

km_dt1_pca = kmeans(train.pca.dt1, 7, nstart=100)
cat("\nExamine the result of the clustering algorithm based on PCA \n")
print(km_dt1_pca)
plot(train_dt1[,c(72,77)], col =(km_dt1_pca$cluster +1) , main="K-Means:7 clusters: IBM Attrition: PCA", pch=20, cex=2)

cat("\n Using Features obtained from feature selecction for IBM Attrition")

km_dt1_fselect <- kmeans(train_dt1[cutoff.k.percent(att.scores.dt1, 0.4)], 6, nstart=100)
cat("\n Examine the result of the clustering algorithm based on Feature Seletion \n")
print(km_dt1_fselect)
plot(train_dt1[cutoff.biggest.diff(att.scores.dt1)], col =(km_dt1_fselect$cluster +1) , main="K-Means:6 clusters: IBM Attrition: Feature Selection", pch=20, cex=2)

#Expectation Maximization on Occupancy detection after PCA
occ <- rbind(train.pca.dt2, test.pca.dt2)

#Fitting EM clusteting object
fit_occ1 <- Mclust(occ)
fit_occ1


summary(fit_occ1)

#names of the fitted model
names(fit_occ1)

#column with classification values
fit_occ1['classification']

#Plot describing components created based on the BIC criterion
plot(fit_occ1, what = "BIC")

plot(fit_occ1, what = "classification")

#Expectation Maximization on IBM Attrition after PCA

attr <- rbind(train.pca.dt1, test.pca.dt1)

#Fitting EM clusteting object
fit_attr1 <- Mclust(attr)
fit_attr1


summary(fit_attr1)

#names of the fitted model
names(fit_attr1)

#column with classification values
fit_attr1['classification']

#Plot describing components created based on the BIC criterion
plot(fit_attr1, what = "BIC")



######################Clustering after Independent Component Analysis######################
cat("\n Using ICA for IBM Attrition")
#ica_dt1_knn
set.seed(7)
km_dt1_ica = kmeans(ica_dt1_knn, 7, nstart=100,iter.max = 200)
cat("\nExamine the result of the clustering algorithm based on ICA \n")
print(km_dt1_ica)
plot(ica_dt1_knn, col =(km_dt1_ica$cluster +1) , main="K-Means:7 clusters: IBM Attrition: ICA", pch=20, cex=2)


cat("\n Using ICA for Occupancy")
#ica_dt2_knn
set.seed(7)
km_dt2_ica = kmeans(ica_dt2_knn, 6, nstart=100,iter.max = 200)
cat("\nExamine the result of the clustering algorithm based on ICA \n")
print(km_dt2_ica)
plot(ica_dt2_knn, col =(km_dt2_ica$cluster +1) , main="K-Means:6 clusters: Occupancy Detection: ICA", pch=20, cex=2)



############################################################################################



########### Task 4:  Neural network learner after dimensionality reduction  ##############


cat("\n Within cluster sum of squares by cluster is lesser using PCA, So using PCA dataset for Neural network for Occupancy Detection \n")

train_ann_dt2 <- data.frame(Occupancy= train_dt2$Occupancy,train.pca.dt2)
levels(train_ann_dt2$Occupancy) <- c("no", "yes")

test_ann_dt2 <-  data.frame(Occupancy= test_dt2$Occupancy,test.pca.dt2)
levels(test_dt2$Occupancy) <- c("no", "yes")
numFolds <- trainControl(method = 'cv', number = 10, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary, preProcOptions = list(thresh = 0.75, ICAcomp = 3, k = 5))

#Tuning parameters:
#size (#Hidden Units)
#decay (Weight Decay)
dt1.grid <- expand.grid(.decay = c(0.5), .size = c(10))
fit2 <- caret::train(x=  train.pca.dt2,y=as.factor(train_ann_dt2$Occupancy), method = 'nnet', preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=dt1.grid)
results2 <- predict(fit2, newdata=train_ann_dt2)
conf2 <- confusionMatrix(results2, train_ann_dt2$Occupancy)
cat("\nFor Train Dataset\n")
print(conf2)
pred_dt2_tr <- ROCR::prediction(predictions = as.numeric(results2),labels = as.numeric(train_ann_dt2$Occupancy))
perf_dt2_tr <- ROCR::performance(pred_dt2_tr, "tpr", "fpr")
auc_dt2_tr <- ROCR::performance(pred_dt2_tr,"auc")
auc_dt2_tr <- round(as.numeric(auc_dt2_tr@y.values),5)
print(paste('AUC of ANN Model on Train Occupany dataset:',auc_dt2_tr))
plot(perf_dt2_tr,type ="o",col="blue", main="AUC: ANN: PCA Occupancy Detection")

cat("\n Within cluster sum of squares by cluster is lesser using PCA, So using PCA dataset for Neural network for IBM Attrition \n")

set.seed(100)
train_ann_dt1 <- data.frame(Attrition= train_dt1$Attrition,train.pca.dt1)
levels(train_ann_dt1$Attrition) <- c("no", "yes")

test_ann_dt1 <-  data.frame(Attrition = test_dt1$Attrition,test.pca.dt1)
levels(test_dt1$Attrition) <- c("no", "yes")

numFolds <- trainControl(method = 'cv', number = 10, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary, preProcOptions = list(thresh = 0.75, ICAcomp = 3, k = 5))

#Tuning parameters:
#size (#Hidden Units)
#decay (Weight Decay)
dt1.grid <- expand.grid(.decay = c(0.5), .size = c(10))

fit1 <- caret::train(x= train_ann_dt1,y=train_ann_dt1$Attrition, method = 'nnet', preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=dt1.grid)
results1 <- predict(fit1, newdata=train_ann_dt1)
conf1 <- confusionMatrix(results1, train_ann_dt1$Attrition)
cat("\nFor Train Dataset\n")
print(conf1)
pred_dt1_tr <- prediction(predictions = as.numeric(results1),labels = as.numeric(train_dt1$Attrition))
perf_dt1_tr <- performance(pred_dt1_tr, "tpr", "fpr")
auc_dt1_tr <- performance(pred_dt1_tr,"auc")
auc_dt1_tr <- round(as.numeric(auc_dt1_tr@y.values),5)
auccuracy_dt1_tr <- conf1$overall["Accuracy"]
print(paste('AUC of Model for Train:',auc_dt1_tr))
cat("\n Model accuracy for Train dataset:", auccuracy_dt1_tr)
plot(perf_dt1_tr,type ="o",col="blue", main="AUC: ANN: PCA IBM Attrition")



########### Task 5:  clustering and Neural Network  ##############


cat("\n Cluster input: Neural network for Occupancy Detection \n")

train_clst_dt2 <- data.frame(Occupancy= train_dt2$Occupancy,dt2_kmeans_trn$cluster)
levels(train_clst_dt2$Occupancy) <- c("no", "yes")

#test_ann_dt2 <-  data.frame(Occupancy= test_dt2$Occupancy,test.pca.dt2)
#levels(test_dt2$Occupancy) <- c("no", "yes")
numFolds <- trainControl(method = 'cv', number = 10, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary, preProcOptions = list(thresh = 0.75, ICAcomp = 3, k = 5))

#Tuning parameters:
#size (#Hidden Units)
#decay (Weight Decay)
dt1.grid <- expand.grid(.decay = c(0.5), .size = c(10))
fit3 <- caret::train(x= train_clst_dt2, y=as.factor(train_clst_dt2$Occupancy), method = 'nnet', preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=dt1.grid)
results3 <- predict(fit3, newdata=train_clst_dt2)
conf3 <- confusionMatrix(results3, train_clst_dt2$Occupancy)
cat("\nFor Train Dataset\n")
print(conf3)
pred_dt2_tr_clst <- ROCR::prediction(predictions = as.numeric(results3),labels = as.numeric(train_clst_dt2$Occupancy))
perf_dt2_tr_clst <- ROCR::performance(pred_dt2_tr_clst, "tpr", "fpr")
auc_dt2_tr_clst <- ROCR::performance(pred_dt2_tr_clst,"auc")
auc_dt2_tr_clst <- round(as.numeric(auc_dt2_tr_clst@y.values),5)
print(paste('AUC of ANN Model on Train Occupany dataset:',auc_dt2_tr_clst))
plot(perf_dt2_tr_clst,type ="o",col="blue", main="AUC: ANN: Cluster: Occupancy Detection")



cat("\n Cluster input: Neural network for IBM \n")

train_clst_dt1 <- data.frame(Attrition= train_dt1$Attrition,dt1_kmeans_trn$cluster)
levels(train_clst_dt1$Attrition) <- c("no", "yes")

#test_ann_dt2 <-  data.frame(Occupancy= test_dt2$Occupancy,test.pca.dt2)
#levels(test_dt2$Occupancy) <- c("no", "yes")
numFolds <- trainControl(method = 'cv', number = 10, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary, preProcOptions = list(thresh = 0.75, ICAcomp = 3, k = 5))

#Tuning parameters:
#size (#Hidden Units)
#decay (Weight Decay)
dt1.grid <- expand.grid(.decay = c(0.5), .size = c(10))
fit4 <- caret::train(x= train_clst_dt1, y=as.factor(train_clst_dt1$Attrition), method = 'nnet', preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=dt1.grid)
results4 <- predict(fit4, newdata=train_clst_dt1)
conf4 <- confusionMatrix(results4, train_clst_dt1$Attrition)
cat("\nFor Train Dataset\n")
print(conf4)
pred_dt1_tr_clst <- ROCR::prediction(predictions = as.numeric(results4),labels = as.numeric(train_clst_dt1$Attrition))
perf_dt1_tr_clst <- ROCR::performance(pred_dt1_tr_clst, "tpr", "fpr")
auc_dt1_tr_clst <- ROCR::performance(pred_dt1_tr_clst,"auc")
auc_dt1_tr_clst <- round(as.numeric(auc_dt1_tr_clst@y.values),5)
print(paste('AUC of ANN Model on Train Occupany dataset:',auc_dt2_tr_clst))
plot(perf_dt2_tr_clst,type ="o",col="Red", main="AUC: ANN: Cluster: IBM Attrition")
