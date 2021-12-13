##Consolidate Midterm Project Code 

library(ISLR)
library(pls)
library(geometry)
library(leaps)
require(MASS)
require(faraway)
require(glmnet)
require(dplyr)
library(rpart)
library(randomForest)
library(caret)
library(gbm)

# ST 516 Midterm Project

##################################################
#########      Data Pre-processing     ###########
##################################################

# Setting directory path to read the file
getwd()
setwd("~/Desktop/NCSU Classwork/Fall '21/Experimental Statistics for Engineers/Midterm Project")

# Function to get mode for the categorical variables during imputing missing values
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# read data 
housingData <- data.frame(read.csv("housingData.csv"))

# Checking for null values
colnames(housingData)[colSums(is.na(housingData)) > 0]
colSums(is.na(housingData[c("LotFrontage","BsmtFinType1","GarageType")]))

# Imputing missing values for LotFrontage (Continuous Variable)
boxplot(housingData[,"LotFrontage"], main="Box Whisker Plot for LotFrontage")
mean(housingData[,"LotFrontage"], na.rm = TRUE)
median(housingData[,"LotFrontage"], na.rm = TRUE)
# Imputing done using mean as mean and median are same for the given data
housingData[is.na(housingData[,"LotFrontage"]), "LotFrontage"] <- mean(housingData[,"LotFrontage"], na.rm = TRUE)

# Imputing missing values for BsmtFinType1, GarageType categorical variables
housingData[is.na(housingData[,"BsmtFinType1"]), "BsmtFinType1"] <- getmode(housingData[,"BsmtFinType1"])
housingData[is.na(housingData[,"GarageType"]), "GarageType"] <- getmode(housingData[,"GarageType"])

# Checking for null values after pre-processing
colnames(housingData)[colSums(is.na(housingData)) > 0]
colSums(is.na(housingData[c("LotFrontage","BsmtFinType1","GarageType")]))

# Converting years to continuous variable
# YearBuilt, YearRemodAdd, YrSold
YearBuilt.min <- min(housingData[,"YearBuilt"])
housingData["YearBuilt"] <- housingData["YearBuilt"] - YearBuilt.min

YearRemodAdd.min <- min(housingData[,"YearRemodAdd"])
housingData["YearRemodAdd"] <- housingData["YearRemodAdd"] - YearRemodAdd.min

YrSold.max <- max(housingData[,"YrSold"])
housingData["YrSold"] <- YrSold.max - housingData["YrSold"]

# Scale the data
num_cols <- unlist(lapply(housingData, is.numeric)) == TRUE         # Identify numeric columns
num_cols
for(i in colnames(housingData[,num_cols])) {
  if(i != "SalePrice")
    housingData[,i] <- scale(housingData[,i], center=T)
}

# Transforming the response variable using log to get an even spread in the data
housingData[,"SalePrice"] <- log10(housingData[,"SalePrice"])

# Convert categorical values to binary variables
housing.mat <- model.matrix(SalePrice~., housingData)[,-1]
old_col_names <- colnames(housing.mat)
housing.mat <- cbind(housing.mat, housingData$SalePrice)
colnames(housing.mat) <- c(old_col_names,"SalePrice")

# Data prepossessing completed
housing.processed <- data.frame(housing.mat)

# Initializing an array for comparing test MSE's of different models
mse <- data.frame(matrix(NA,nrow=9,ncol=2))  # create dummy data frame to store MSE values
colnames(mse) <- c("Model","MSE")
mse$Model <- c("OLS","Ridge","Lasso","Regression Tree", "Pruned Tree", "Bagging","Random Forest", "Gradient Boosting", "PCR")


##################################################
#########      Subset Selection     #############
##################################################

# forward selection
# start with intercept, add predictor that produces lowest RSS
# take that one predictor model and add the predictor from the
# remaining predictors that produces the lowest RSS
# continue until all 9 predictors have been added

maxval <- 70

fwd.mods <- regsubsets(SalePrice~.,data=housing.processed,nvmax=maxval,method="forward")
fwd.sum <- summary(fwd.mods)
fwd.sum

p=1:maxval

# can use criterion to select best model
aic <- fwd.sum$bic+2*p-log(dim(housing.processed)[1])*p
which.max(fwd.sum$adjr2)
which.min(fwd.sum$bic)
which.min(aic)


# plot criteria to get visual confirmation

par(mfrow=c(2,2))
plot(p,aic,pch=19,type="b",main="AIC")
points(which.min(aic),aic[which.min(aic)],cex=1.5,col="red",lwd=2)
abline(v=c(1:maxval),lty=3)

plot(p,fwd.sum$bic,pch=19,type="b",main="BIC")
points(which.min(fwd.sum$bic),fwd.sum$bic[which.min(fwd.sum$bic)],
       cex=1.5,col="red",lwd=2)
abline(v=c(1:maxval),lty=3)

plot(p,fwd.sum$adjr2,pch=19,type="b",main="adj-R2")
points(which.max(fwd.sum$adjr2),fwd.sum$adjr2[which.max(fwd.sum$adjr2)],
       cex=1.5,col="red",lwd=2)
abline(v=c(1:maxval),lty=3)

# This time the BIC says p=36
# check to see coef of each model selected above

#best AIC: p= 55
coef(fwd.mods,id=47)

#best BIC: p= 31
coef(fwd.mods,id=36)

#best adjr2: p= 62
coef(fwd.mods,id=63)

#choose to go with BIC model (p=31) bcz BIC chooses best "true" model
#  fit best model for p=36

#Housing.best is our new data set because Subset selection chose it- it will be used throughout

xvarm <- names(coef(fwd.mods,id=36))[2:35]
Housing.subset <- housing.processed[,c("SalePrice",xvarm)]

lmod <- lm(SalePrice~.,data=Housing.subset)
par(mfrow=c(2,2))
plot(lmod)
summary(lmod)
vif(lmod)


# Create test and train (again because some people used different names)
RNGkind(sample.kind = "Rounding")
set.seed(123)
test_indices = sample(1:dim(Housing.subset)[1], dim(Housing.subset)[1]*0.2)
housing.test = Housing.subset[test_indices,]
housing.train = Housing.subset[-test_indices,]


##################################################
#########      OLS     ###########
##################################################

fit <- lm(SalePrice~.,data=housing.train)
par(mfrow=c(2,2))
plot(fit)  # residual diagnostics
summary(fit) # fit summary
vif(fit)


# Calculate MSE
predict.lmod <- predict(fit, newdata=housing.test)
mse[1,2] <- mean((10^predict.lmod - 10^housing.test$SalePrice)^2)
print(paste("OLS MSE: ",mse[1,2]))
print(mse)

##################################################
#########      Ridge Regression    ###########
##################################################

#  create predictor matrix and vector for response
x_train <- model.matrix(SalePrice~.,housing.train)[,-1]
y_train <- housing.train$SalePrice
x_test <- model.matrix(SalePrice~.,housing.test)[,-1]
y_test <- housing.test$SalePrice

# create grid for lambda, fit model using all lambdas
grid <- 10^seq(-10,10,length=100) # lambda ranges from 0.1 to 0.00001 
ridge.mod <- glmnet(x_train, y_train ,alpha=0, lambda=grid)  

# plot coefficent values as we change lambda
plot(ridge.mod,xlab="L2 Norm")  # x-axis is in terms of sqrt(sum(beta^2))
abline(h=0,lty=3)

# optimize lambda using cross-validation
cv.ridge <- cv.glmnet(x_train, y_train, alpha=0, lambda=grid)
plot(cv.ridge)
bestlam.r <- cv.ridge$lambda.min # best lambda
mse.r <- min(cv.ridge$cvm)
bestlam.r
mse.r

# get coefficients for best model and compare to OLS
ridge.coef <- predict(ridge.mod,type="coefficients",s=bestlam.r)
ridge.coef

# Calculate MSE
fit.ridge <- predict(ridge.mod, s=bestlam.r, newx = x_test)
mse[2,2] <- mean((10^fit.ridge - 10^y_test)^2)
print(paste("Ridge MSE: ",mse[2,2]))
print(mse)

# plot fitted values for OLS and Ridge, compare with actual with actual
plot(predict.lmod, housing.test$SalePrice,pch=19,col="blue")
points(fit.ridge, housing.test$SalePrice,col="red",lwd=2)
abline(a=0,b=1)

# compare R2 for each fit
R2.ridge <- cor(fit.ridge, housing.test$SalePrice)^2
R2.lm <- cor(predict.lmod, housing.test$SalePrice)^2
R2.ridge
R2.lm


##################################################
#########      Lasso Regression    ###########
##################################################


# create grid for lambda, fit model using all lambdas
lasso.mod <- glmnet(x_train, y_train ,alpha=1, lambda=grid)  

# plot coefficent values as we change lambda
plot(lasso.mod,xlab="L2 Norm")  # x-axis is in terms of sqrt(sum(beta^2))
abline(h=0,lty=3)

# optimize lambda using cross-validation
cv.lasso <- cv.glmnet(x_train, y_train, alpha=1, lambda=grid)
plot(cv.lasso)
bestlam.r <- cv.lasso$lambda.min # best lambda
mse.r <- min(cv.lasso$cvm)
bestlam.r
mse.r

# get coefficients for best model and compare to OLS
lasso.coef <- predict(lasso.mod,type="coefficients",s=bestlam.r)
lasso.coef
sort(data.frame(lasso.coef))

# Calculate MSE
fit.lasso <- predict(lasso.mod, s=bestlam.r, newx = x_test)
mse[3,2] <- mean((10^fit.lasso - 10^y_test)^2)
print(paste("Lasso MSE: ",mse[3,2]))
print(mse)

# plot fitted values for OLS and Ridge, compare with actual with actual
plot(predict.lmod, housing.test$SalePrice,pch=19,col="blue")
points(fit.lasso, housing.test$SalePrice,col="red",lwd=2)
abline(a=0,b=1)

# compare R2 for each fit
R2.lasso <- cor(fit.ridge, housing.test$SalePrice)^2
R2.lasso
R2.lm


##################################################
#########      Regression Tree    ###########
##################################################      

tree.fit <- rpart(SalePrice ~.,method='anova', data=housing.train, minsplit=2)

plot(tree.fit, uniform = TRUE)
text(tree.fit)

summary(tree.fit)

printcp(tree.fit)
plotcp(tree.fit)

tree.predict <- predict(tree.fit, newdata = housing.test)
mse[4,2] <- mean((10^tree.predict - 10^housing.test$SalePrice)^2)
print(paste("Regression Tree MSE: ",mse[4,2]))
print(mse)

table(housing.test$SalePrice, tree.predict)

# Tree size 8 is the smallest size tree with its error within the 1 standard deviation of the lowest error
prune.tree <- prune(tree.fit, cp <- tree.fit$cptable[8])

plot(prune.tree, uniform = TRUE)
text(prune.tree)

prune.tree.predict <- predict(prune.tree, newdata = housing.test)
mse[5,2] <- mean((10^prune.tree.predict - 10^housing.test$SalePrice)^2)
print(paste("Pruned Tree MSE: ",mse[5,2]))
print(mse)

table(housing.test$SalePrice, prune.tree.predict)

#the plot shows prediction by tree with those red lines. It has higers error since it predicts one of those 7 values irrespective of the input
plot(predict.lmod, housing.test$SalePrice,pch=19,col="blue")
points(prune.tree.predict, housing.test$SalePrice,col="red",lwd=2)
abline(a=0,b=1)


##################################################
#########      Bagging   ###########
################################################## 

p <- dim(housing.train)[2] - 1
bag.mod <- randomForest(SalePrice~., data=housing.train, mtry=p, ntree=1000,importance=T)
bag.mod

#take ntree value more than 1000 since that's where the graph gets almost flat
plot(bag.mod)

#important variables
varImpPlot(bag.mod,type=1, pch=19)

bag.predict <- predict(bag.mod, newdata=housing.test, n.trees=1000)
mse[6,2] <- mean((10^bag.predict - 10^housing.test$SalePrice)^2)
print(paste("Bagging MSE: ",mse[6,2]))
print(mse)

# plot fitted values for OLS and Bagging, compare with actual with actual
plot(predict.lmod, housing.test$SalePrice,pch=19,col="blue")
points(bag.predict, housing.test$SalePrice,col="red",lwd=2)
abline(a=0,b=1)


##################################################
#########      Random Forest   ###########
##################################################     

control <- trainControl(method="cv", number=5, search="grid")

#####################
#       WARNING     #
#####################
#This part would take around 30+ mins to run. Since it takes into account all sets upto 90 variables

tunegrid <- expand.grid(mtry=c(1:35))
rf_gridsearch <- train(SalePrice~., data=housing.train, method="rf", metric="RMSE", 
                       tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)
print(rf_gridsearch)


#from graph it's best to use a subset of 25 variables out of given 35.
rf.mod <- randomForest(SalePrice~.,data=housing.train, mtry=25, ntree=1000, importance=T)
rf.mod

rf.mod.pred <- predict(rf.mod, newdata=housing.test)
plot(rf.mod)
mse[7,2] <- mean((10^rf.mod.pred - 10^housing.test$SalePrice)^2)
#mse[7,2] <- 422718822
print(paste("Random Forest MSE: ", mse[7,2]))
print(mse)

varImpPlot(rf.mod,type=1,pch=19,main = "Important Variables")

plot(predict.lmod, housing.test$SalePrice,pch=19,col="blue")
points(rf.mod.pred, housing.test$SalePrice,col="red",lwd=2)
abline(a=0,b=1)

##################################################
#########      Gradient Boosting   ###########
##################################################     


# tune model hyperparamters using caret
control <- trainControl(method="cv", number=5, search="grid")
tunegrid <- expand.grid(n.trees=c(100,500,1000),
                        interaction.depth=c(1,3,5),
                        shrinkage=c(0.001,0.05,0.1),
                        n.minobsinnode=c(1,3,5))
gb_gridsearch <- train(SalePrice~.,data=housing.train, 
                       method="gbm", metric="RMSE",
                       tuneGrid=tunegrid, trControl=control)

plot(gb_gridsearch, ylab="Train RMSE")

#choose shrinkage = 0.1, depth=5, trees = 1000, minobsinnode = 3 which gives minnimum MSE
gb.mod <- gbm(SalePrice ~ ., data=housing.train,
              distribution = "gaussian",n.trees = 1000,
              shrinkage = 0.1, interaction.depth = 5, 
              n.minobsinnode=3)

gb.predict <- predict(gb.mod, newdata=housing.test ,n.trees=1000 ,shrinkage = 0.1,
                      interaction.depth = 5, n.minobsinnode=3)

# Converting SalePrice back to its original scale using exponential function

mse[8,2] <- mean((10^gb.predict - 10^housing.test$SalePrice)^2)
print(paste("Gradient Boosting MSE: ", mse[8,2]))
print(mse)
summary(gb.mod,cBars=10)
summary(gb.mod)

plot(predict.lmod,housing.test$SalePrice,pch=19,col="blue")
points(gb.predict,housing.test$SalePrice,col="red",lwd=2)
abline(a=0,b=1)

# We can observe that a lot of the points of the Gradient Boosting model do not overlap with the OLS model
# Although, we get a lower MSE with the gradient boosting model



##################################################
#########             PCR             ###########
##################################################    

# pca analysis on subset data

pcr.housing <- prcomp(housing.train, center = FALSE)
summary(pcr.housing)

pcr.housing$rotation    # Eigen vectors of the new principal components
pcr.housing$x           # Calculates the new values for PCA from the original x
biplot(pcr.housing,choices=c(1,2), main="PCR - PC1 vs PC2")

# building a mode using principal component regression
pcr.mod <- pcr(SalePrice ~ ., data=housing.train, validation="CV")
summary(pcr.mod)
validationplot(pcr.mod,val.type="MSEP", main="Scree plot for PCR Components")    # scree plot to see best number of PCs
plot(pcr.mod,  main="PCR Model - Prediction vs Measured")

# Principal Components that explain most of the variance = 5 and all components after explain variance lesser than the original variables

# Prediction using PCR
predict.pcr <- predict(pcr.mod, housing.test, ncomp=5)

# Converting Sale Price back using the exponential function
mse[9,2] <- mean((10^predict.pcr - 10^housing.test$SalePrice)^2)
print(paste("PCR MSE: ", mse[9,2]))
print(mse)

plot(predict.lmod,housing.test$SalePrice,pch=19,col="blue", main="OLS vs PCR")
points(predict.pcr,housing.test$SalePrice,col="red",lwd=2)
abline(a=0,b=1)

# We can observe that most of the point of the OLS model and our trained PCR model are overlapping
# which indicates that most of the variance is explained using the PCR model
# Although, we can observe that the Test MSE is a little higher for our PCR model, but we obtain a drastically
# more interpret able mode with only 7 predictors compared to the 30 in the OLS model.

# PLotting MSE's for all models
View(mse)
barplot(MSE~Model,data=mse,col="blue",ylab="test MSE estimate")




