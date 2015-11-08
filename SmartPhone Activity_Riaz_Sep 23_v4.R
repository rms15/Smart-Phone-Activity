##########################
#smart phone activity competition

# load required packages #
require("MASS")  # for LDA, QDA
library("MASS") 
require("class") # for k-nearest neighbours
library("class")
require("e1071") # for SVM
library("e1071")
require("kernlab") # for k-SVM
library("kernlab")
require("glmnet")
library("glmnet")
library("ggplot2")
?glmnet

setwd("/Users/riazm_shaik/Rice/STAT 640 - Data Mining/Kaggle/data")
rm=(list=ls())

# load training and test data for Smart Phone activity
xtrain = as.matrix(read.csv(file="training_data.csv",header=FALSE))
ytrain = as.matrix(read.csv(file="training_labels.csv",header=FALSE))
strain = as.matrix(read.csv(file="training_subjects.csv",header=FALSE))
xtest = as.matrix(read.csv(file="test_data.csv",header=FALSE))
stest = as.matrix(read.csv(file="test_subjects.csv",header=FALSE))
labels = read.table(file="feature_labels.txt",header=FALSE)

cols = paste('V',labels$V1,sep="")
cols
colnames(xtrain) = cols
xtrain[1:10,1:10]
xtrain.cov<-cov(xtrain)
xtrain.cov[1:10,1:10]
dim(xtrain.cov)

(ytrain==1)[1:120]

plot(xtrain[1:100,1])

ggplot(factor(xtrain[,1]))
p <- ggplot(factor(xtrain[,1]), aes(factor(xtrain[,1]), factor(ytrain)),binwidth=5)
?ggplot

# For creating a cross-validation dataset out of training data set
part1<-1
endtrain<-round(6831*part1)
endtrain

# To check if the variables are normal distributed. Use only if required
test.normal <- function (xtrain)
{
      norm.test.out<-matrix(data=NA,nrow=561,ncol=3)
      norm.test<-NULL
      for (i in (1:ncol(xtrain)))
      {  
      norm.test<-shapiro.test(xtrain[1:5000,i])  
      norm.test.out[i,1]<-i
      norm.test.out[i,2]<-norm.test$p.value
          if (norm.test$p.value <= 0.00001 )
          {
             norm.test.out[i,3] <- 'Not Normal'
          }
          else
          {
            norm.test.out[i,3] <- 'Normal'
          }
      }
      
      notNormal.vars<-norm.test.out[norm.test.out[1:561,3]!='Normal',]
      normal.vars<-norm.test.out[norm.test.out[1:561,3]=='Normal',]
      nrow(norm.test.out[norm.test.out[1:561,3]!='Normal',])
      nrow(norm.test.out[norm.test.out[1:561,3]=='Normal',])
      normal.vars
      write.csv(notNormal.vars,file="notNormal.csv",row.names=FALSE)
      write.csv(normal.vars,file="normal.csv",row.names=FALSE)
}
# normality test ends #


# To calculate the prior probabilities P(X=X|G=K)
prob.prior <- function (ytrain)
{
    count.act<-NULL
    count.act[1]<-length(ytrain[ytrain[,1]==1,])
    count.act[2]<-length(ytrain[ytrain[,1]==2,])
    count.act[3]<-length(ytrain[ytrain[,1]==3,])
    count.act[4]<-length(ytrain[ytrain[,1]==4,])
    count.act[5]<-length(ytrain[ytrain[,1]==5,])
    count.act[6]<-length(ytrain[ytrain[,1]==6,])
    sum(count.act)
    p.act<-count.act/sum(count.act)
    p.act
    return(p.act)
}
p.act<-prob.prior(ytrain)
p.act
# k-NN(5)
predK = knn(test=xtest,train=xtrain,cl=as.factor(ytrain),k=5)
predK = knn(test=xtrain,train=xtrain,cl=as.factor(ytrain),k=5)

cbind(predK[1:10],ytrain[1:10,1])
str(ytrain)
length(predK)
ytrain[1:10,1]
diff<-as.numeric(predK[1:6831])-as.numeric(ytrain[1:6831,1])
sum(diff==0)
predK[1:10] - as.factor(ytrain[1:10])
predK[predK[,1]=ytrain[,1],]
dim(ytrain)
cols = c("Id","Prediction")
submitK = cbind(1:length(predK),predK)
colnames(submitK) = cols
write.csv(submitK,file="benchmark_KNN5_latest.csv",row.names=FALSE)


#LDA
?lda
trainK<-lda(xtrain,grouping=ytrain,prior=p.act)
predK<-predict(trainK,xtest)$class
trainK
predK.train<-predict(trainK,xtrain)$class
predK.train
diff<-as.numeric(predK.train[1:6831])-as.numeric(ytrain[1:6831,1])
cols = c("Id","Prediction")
submitLDA = cbind(1:length(predK),predK)
colnames(submitLDA) = cols
write.csv(submitLDA,file="LDA.csv",row.names=FALSE)



?qda
# QDA
library("MASS")
trainK<-qda(xtrain,grouping=as.factor(ytrain),CV=TRUE)
trainK
predK<-predict(trainK,xtest)$class
predK
cols = c("Id","Prediction")
submitQDA = cbind(1:length(predK),predK)
sum(diff!=0)
length(diff)
colnames(submitQDA) = cols
write.csv(submitQDA,file="QDA.csv",row.names=FALSE)
?as.factor
83/6831

# SVM
nu=6

trainK<-svm(xtrain,y=as.factor(ytrain),type="nu-classification",nu)
print(trainK)
summary(trainK)
predK<-predict(trainK,xtest)
predK
cols = c("Id","Prediction")
submitSVM = cbind(1:length(predK),predK)
colnames(submitSVM) = cols
submitSVM
write.csv(submitSVM,file="SVM.csv",row.names=FALSE)

# ksvm
?ksvm
nu=6
trainK<-ksvm(xtrain,y=as.factor(ytrain),type="nu-svc",kernel="rbfdot",nu,cross=5)
print(trainK)
summary(trainK)
predK<-predict(trainK,xtest)
predK
cols = c("Id","Prediction")
submitKSVM = cbind(1:length(predK),predK)
colnames(submitKSVM) = cols
submitKSVM
write.csv(submitKSVM,file="kSVM.csv",row.names=FALSE)


# to find correlation between variables
cor.train<-cor(xtrain)
dim(cor.train)
length(cor.train[,1]==1,cor.train[1,]=1)
cor.train[cor.train[,1:1]==1,cor.train[1,]==1]

#SVM
trainK<-svm(xtrain,y=as.factor(ytrain),type="C-classification")
trainK
trainK<-svm(xtrain,y=as.factor(ytrain),type="C-classification",kernel='linear',cost=0.1)
trainK
#trainK.tune <- tune.svm(x=xtrain,y=as.factor(ytrain),type='C-classification',cost=10^(-4:4),kernel='linear')
print(trainK)
summary(trainK)
predK<-predict(trainK,xtest)
#predK<-predict(trainK,xtrain)
diff<-as.numeric(predK[1:6831])-as.numeric(ytrain[1:6831,1])
sum(diff!=0)
46/6831

predK
cols = c("Id","Prediction")
submitSVM = cbind(1:length(predK),predK)
colnames(submitSVM) = cols
submitSVM
write.csv(submitSVM,file="SVM_c0.1_latest.csv",row.names=FALSE)



?svm
# ksvm - C-svc
?ksvm
?svm
#trainK<-ksvm(xtrain,y=as.factor(ytrain),type="C-svc",kernel="rbfdot",cross=10,C=5)
trainK<-ksvm(xtrain,y=as.factor(ytrain),type="C-svc",kernel="rbfdot")
print(trainK)
summary(trainK)
train.ksvm.c15<-trainK
predK<-predict(trainK,xtest)
#predK<-predict(trainK,xtrain)
diff<-as.numeric(predK[1:6831])-as.numeric(ytrain[1:6831,1])
sum(diff!=0)
81/6831

predK
cols = c("Id","Prediction")
submitKSVM.basic = cbind(1:length(predK),predK)
colnames(submitKSVM.basic) = cols
submitKSVM.c15
write.csv(submitKSVM.basic,file="kSVM_basic_latest.csv",row.names=FALSE)

#Carr <- c(1,4,5,5.5,6,8,10,15)
#tr.err<- c(0.011858, 0.003953, 0.002635, 0.002342, 0.001903, 0.001171, 0.000293, 0)
#tst.err<-c(0.023715, 0.016689, 0.015371, 0.015517, 0.016396, 0.014639, 0.014493, 0.015078)

C-val<-c(12,15)
tr.err <- c(0.000146,0)
tst.err<- c(0.009663,0.009222)

plot(tr.err ~ Carr)
plot(tst.err ~ Carr)
?qda
