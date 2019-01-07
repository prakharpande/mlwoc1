import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
#sigmoid function
def sigmoid(z):
    sig =  1/(1 + np.exp(-z))
    sig = np.clip(sig,1e-6,1)
    return sig
#costfunction
def costfunc(X,Y,Theta,lamda):
    k = np.dot(X,Theta)
    h_theta = sigmoid((k))
    cost = (1/len(Y))*(np.dot(-np.transpose(Y) , np.log10(h_theta)) + np.dot(np.transpose(1-Y) , np.log10(1-h_theta)))
    cost = cost + (lamda/2*len(Y))*(np.dot(np.transpose(Theta),Theta))
    return cost
#gradient function
def gradfunc(X,Y,Theta,lamda):
    k = np.dot(X,Theta)
    h_theta = sigmoid((k))
    Y = np.reshape(Y,(len(Y),1))
    grad = np.dot(np.transpose(h_theta - Y),X)/len(Y)
    grad = np.transpose(grad)
    grad = grad + (lamda/len(Y))*Theta
    return grad
#minimizing cost
def gradientdescent(X,Y,Theta,alpha,maxiter,lamda):
    for i in range(0,maxiter):
        Theta = Theta - alpha*gradfunc(X,Y,Theta,lamda)
    return Theta
def predict(Y_predict,Y_val):
    Y_val = Y_val.reshape(len(Y_predict),1)
    Y_new = np.zeros((len(Y_val)))
    Y_new =  Y_predict >= 0.5
    Y_new  = Y_new*1
    A = (Y_val==Y_new)
    acc = np.mean(A)*100
    return acc
#Model Begins
print("This is a machine learning model..........")
print("this model works on the algorithm of logistic regression..........")
print("This model predicts the stability of a 4 node star system..........")
print("Loading Data........")
#Loading Data
data = pd.read_csv("data3.csv")
data = data.dropna(axis = 'rows')
X_train = data.iloc[0:8000,1:13].values
Y_train = data.iloc[0:8000,0].values
X_val = data.iloc[8000:9900,1:13].values
Y_val = data.iloc[8000:9900,0].values
m = len(Y_train)
X_train = np.hstack([np.ones((m,1)),X_train])
X_train = preprocessing.normalize(X_train)
X_val = np.hstack([np.ones((len(Y_val),1)),X_val])
X_val = preprocessing.normalize(X_val)
#initializing parameters
Theta = np.random.rand(13,1)
#calculating cost
lamda = 0.01
cost = costfunc(X_train,Y_train,Theta,lamda)
grad = gradfunc(X_train,Y_train,Theta,lamda)
print(cost)
#print(grad)
#Running gradient descent
alpha = 10
maxiter = 10000
Theta = gradientdescent(X_train,Y_train,Theta,alpha,maxiter,lamda)
print("The optimal value of Theta is:")
print(Theta)
#finding accuracy
Y_predict = sigmoid(np.dot(X_val,Theta))
accuracy = predict(Y_predict,Y_val)
print(accuracy)
