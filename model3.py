import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn import preprocessing
#sigmoid function
def sigmoid(z):
    sig =  1./(1 + np.exp(-z))
    sig = np.clip(sig,1e-6,1)
    return sig
#costfunction
def costfunc(X,Y,Theta):
    h_theta = np.zeros((len(X),1))
    k = X.dot(Theta)
    h_theta = sigmoid((k))
    M = len(X)
    cost = (1/M)*(-np.transpose(Y).dot(np.log10(h_theta))-np.transpose(1-Y).dot(np.log10(1-h_theta)))
    return cost
#gradient function
def gradfunc(X,Y,Theta):
    m = len(Y)
    k = X.dot(Theta)
    h_theta = sigmoid((k))
    grad = (np.dot(X.transpose(),np.subtract((h_theta),Y.reshape(8000,1))))/m
    return grad
#Model Begins
print("This is a machine learning model..........")
print("this model works on the algorithm of logistic regression..........")
print("This model predicts the stability of a 4 node star system..........")
print("Loading Data........")
#Loading Data
data = pd.read_csv("data3.csv")
X_train = data.iloc[0:8000,1:13].values
Y_train = data.iloc[0:8000,0].values
X_val = data.iloc[8000:10000,1:13].values
Y_val = data.iloc[8000:10000,0].values
m = len(Y_train)
X_train = np.hstack([np.ones((m,1)),X_train])
X_val = np.hstack([np.ones((len(Y_val),1)),X_val])
#initializing parameters
Theta = np.random.rand(13,1)
#calculating cost
cost = costfunc(X_train,Y_train,Theta)
grad = gradfunc(X_train,Y_train,Theta)
print(cost)
print(grad)
