import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import math
#sigmoid function
def sigmoid(z):
    return 1/(1+math.exp(-z))
#costfunction
def costfunc(X,Y,Theta):
    h_theta = np.zeros((len(X),1))
    k = X@Theta
    h_theta = sigmoid(k)
    cost = (1/m)*(-Y.transpose()@math.log(h_theta)-((np.ones(len(Y),1)-Y).transpose()@math.log(np.ones(len(h_Theta),1)-h_theta)))
    return cost
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
print(cost)
