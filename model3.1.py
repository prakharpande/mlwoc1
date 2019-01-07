# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 20:56:25 2019

@author: Prakhar Pande
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
#Model Begins
print("This is a machine learning model..........")
print("this model works on the algorithm of logistic regression..........")
print("This model predicts the stability of a 4 node star system..........")
print("Loading Data........")
#Loading Data and splitting it into test set and training set
data = pd.read_csv("data3.csv")
data = data.dropna(axis = 'rows')
X_train = data.iloc[0:8000,1:13].values
Y_train = data.iloc[0:8000,0].values
X_val = data.iloc[8000:9900,1:13].values
Y_val = data.iloc[8000:9900,0].values
m = len(Y_train)
print(data.isnull().sum().sum())
#np.clip(X_train,-10000,10000)
#Adding a cloumn of 1's to the imput sets
#also normalizing the data
X_train = np.hstack([np.ones((m,1)),X_train])
X_train = preprocessing.normalize(X_train)
X_val = np.hstack([np.ones((len(Y_val),1)),X_val])
X_val = preprocessing.normalize(X_val)
#making an instance of the model
logisticRegr = LogisticRegression()
#training the logistic regression model using scikit
logisticRegr.fit(X_train,Y_train)
#predicting data
logisticRegr.predict(X_val[0:10])
