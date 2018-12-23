import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#cost function
def costfunc(X,Y,Theta):
    cost = 0
    cost = np.sum(((X[:,0:3].dot(Theta))-Y)**2)/(2*m)
    return cost
#Gradient Decent
def gradientdescent(X,Y,alpha,Theta,iterations):
    m = len(Y)
    for iter in range(1,iterations):
        h = (np.dot(X[:,0:3],Theta) - Y)
        h = np.clip(h,-100000,100000)
        t1 = np.sum(h)
        t2 = np.sum(h*X[:,1])
        t3 = np.sum(h*X[:,2])
        Theta[0] = Theta[0] - (alpha/m)*t1
        Theta[1] = Theta[1] - (alpha/m)*t2
        Theta[2] = Theta[2] - (alpha/m)*t3
    return Theta
#model
print("This program is machine learning model.....")
print("This model predicts your weight based upon your height and gender......")
print("Loading Data.....")
data = pd.read_csv("data1")
# print(data)
X = data.iloc[0:500,0:2].values
Y = data.iloc[0:500,2].values
# print(X)
# print(Y)
m = len(Y)
#plt.plot(X[:,1],Y,'ro')
#plt.xlabel('height in cm')
#plt.ylabel('weight in kg')
#plt.axis([0,200,0,200])
#plt.grid(True)
#plt.show()
print("Now Calculating Cost and Gradient")
X = np.hstack([np.ones((m,1)),X])
Theta = np.zeros((3,1))
# parameters for gradient descent
iterations = 1000
alpha = 0.000000001
#initial cost
J = costfunc(X,Y,Theta)
print(J)
Theta =  gradientdescent(X,Y,alpha,Theta,iterations)
print("The optimal value of Theta is:")
print(Theta)
J = costfunc(X,Y,Theta)
print("The minimum cost is:")
print(J)
#plotting the output
#plt.plot(X[:,1],Y,'ro')
#plt.plot(X)
#plt.xlabel('height in cm')
#plt.ylabel('weight in kg')
#plt.axis([0,200,0,200])
#plt.grid(True)
#plt.show()
#Real Test
test = np.zeros((1,3))
test = np.array([1,0,152.5])
predictout = test.dot(Theta)
print("The value of weight(in kgs) for the test input is:")
print(predictout)
