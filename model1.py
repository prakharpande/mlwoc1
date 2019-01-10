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
        h = np.clip(h,-10000000,10000000)
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
X = data.iloc[0:400,0:2].values
Y = data.iloc[0:400,2].values
X_val = data.iloc[400:500,0:2].values
Y_val = data.iloc[400:500,2].values
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
X_val = np.hstack([np.ones((100,1)),X_val])
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
J_val = costfunc(X_val,Y_val,Theta)
print("The validation cost is:")
print(J_val)
#plotting the output
#plt.plot(X[:,1],Y,'ro')
# t = [0,200,1]
#plt.plot(t,Theta[0]+Theta[1]+t*Theta[2],'r--',t,Theta[0]+t*Theta[2],'g--')
#plt.xlabel('height in cm')
#plt.ylabel('weight in kg')
#plt.axis([0,200,0,200])
#plt.grid(True)
#plt.show()
#Real Test
test = np.zeros((1,3))
test = np.array([1,1,173])
predictout = test.dot(Theta)
print("The value of weight(in kgs) for the test input is:")
print(predictout)
