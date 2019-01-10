import numpy as np
from loader import MNIST
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
#sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))
#sigmoid gradient function
def sigmoidgradient(z):
    sig = sigmoid(z)*(1-sigmoid(z))
    return sig
#forwardpropagation
def forwardprop(Theta1,Theta2,X_train,Y_train):
    a1 = np.hstack([np.ones((len(X_train),1)),X_train])
    a2 = sigmoid((np.dot(a1,np.transpose(Theta1))))
    a2 = np.hstack([np.ones((len(X_train),1)),a2])
    a3 = sigmoid((np.dot(a2,np.transpose(Theta2))))
    return a3
#cost function
def costfunc(Theta1,Theta2,X_train,Y_train,lamda,output_layer_size):
    a3 = forwardprop(Theta1,Theta2,X_train,Y_train)
    cost = (-1/len(Y_train))*np.sum(Y_train*np.log10(a3) + (1-Y_train)*np.log10(1-a3))
    error = (lamda/(2*m))*(np.sum(Theta1[:,1:len(Theta1)]**2) + np.sum(Theta2[:,1:len(Theta2)]**2))
    cost = cost + error
    return cost
#backpropagation
def grad(X_train,Y_train,Theta1,Theta2,lamda,alpha):
    a1 = np.hstack([np.ones((len(X_train),1)),X_train])
    a2 = sigmoid((np.dot(a1,np.transpose(Theta1))))
    a2 = np.hstack([np.ones((len(X_train),1)),a2])
    a3 = forwardprop(Theta1,Theta2,X_train,Y_train)
    del1 = np.zeros(np.shape(Theta1))
    del2 = np.zeros(np.shape(Theta2))
    print(np.shape(Theta1))
    print(np.shape(a1))
    for i in range(0,len(Y_train)-1):
        d3 = a3[i:i+1,:] - Y_train[i:i+1,:]
        d2 = np.dot(np.transpose(Theta2[:,1:len(Theta2[i,:])]),np.transpose(d3))*sigmoidgradient(np.dot(Theta1,np.transpose(a1[i,:])))
        print(np.shape(d2))
        del1 = del1 + np.dot(d2[1:len(d2),:],a1[i,:])
        del2 = del2 + np.dot(np.transpose(d3),a2[i,:])
    Theta1_grad = (1/m)*del1
    Theta2_grad = (1/m)*del2
    print(Theta1_grad)
    print(Theta2_grad)
#model begins
print("This is a machine learning model.........")
print("This model identifies the character from a image........")
print("Loading data...........")
#Loading Data
mndata = MNIST("C:/Users/Prakhar Pande/Documents/mlwoc1")
images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
m = len(labels)
images1 = pd.DataFrame(images)
labels1 = pd.DataFrame([labels])
labels1= np.transpose(labels1)
images2 = pd.DataFrame(test_images)
labels2 = pd.DataFrame([test_labels])
labels2= np.transpose(labels2)
X_train = images1.iloc[0:50000,:].values
Y_train = labels1.iloc[0:50000,:].values
X_val = images2.iloc[50000:m,:].values
Y_val = labels2.iloc[50000:m,:].values
#Defining architecture of the network
input_layer_size = 784
hidden_layer_size = 50
output_layer_size = 10
Theta1 = np.random.rand(50,785)
Theta2 = np.random.rand(10,51)
#calculating cost and forward propagation
lamda = 1
Y_train = np.tile(np.arange(output_layer_size),(len(Y_train),1)) == np.tile(Y_train,(1,output_layer_size))
Y_train = Y_train*1
cost = costfunc(Theta1,Theta2,X_train,Y_train,lamda,output_layer_size)
print(cost)
#calculating gradient using backpropagation
alpha = 0.1
grad(X_train,Y_train,Theta1,Theta2,lamda,alpha)
