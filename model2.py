import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import math
#sigmoid function
def sigmoid(z):
    return 1/(1+math.exp(-z))
#cost function
def costfunc(Theta1,Theta2,X,Y,lamda,output_layer_size):
    n = len(Y_train)
    a1 = np.hstack(np.ones((n,1)),X_train)
    a2 = sigmoid(a1*Theta1.transpose())
    a2 = np.hstack(np.ones((n,1)),a2)
    a3 = sigmoid(a2*Theta2.transpose())
    Y = np.tile(np.arange(output_layer_size),(n,1)) == np.tile(Y,(1,output_layer_size))
    print(Y)
#model begins
print("This is a machine learning model.........")
print("This model identifies the character from a image........")
print("Loading data...........")
#Loading Data
mndata = MNIST('/home/monika/Documents/mlwoc1-master')
images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
m = len(labels)
images1 = pd.DataFrame(images)
labels1 = pd.DataFrame([labels])
images2 = pd.DataFrame(test_images)
labels2 = pd.DataFrame([test_labels])
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
lamda = 0
costfunc(Theta1,Theta2,X_train,Y_train,lamda,output_layer_size)