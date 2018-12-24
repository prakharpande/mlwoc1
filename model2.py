import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
print("This is a machine learning model.........")
print("This model identifies the character from a image........")
print("Loading data...........")
#Loading Data
mndata = MNIST('/home/monika/Documents/mlwoc1-master')
images, labels = mndata.load_training()
print(labels[10000:10030])
