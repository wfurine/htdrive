from nn import simple_neural as nn
import numpy as np
from numpy import genfromtxt
import csv
import datagen
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt



def generate_tests(num_tests=36, num_drivers=10000):
	data, classes = datagen.function(num_drivers)
	X_train, X_test, y_train, y_test = train_test_split(data, classes, 
														test_size=0.20, random_state=420)
	return X_train, X_test, y_train, y_test



X_train, X_test, y_train, y_test = generate_tests()
nn.multilayer_perceptron(X_train, X_test, y_train, y_test)
