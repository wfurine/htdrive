from nn import simple_neural as nn
import numpy as np
from numpy import genfromtxt
import csv
import datagen
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt





def generate_tests(num_tests=36, num_drivers=10000, beta_weight=1):
	data, classes = datagen.function(num_drivers, beta_weight=beta_weight)
	X_train, X_test, y_train, y_test = train_test_split(data, classes, 
														test_size=0.20, random_state=420)
	return X_train, X_test, y_train, y_test


def index_of_threshold(array, threshold=1):
	count = len(array)
	for i in xrange(count):
		if (array[i] >= threshold):
			return i+1
		
	return -1

nn_parameters = {
    'learning_rate': 0.001,
    'training_epochs': 10,
    'batch_size': 1,
    'display_step': 1,
    'test_step': 1,
}

nn_network_def = {
    'dim_input': 36,
    'dim_layer1': 150,
    'dim_layer2': 250,
    'dim_output': 4,
}




beta_weights_list = [0.01, 0.05, 0.15, 0.25, 0.5, 0.75, 1.0]
size95array = []
nn_accuracy_data = []
for weight in beta_weights_list:
	# Size of the feature vector
	size_for_95 = -1

	for size in xrange(1, 36):
		nn_network_def['dim_input'] = size
		X_train, X_test, y_train, y_test = generate_tests(beta_weight = weight)
		X_train = X_train[:,0:size]
		X_test = X_test[:,0:size]
		epochs, accuracy, cost = nn.multilayer_perceptron(X_train, X_test, 
								y_train, y_test, 
								nn_parameters=nn_parameters, 
								nn_network_def=nn_network_def)

		if index_of_threshold(accuracy, threshold=0.95) > -1:
			size_for_95 = index_of_threshold(accuracy, threshold=0.95)
			break

	size95array.append(size_for_95)

	# We want to plot accuracy in the y axis and epochs in the x

plt.plot(beta_weights_list, size95array)
plt.show()

