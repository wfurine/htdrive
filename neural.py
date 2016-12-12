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


beta_weights_list = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
nn_accuracy_data = []
for weight in beta_weights_list:
	X_train, X_test, y_train, y_test = generate_tests(beta_weight = weight)
	epochs, accuracy, cost = nn.multilayer_perceptron(X_train, X_test, 
							y_train, y_test, 
							nn_parameters=nn_parameters, 
							nn_network_def=nn_network_def)

	# We want to plot accuracy in the y axis and epochs in the x
	plt.plot(epochs, accuracy, label="beta_weight = {}".format(weight))

plt.show()

