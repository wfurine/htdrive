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



drive_data_path = 'output_tests.csv'
class_data_path = 'output_classes.csv'
model_save_path = 'multilayer_perceptron.ckpt'

## ******************************
## DATA PREPROCESSING AHEAD HERE
## ******************************


data_drive = genfromtxt(drive_data_path, delimiter=',')
data_classes = genfromtxt(class_data_path, delimiter=',')

# print data_drive
# print data_classes


X_train, X_test, y_train, y_test = train_test_split(data_drive, data_classes, 
                                          test_size=0.20, random_state = 420)

num_train = X_train.shape[0]
num_test = X_test.shape[0]


# Save the testing data into files for later use...
with open("test_driver.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(X_test)


with open("test_classes.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(y_test)


# X_train, X_test, y_train, y_test = generate_tests()
nn.multilayer_perceptron(X_train, X_test, y_train, y_test, 
						nn_parameters=nn_parameters, 
						nn_network_def=nn_network_def)
