import numpy as np
from numpy import genfromtxt
import sklearn
#from sklearn import train_test_split
#from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv


# drive_data_path = '/Users/Stephen/htdrive/output_tests.csv' 

# class_data_path = '/Users/Stephen/htdrive/output_tests.csv'

# data_drive = genfromtxt(drive_data_path, delimiter=',')
# data_classes = genfromtxt(class_data_path, delimiter=',')

# print data_drive


X_train, X_test, y_train, y_test = train_test_split(data_drive, data_classes, 
                                          test_size=0.20, random_state = 420)



num_train = X_train.shape[0]
num_test = X_test.shape[0]

pca = PCA(n_components=8000)

pca.fit(X_train)

print(pca.explained_variance_ratio_) 