import numpy as np
from numpy import genfromtxt
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv
import datagen

beta_vals = [0.01, 0.05, 0.15, 0.25, 0.50, 0.75, 1.00]

betas = x * numpy.array([[[0, 0, 0, 0],
                              [1, 0, 0, 0],
                              [1, 0, 0, 0],
                              [0, 0, 0, 0]],
                             [[.6, 0, .6, 1],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [.6, 0, .6, 1]],
                             [[.3, 2, 0, .8],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [.3, 2, 0, .8]],
                             [[0.2, 1, 0, 0.8],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0.2, 1, 0, 0.8]],
                             [[0, 0, 0, 0],
                              [0.6, 0, 1, 0],
                              [0.6, 0, 1, 0],
                              [0, 0, 0, 0]],
                             [[0.2, 0.2, 1.2, 0.8],
                              [0.6, 0.6, 0.2, 0],
                              [0.6, 0.6, 0.2, 0],
                              [0.2, 0.2, 1.2, 0.8]],
                             [[0.2, 0, 1.2, 0.8],
                              [0.4, 0, 1, 1],
                              [0, 0, 0, 0],
                              [0.2, 0, 1, 0.8]],
                             [[0, 0, 0, 0],
                              [0.8, 0, 1, 0.2],
                              [0.8, 0, 1, 0.2],
                              [0, 0, 0, 0]],
                             [[0.6, 0, 0, 1],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]]])


data_actual, classes_actual = datagen.function(10000, betas)


pca = PCA(n_components=36)

pca.fit(data_actual)

print x
print(pca.explained_variance_ratio_) 
array = []
array2 = [pca.explained_variance_ratio_[0]]
for i in range(1, 37):
    array.append(i)
print array
for i in range(1, 36):
    array2.append(array2[i-1] + pca.explained_variance_ratio_[i])
print array2
plt.plot(array, array2)
plt.xlabel('N features')
plt.ylabel('% variance explained')
plt.show()