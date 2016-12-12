import random
import numpy
from operator import add
import csv
from sklearn.decomposition import PCA
from numpy import genfromtxt
import matplotlib.pyplot as plt



betas_list = numpy.array([[[0, 0, 0, 0],
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
    
sd = 0.1
def checker(i,n):
    if i[n] == '1':
        return 1
    else:
        return 0

def genclass(n):
    classes = ['1000', '1100', '1110', '1101', '1010', '1001', '1011', '1111', '0100', '0110', '0101', '0111', '0010', '0011', '0001', '0000']
    people = []
    for i in range(n):
        people.append(random.choice(classes))
    return people



def function(n, betas=betas_list):
    final  = []
    mylist = genclass(n)
    a = 0
    
    # create numpy array to store actual classes
    # array with shape ([class], 4)
    classes_actual = numpy.zeros(shape=(n, 4))
    for i in mylist:
        
        ovr = []
        for j in range(9):
            values = [random.gauss(0, sd), random.gauss(0, sd),
                      random.gauss(0, sd), random.gauss(0, sd)]
            for c in range(4):
                newvalues = [checker(i,c)*random.gauss(betas[j][0][c], sd),
                           checker(i,c)*random.gauss(betas[j][1][c], sd),
                           checker(i,c)*random.gauss(betas[j][2][c], sd),
                           checker(i,c)*random.gauss(betas[j][3][c], sd) ]
            for z in range(3):
                values[z] = values[z] + newvalues[z]
            ovr = ovr + values
        if a == 0:
            final = ovr
        else:
            final = numpy.column_stack((final, ovr))

        # add class information to numpy array
        for ii in range(4):
            classes_actual[a][ii] = checker(i, ii)
            
        a = a+1
    # print i, numpy.mean(ovr)

    # transpose shape for classes for convenience
    final = numpy.transpose(final)
    return final, classes_actual
    
