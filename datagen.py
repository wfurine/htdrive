import random
import numpy
from operator import add
import csv
betas = [[[0, 0, 0, 0],
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
         [0, 0, 0, 0]]]
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



def function(n, betas):
    final  = []
    mylist = genclass(n)
    a = 0
    for i in mylist:
        a = a+1
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
        if(a==1):
            final = ovr
        else:
            final = numpy.column_stack((final, ovr))
        print i, numpy.mean(ovr)

    with open("output.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(final)
        
    
print function(10, betas)
