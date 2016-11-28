import random
import numpy

def gen(n, mean, sd):
    values = [random.gauss(mean, sd) for i in range(n)]
    return values

def genclass(n):
    classes = ['1000', '1100', '1110', '1101', '1010', '1001', '1011', '1111', '0100', '0110', '0101', '0111', '0010', '0011', '0001', '0000']
    people = []
    for i in range(n):
        people.append(random.choice(classes))
    return people


def function(n, beta):
    mylist = genclass(n)
    motor = 0 ## number of people in motor control class
    for x in mylist:
        if x[0]=='1':
            motor = motor + 1
    valuesb = gen(motor, beta, 0.2) ## generating motor numbers with mean beta
    valuesn = gen(n-motor, 0, 0.2) ## generating normal people with mean 0
    newvalues = []
    b = 0
    n = 0
    for x in mylist:
        if x[0]=='1':
            newvalues.append(valuesb[b])
            b = b+1
        else:
            newvalues.append(valuesn[n])
            n = n+1
    return newvalues
    
    
print function(100, 1)
