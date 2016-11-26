import random
import numpy

def gen(n, mean):
    values = [random.random() for i in range(n)]
    truevalues = []
    for x in values:
        x = x*mean/numpy.mean(values)
        truevalues.append(x)
    return truevalues

def function(n, mylist, beta):
    motor = 0 ## number of people in motor control class
    for x in mylist:
        if x[0]=='1':
            motor = motor + 1
    valuesb = gen(motor, 1+beta) ## generating motor numbers with mean 1 + beta
    if((n-(1+beta)*motor)<0): ## limit for beta (or else values will be negative)
        return "Not possible for this value of beta"
    valuesn = gen(n-motor, (n - (1+beta)*motor)/(n - motor)) ## generating rest of values so that total mean is 1
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
    
    
print function(5, ['1000', '0111','1000','1011','0011'], .5)
