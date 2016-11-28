import random
import numpy
import math

# Generate random list of people with any combination of classes
def gen_class(n):
	classes = ['1000', '1100', '1110', '1101', '1010', '1001', '1011', '1111', '0100', '0110', '0101', '0111', '0010', '0011', '0001', '0000'] 
	classified = []
	for i in range(n):
		classified.append(random.choice(classes))
	return classified

# Generate random values to satisfy overall mean conditions
def gen(n, mean, sd):
	values = [round(random.gauss(mean, sd), 6) for i in range(n)]
	return values

# Those who fit into the reaction rate class have mean value 1 + beta + offset
# Mean overall is 1
def function(n, beta):
	mylist = gen_class(n)
	rxn_rate = 0
	offset = 0.3
	target = 1 + beta*(offset)
	for x in mylist:
		if x[3]=='1':
			rxn_rate = rxn_rate + 1
	valuesb = gen(rxn_rate, target, 0.2)
	if ((n-target*rxn_rate)<0):
		return "Not possible for this value of beta"
	valuesn = gen(n-rxn_rate, (n - target*rxn_rate)/(n - rxn_rate), 0.2)
	newvalues = []
	b = 0
	n = 0
	for x in mylist:
		if x[3]=='1':
			newvalues.append(valuesb[b])
			b = b+1
		else:
			newvalues.append(valuesn[n])
			n = n+1
	print reduce(lambda x, y: x + y, valuesb) / len(valuesb)
	print target
	print reduce(lambda x, y: x + y, newvalues) / len(newvalues)
	return newvalues

print function(100, .5)