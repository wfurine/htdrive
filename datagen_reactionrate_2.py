import random
import numpy
import math

# README: the math is wrong in calculating the average of all the people classified as xxx0
# In other words, those who don't fall into the xxx1 category (anyone with poor reaction rates)
# So the overall mean is NOT 1. Need to figure out the math for the 2nd argument of gen() on line 84

# Generate random list of people with any combination of classes
def gen_class(n):
	classes = ['1000', '1100', '1110', '1101', '1010', '1001', '1011', '1111', '0100', '0110', '0101', '0111', '0010', '0011', '0001', '0000'] 
	classified = []
	x = 0
	while x < n:
		selected_class = random.choice(classes)
		classified.append(selected_class)
		x = x + 1
	return classified

# Generate random values to satisfy overall mean conditions
def gen(n, mean):
    values = [random.random() for i in range(n)]
    truevalues = []
    for x in values:
        y = round((x*mean/numpy.mean(values)), 6)
        truevalues.append(y)
    return truevalues

# Those who fit into the reaction rate class have mean value 1 + beta + offset
# Mean overall is 1
def function(n, beta):
	mylist = gen_class(n)
	rxn_rate = 0
	rxns = []
	off_0 = 0.4
	off_1 = 0.25
	off_2 = 0.1
	off_3 = 0.55
	target_0 = 1 + beta*(off_3)
	target_1 = 1 + beta*(off_2 + off_3)
	target_2 = 1 + beta*(off_1 + off_3)
	target_3 = 1 + beta*(off_1 + off_2 + off_3)
	target_4 = 1 + beta*(off_0 + off_3)
	target_5 = 1 + beta*(off_0 + off_2 + off_3)
	target_6 = 1 + beta*(off_0 + off_1 + off_3)
	target_7 = 1 + beta*(off_0 + off_1 + off_2 + off_3)
	rxn_0 = 0
	rxn_1 = 0
	rxn_2 = 0
	rxn_3 = 0
	rxn_4 = 0
	rxn_5 = 0
	rxn_6 = 0
	rxn_7 = 0
	rxns = rxn_0 + rxn_1 + rxn_2 + rxn_3 + rxn_4 + rxn_5 + rxn_6 + rxn_7

	for x in mylist:
		if x[3]=='1':
			off_3 = 0.7
			if x[0] == '0' and x[1] == '0' and x[2] == '0':
				rxn_0 += 1 
			if x[0] == '0' and x[1] == '0' and x[2] == '1':
				rxn_1 +=  1
			elif x[0] == '0' and x[1] == '1' and x[2] == '0':
				rxn_2 += 1
			elif x[0] == '0' and x[1] == '1' and x[2] == '1':
				rxn_3 += 1
			elif x[0] == '1' and x[1] == '0' and x[2] == '0':
				rxn_4 += 1
			elif x[0] == '1' and x[1] == '0' and x[2] == '1':
				rxn_5 += 1
			elif x[0] == '1' and x[1] == '1' and x[2] == '0':
				rxn_6 += 1
			else:
				rxn_7 += 1

	vals_0 = gen(rxn_0, target_0)
	vals_1 = gen(rxn_1, target_1)
	vals_2 = gen(rxn_2, target_2)
	vals_3 = gen(rxn_3, target_3)
	vals_4 = gen(rxn_4, target_4)
	vals_5 = gen(rxn_5, target_5)
	vals_6 = gen(rxn_6, target_6)
	vals_7 = gen(rxn_7, target_7)
	if ((n-target_0)*rxn_0)<0 or ((n-target_1)*rxn_1)<0 or ((n-target_2)*rxn_2)<0 or ((n-target_3)*rxn_3)<0 or ((n-target_4)*rxn_4)<0 or ((n-target_5)*rxn_5)<0 or ((n-target_6)*rxn_6)<0 or ((n-target_7)*rxn_7)<0:
		return "Not possible for this value of beta"

	valuesn = gen(n-rxns, (n - (target_0*rxn_0 + target_1*rxn_1 + target_2*rxn_2 + target_3*rxn_3 + target_4*rxn_4 + target_5*rxn_5 + target_6*rxn_6 + target_7*rxn_7)/(n - rxns)))
	#valuesn = gen(n-rxns, (((n - target_0*rxn_0) + (n - target_1*rxn_1) + (n - target_2*rxn_2) + (n - target_3*rxn_3) + (n - target_4*rxn_4) + (n - target_5*rxn_5) + (n - target_6*rxn_6) + (n - target_7*rxn_7))/(n - rxns)))
	newvalues = []
	rxn_0 = 0
	rxn_1 = 0
	rxn_2 = 0
	rxn_3 = 0
	rxn_4 = 0
	rxn_5 = 0
	rxn_6 = 0
	rxn_7 = 0
	i = 0
	for x in mylist:
		if x[3]=='1':
			if x[0] == '0' and x[1] == '0' and x[2] == '0':
				newvalues.append(vals_0[rxn_0])
				rxn_0 += 1
			if x[0] == '0' and x[1] == '0' and x[2] == '1':
				newvalues.append(vals_1[rxn_1])
				rxn_1 +=  1
			elif x[0] == '0' and x[1] == '1' and x[2] == '0':
				newvalues.append(vals_2[rxn_2])
				rxn_2 += 1
			elif x[0] == '0' and x[1] == '1' and x[2] == '1':
				newvalues.append(vals_3[rxn_3])
				rxn_3 += 1
			elif x[0] == '1' and x[1] == '0' and x[2] == '0':
				newvalues.append(vals_4[rxn_4])
				rxn_4 += 1
			elif x[0] == '1' and x[1] == '0' and x[2] == '1':
				newvalues.append(vals_5[rxn_5])
				rxn_5 += 1
			elif x[0] == '1' and x[1] == '1' and x[2] == '0':
				newvalues.append(vals_6[rxn_6])
				rxn_6 += 1
			else:
				newvalues.append(vals_7[rxn_7])
				rxn_7 += 1
		else:
			newvalues.append(valuesn[i])
			i = i+1
	
	print reduce(lambda x, y: x + y, vals_0) / len(vals_0)
	print target_0
	print reduce(lambda x, y: x + y, vals_1) / len(vals_1)
	print target_1
	print reduce(lambda x, y: x + y, vals_2) / len(vals_2)
	print target_2
	print reduce(lambda x, y: x + y, vals_3) / len(vals_3)
	print target_3
	print reduce(lambda x, y: x + y, vals_4) / len(vals_4)
	print target_4
	print reduce(lambda x, y: x + y, vals_5) / len(vals_5)
	print target_5
	print reduce(lambda x, y: x + y, vals_6) / len(vals_6)
	print target_6
	print reduce(lambda x, y: x + y, vals_7) / len(vals_7)
	print target_7
	print reduce(lambda x, y: x + y, newvalues) / len(newvalues)
	print mylist

	return newvalues

print function(100, .5)


