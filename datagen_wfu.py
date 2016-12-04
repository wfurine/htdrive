import numpy as np
import random

# TODO: Add data generation.
# Sorry in advance about bad syntax and non-pythonic practices
# I still don't really use python all that much compared 
# to other languages. 
# Also, using python classes to generate these probably requires a huge amount of overhead
# but at this point I don't really care...

def construct_dataset():
	driverlist = []
	for i in xrange(10000):

	return data




class Test:
	reaction_time_stop = 0
	speed_variability = 0
	road_variability = 0
	reaction_time_emergency = 0	
	reaction_time_peripheral = 0
	
	


class Driver:

	line_of_sight = 0        
	motor_control = 0
	situational_awareness = 0
	situational_lapse = 0 
	test_list = []

	def __init__():
		# default constructor
		line_of_sight = 0
		motor_control = 0
		situational_awareness = 0
		situational_lapse = 0
		test_list = []
	
	def __init__(los, m_control, sit_aware, sit_lapse):
		line_of_sight = los
		motor_control = m_control
		situational_awareness = sit_aware
		situational_lapse = sit_lapse
		test_list = []

	def return_numpy_array():
		x = np.zeros(5)
		if reaction_time_stop == 1:
			x[0] = 1
		if motor_control == 1:
			x[1] = 1
		if road_variability == 1:
			x[2] = 1
		if reaction_time_emergency == 1:
			x[3] = 1
		if reaction_time_peripheral == 1:
			x[4] = 1



def gen(n, mean, sd):
    values = [random.gauss(mean, sd) for i in range(n)]
    return values


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
    



