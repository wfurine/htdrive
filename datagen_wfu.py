import numpy as np

# TODO: Add data generation.
# Sorry in advance about bad syntax and non-pythonic practices
# I still don't really use python all that much compared 
# to other languages. 

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



