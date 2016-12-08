import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import csv


test_data_path = 'test_driver.csv'
test_class_path = 'test_classes.csv'
model_path = '/home/wfu/ml/htdrive/multilayer_perceptron.ckpt'

X_test = genfromtxt(test_data_path)
y_test = genfromtxt(test_class_path)



## Load in our previous neural network trained model
## Use our test data as testing to give a percentage

saver = tf.train.Saver()
sess = tf.Session()

saver.restore(sess, model_path)
print("Model loaded.")



# Since we are using multilabel, we will find the percentage of 
# classifications that are 100% correct, so if we guessed randomly
# the percentage would be around 6%

