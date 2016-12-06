import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



## ******************************
## DATA PREPROCESSING AHEAD HERE
## ******************************






















## ******************************
## MACHINE LEARNING SECTION AHEAD
## ******************************

# SIMPLE NEURAL NETWORK
# This neural network will just be composed of two hidden, fully connected
# layers. Basically the implementation of a multilayer perceptron machine. 
# It can learn non-linear relationships (hopefully)

dim_input = 100
dim_layer1 = 150
dim_layer2 = 250
dim_output = 4


sess = tf.InteractiveSession()

# [None] is used because we might change up this size, by using batch_size
# a batch of inputs of dim_input size

inputs = tf.placeholder(tf.float32, shape=[None, dim_input])
outputs = tf.placeholder(tf.float32, shape=[None, dim_output])
outputs_actual = tf.placeholder(tf.float32, shape=[None, dim_output])




# connect inputs to hidden units
# also, initialize weights with random numbers
weights_1 = tf.Variable(tf.truncated_normal([dim_input, dim_layer1]))
biases_1 = tf.Variable(tf.zeros([dim_layer1]))
layer_1_outputs = tf.nn.sigmoid(tf.matmul(inputs, weights_1) + biases_1)

weights_2 = tf.Variable(tf.truncated_normal([dim_layer1, dim_layer2]))
biases_2 = tf.Variable(tf.zeros([dim_layer2]))
layer_2_outputs = tf.nn.sigmoid(tf.matmul(layer_1_outputs, weights_2) + biases_2)



weights_3 = tf.Variable(tf.truncated_normal([dim_layer2, dim_output]))
biases_3 = tf.Variable(tf.zeros([dim_output]))
output = tf.nn.sigmoid(tf.matmul(layer_2_outputs, weights_3) + biases_3)

# [!] The error function chosen is good for multiclass classifications
# takes the difference of all of the classes in the output
error_function = 0.5 * tf.reduce_sum(tf.sub(output, outputs_actual) \
  * tf.sub(output, outputs_actual))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(error_function)

sess.run(tf.initialize_all_variables())

training_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

training_outputs = [[0.0], [1.0], [1.0], [0.0]]

for i in range(20000):
    _, loss = sess.run([train_step, error_function],
                       feed_dict={inputs: np.array(training_inputs),
                                  outputs_actual: np.array(training_outputs)})
    print(loss)




# This is for the case of XOR network modeling with simple feedforward perceptron. 
# print(sess.run(logits, feed_dict={inputs: np.array([[0.0, 0.0]])}))
# print(sess.run(logits, feed_dict={inputs: np.array([[0.0, 1.0]])}))
# print(sess.run(logits, feed_dict={inputs: np.array([[1.0, 0.0]])}))
# print(sess.run(logits, feed_dict={inputs: np.array([[1.0, 1.0]])}))



