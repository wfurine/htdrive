import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# SIMPLE AUTOENCODER
# An autoencoder is an unsupervised ML technique that tries to learn 
# a representation of the data that is the most efficient, 
# so we can reduce the dimensionality of the data.




# Import data here, I will use MNIST data as a placeholder right now
# MNIST is like Hello World basically, replace with whatever data
# we are going to feed the neural network
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)


# Neural Net Parameters
lr = 0.001
num_epochs = 100
batch_size = 16
display_step = 10
num_examples = 10

# Layer Parameters
dim_hidden1 = 128 
dim_hidden2 = 64
dim_input = 784

# tensorflow Input
X = tf.placeholder("float", [None, dim_input])

# Weight and biases definition and initialization
weights = {
	'encoder_h1': tf.Variable(tf.random_normal([dim_input, dim_hidden1])),
	'encoder_h2': tf.Variable(tf.random_normal([dim_hidden1, dim_hidden2])),
	'decoder_h1': tf.Variable(tf.random_normal([dim_hidden2, dim_hidden1])),
	'decoder_h2': tf.Variable(tf.random_normal([dim_hidden1, dim_input])),
}

biases = {
	'encoder_b1': tf.Variable(tf.random_normal([dim_hidden1])),
	'encoder_b2': tf.Variable(tf.random_normal([dim_hidden2])),
	'decoder_b1': tf.Variable(tf.random_normal([dim_hidden1])),
	'decoder_b2': tf.Variable(tf.random_normal([dim_input])),
}


# definition of encoder and decoders 
# we are using sigmoid activiation function here
def encoder(x):
	# Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


# model construction 
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# prediction
# y will be the prediction, y_ will be the actual
y = decoder_op
y_ = X 

# Cost function 
cost = tf.reduce_mean(tf.pow(y_ - y, 2))
optimizer = tf.train.RMSPropOptimizer(lr).minimize(cost)

# Initialize everything 
init = tf.initialize_all_variables()



# Tensorflow stuff
session = tf.InteractiveSession()
session.run(init)

total_batch = int(mnist.train.num_examples/batch_size)
for epoch in range(num_epochs):
    # Loop over total number of batches it takes to use all of the data in training set
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Run backpropogation, trying to minimize the COST variable
        _, c = session.run([optimizer, cost], feed_dict={X: batch_xs})
    # Display intermediate results to screen.
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(c))


