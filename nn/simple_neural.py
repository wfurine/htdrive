import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv


drive_data_path = '../output_tests.csv'
class_data_path = '../output_classes.csv'
model_save_path = '../multilayer_perceptron.ckpt'

## ******************************
## DATA PREPROCESSING AHEAD HERE
## ******************************


data_drive = genfromtxt(drive_data_path, delimiter=',')
data_classes = genfromtxt(class_data_path, delimiter=',')

# print data_drive
# print data_classes


X_train, X_test, y_train, y_test = train_test_split(data_drive, data_classes, 
                                          test_size=0.20, random_state = 420)

num_train = X_train.shape[0]
num_test = X_test.shape[0]


# Save the testing data into files for later use...
with open("test_driver.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(X_test)


with open("test_classes.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(y_test)


## ******************************
##    NETWORK  PARAMETERS   HERE
## ******************************

learning_rate = 0.05
training_epochs = 80
batch_size = 1
batch_total = int(num_train/batch_size)
display_step = 10
test_step = 10

## ******************************
## MACHINE LEARNING SECTION AHEAD
## ******************************

# SIMPLE NEURAL NETWORK
# This neural network will just be composed of two hidden, fully connected
# layers. Basically the implementation of a multilayer perceptron machine. 
# It can learn non-linear relationships (hopefully)

dim_input = 36
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
# error_function = 0.5 * tf.reduce_sum(tf.sub(output, outputs_actual) \
#   * tf.sub(output, outputs_actual))

error_function = -tf.reduce_sum( (  (outputs_actual*tf.log(output + 1e-9)) 
    + ((1-outputs_actual) * tf.log(1 - output + 1e-9)) )  , name='xentropy' )    

train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(error_function)
sess.run(tf.initialize_all_variables())




## **********************************
##   ACTUAL NETWORK DEPLOYMENT HERE
## **********************************

# Convert numpy array [x1, x2, x3...] into
# [0, 1, 0, 1, 1, ...]
def threshold_array(array, middle=0.5):
    for x in xrange(array.shape[0]):
        for y in xrange(array.shape[1]):
            if array[x][y] < middle:
                array[x][y] = 0
            if array[x][y] >= middle:
                array[x][y] = 2 * middle

    return array


for epoch in range(training_epochs):
    avg_cost = 0.
    
    # Loop over all batches
    for i in range(batch_total):
        # we want to take slice of length batch_size
        I = [i]
        X = X_train[I, :]
        y = y_train[I, :]

        # Run backprop and cost operation to get loss value
        _, cost = sess.run([train_step, error_function], 
                            feed_dict= {inputs: X,
                                        outputs_actual: y})
        # Compute average loss
        avg_cost += cost / batch_total

    # Display some output every epoch
    if epoch % display_step == 0:
        print("Epoch:", '%06d' % (epoch), "cost=", \
            "{:.9f}".format(avg_cost))

    if epoch % test_step == 0:
        test_total = X_test.shape[0]
        result = output
        predictions = result.eval(feed_dict={inputs: X_test,})
        predictions = threshold_array(predictions)
        correct = 0
        for i in xrange(test_total):
            if np.array_equal(y_test[i], predictions[i]):
                correct = correct + 1

        perfect_accuracy = correct / test_total
        print("Testing accuracy: %06d / %06d".format(correct, test_total))
        print("Percentage: %.9f" % perfect_accuracy)

        





print("Optimization Finished!")

# Save the model for testing later
saver = tf.train.Saver()
save_path = saver.save(sess, model_save_path)
print("Model saved for future use in: %s" % save_path)



