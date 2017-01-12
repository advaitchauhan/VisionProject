'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops

# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 =  256 # 2nd layer number of features
n_hidden_3 = 256 
n_hidden_4 = 256 
n_hidden_5 = 256 
n_hidden_6 = 256 
n_hidden_7 = 256 
n_hidden_8 = 256 
n_hidden_9 = 256 
n_hidden_10 = 256 
n_hidden_11 = 256 
n_hidden_12 = 256 
n_hidden_13 = 256 
n_hidden_14 = 256 
n_hidden_15 = 256 
n_hidden_16 = 256 
n_hidden_17 = 256 
n_hidden_18 = 256 
n_hidden_19 = 256 
n_hidden_20 = 256 

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# define gradient noise scale
gradient_noise_scale = 1.0

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):

    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    # Hidden layer with RELU activation
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.relu(layer_5)
    # Hidden layer with RELU activation
    layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    layer_6 = tf.nn.relu(layer_6)
    # Hidden layer with RELU activation
    layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
    layer_7 = tf.nn.relu(layer_7)
    # Hidden layer with RELU activation
    layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
    layer_8 = tf.nn.relu(layer_8)
    # Hidden layer with RELU activation
    layer_9 = tf.add(tf.matmul(layer_8, weights['h9']), biases['b9'])
    layer_9 = tf.nn.relu(layer_9)
    # Hidden layer with RELU activation
    layer_10 = tf.add(tf.matmul(layer_9, weights['h10']), biases['b10'])
    layer_10 = tf.nn.relu(layer_10)
    # Hidden layer with RELU activation
    layer_11 = tf.add(tf.matmul(layer_10, weights['h11']), biases['b11'])
    layer_11 = tf.nn.relu(layer_11)
    # Hidden layer with RELU activation
    layer_12 = tf.add(tf.matmul(layer_11, weights['h12']), biases['b12'])
    layer_12 = tf.nn.relu(layer_12)
    # Hidden layer with RELU activation
    layer_13 = tf.add(tf.matmul(layer_12, weights['h13']), biases['b13'])
    layer_13 = tf.nn.relu(layer_13)
    # Hidden layer with RELU activation
    layer_14 = tf.add(tf.matmul(layer_13, weights['h14']), biases['b14'])
    layer_14 = tf.nn.relu(layer_14)
    # Hidden layer with RELU activation
    layer_15 = tf.add(tf.matmul(layer_14, weights['h15']), biases['b15'])
    layer_15 = tf.nn.relu(layer_15)
    # Hidden layer with RELU activation
    layer_16 = tf.add(tf.matmul(layer_15, weights['h16']), biases['b16'])
    layer_16 = tf.nn.relu(layer_16)
    # Hidden layer with RELU activation
    layer_17 = tf.add(tf.matmul(layer_16, weights['h17']), biases['b17'])
    layer_17 = tf.nn.relu(layer_17)
    # Hidden layer with RELU activation
    layer_18 = tf.add(tf.matmul(layer_17, weights['h18']), biases['b18'])
    layer_18 = tf.nn.relu(layer_18)
    # Hidden layer with RELU activation
    layer_19 = tf.add(tf.matmul(layer_18, weights['h19']), biases['b19'])
    layer_19 = tf.nn.relu(layer_19)
    # Hidden layer with RELU activation
    layer_20 = tf.add(tf.matmul(layer_19, weights['h20']), biases['b20'])
    layer_20 = tf.nn.relu(layer_20)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_20, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),
    'h7': tf.Variable(tf.random_normal([n_hidden_6, n_hidden_7])),
    'h8': tf.Variable(tf.random_normal([n_hidden_7, n_hidden_8])),
    'h9': tf.Variable(tf.random_normal([n_hidden_8, n_hidden_9])),
    'h10': tf.Variable(tf.random_normal([n_hidden_9, n_hidden_10])),
    'h11': tf.Variable(tf.random_normal([n_hidden_10, n_hidden_11])),
    'h12': tf.Variable(tf.random_normal([n_hidden_11, n_hidden_12])),
    'h13': tf.Variable(tf.random_normal([n_hidden_12, n_hidden_13])),
    'h14': tf.Variable(tf.random_normal([n_hidden_13, n_hidden_14])),
    'h15': tf.Variable(tf.random_normal([n_hidden_14, n_hidden_15])),
    'h16': tf.Variable(tf.random_normal([n_hidden_15, n_hidden_16])),
    'h17': tf.Variable(tf.random_normal([n_hidden_16, n_hidden_17])),
    'h18': tf.Variable(tf.random_normal([n_hidden_17, n_hidden_18])),
    'h19': tf.Variable(tf.random_normal([n_hidden_18, n_hidden_19])),
    'h20': tf.Variable(tf.random_normal([n_hidden_19, n_hidden_20])),

    'out': tf.Variable(tf.random_normal([n_hidden_20, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'b6': tf.Variable(tf.random_normal([n_hidden_6])),
    'b7': tf.Variable(tf.random_normal([n_hidden_7])),
    'b8': tf.Variable(tf.random_normal([n_hidden_8])),
    'b9': tf.Variable(tf.random_normal([n_hidden_9])),
    'b10': tf.Variable(tf.random_normal([n_hidden_10])),
    'b11': tf.Variable(tf.random_normal([n_hidden_11])),
    'b12': tf.Variable(tf.random_normal([n_hidden_12])),
    'b13': tf.Variable(tf.random_normal([n_hidden_13])),
    'b14': tf.Variable(tf.random_normal([n_hidden_14])),
    'b15': tf.Variable(tf.random_normal([n_hidden_15])),
    'b16': tf.Variable(tf.random_normal([n_hidden_16])),
    'b17': tf.Variable(tf.random_normal([n_hidden_17])),
    'b18': tf.Variable(tf.random_normal([n_hidden_18])),
    'b19': tf.Variable(tf.random_normal([n_hidden_19])),
    'b20': tf.Variable(tf.random_normal([n_hidden_20])),

    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y) + 1e-9)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
grads_and_vars = optimizer.compute_gradients(cost)
gradients, variables = zip(*grads_and_vars)
noisy_gradients = []
for gradient in gradients:
    if gradient is None:
      noisy_gradients.append(None)
      continue
    if isinstance(gradient, ops.IndexedSlices):
      gradient_shape = gradient.dense_shape
    else:
      gradient_shape = gradient.get_shape()
    noise = random_ops.truncated_normal(gradient_shape) * gradient_noise_scale
    noisy_gradients.append(gradient + noise)

noisy_grads_and_vars = list(zip(noisy_gradients, variables))
train_opt = optimizer.apply_gradients(noisy_grads_and_vars)


# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_opt, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

