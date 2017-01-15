from __future__ import print_function
import math
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops


def get_sigma(eta, gamma, step):

    if step is None:
        raise ValueError("global_step is required for exponential_decay.")

    eta = ops.convert_to_tensor(eta, name="eta")
    dtype = eta.dtype
    global_step = math_ops.cast(step, dtype)

    print math_ops.div(eta, math_ops.pow(math_ops.add(math_ops.cast(1,dtype), global_step), gamma))

    return math_ops.div(eta, math_ops.pow(math_ops.add(math_ops.cast(1,dtype), global_step), gamma))



def percept10(noise, lr, n_hidden, numEpochs):

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
 
    # Parameters
    learning_rate = lr
    training_epochs = numEpochs
    batch_size = 200
    display_step = 1

    # Network Parameters
    n_hidden = 50
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_classes = 10 # MNIST total classes (0-9 digits)

    # define gradient noise scale
    gradient_noise_scale = noise

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
        #Hidden layer, RELU Activation
        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_3 = tf.nn.relu(layer_3)
        #Hidden layer, RELU Activation
        layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
        layer_4 = tf.nn.relu(layer_4)
        #Hidden layer, RELU Activation
        layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
        layer_5 = tf.nn.relu(layer_5)
        #Hidden layer, RELU Activation
        layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
        layer_6 = tf.nn.relu(layer_6)
        #Hidden layer, RELU Activation
        layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
        layer_7 = tf.nn.relu(layer_7)
        #Hidden layer, RELU Activation
        layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
        layer_8 = tf.nn.relu(layer_8)
        #Hidden layer, RELU Activation
        layer_9 = tf.add(tf.matmul(layer_8, weights['h9']), biases['b9'])
        layer_9 = tf.nn.relu(layer_9)
        #Hidden layer, RELU Activation
        layer_10 = tf.add(tf.matmul(layer_9, weights['h10']), biases['b10'])
        layer_10 = tf.nn.relu(layer_10)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_10, weights['out']) + biases['out']
        return out_layer

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'h2': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'h3': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'h4': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'h5': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'h6': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'h7': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'h8': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'h9': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'h10': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], stddev = 1/math.sqrt(n_hidden))),
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'b2': tf.Variable(tf.random_normal([n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'b3': tf.Variable(tf.random_normal([n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'b4': tf.Variable(tf.random_normal([n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'b5': tf.Variable(tf.random_normal([n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'b6': tf.Variable(tf.random_normal([n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'b7': tf.Variable(tf.random_normal([n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'b8': tf.Variable(tf.random_normal([n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'b9': tf.Variable(tf.random_normal([n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'b10': tf.Variable(tf.random_normal([n_hidden], stddev = 1/math.sqrt(n_hidden))),
        'out': tf.Variable(tf.random_normal([n_classes], stddev = 1/math.sqrt(n_hidden))),
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    step = tf.Variable(0, name='global_step', trainable=False)

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

        sigma = get_sigma(.01, 0.55, step)

        noise = random_ops.truncated_normal(gradient_shape, stddev=sigma) * gradient_noise_scale
        noisy_gradients.append(gradient + noise)

    noisy_grads_and_vars = list(zip(noisy_gradients, variables))
    train_opt = optimizer.apply_gradients(noisy_grads_and_vars, global_step=step)


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
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        acc = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
        print("Accuracy:", acc)
        return acc

#experiment where we compare gradient noise vs no gradient noise for 15 epochs 
#(return accuracy results of 10 trials)
def test1():
    noNoiseAcc = [0]*20
    noiseAcc = [0]*20
    
    #With no noise
    print("NO NOISE!!!")
    for i in range(20):
        noNoiseAcc[i] = percept10(noise=0, lr =.005, n_hidden = 50, numEpochs = 15)
        
    #With noise
    print("NOISE!!!")
    for i in range(20):
        noiseAcc[i] = percept10(noise=1, lr =.005, n_hidden = 50, numEpochs = 15)
        
    return noNoiseAcc, noiseAcc

#experiment where we compare gradient noise vs no gradient noise for 30 epochs 
#(return accuracy results of 10 trials)
def test2():
    noNoiseAcc = [0]*10
    noiseAcc = [0]*10
    
    #With no noise
    print("NO NOISE!!!")
    for i in range(10):
        noNoiseAcc[i] = percept10(noise=0, lr =.005, n_hidden = 50, numEpochs = 30)
        
    #With noise
    print("NOISE!!!")
    for i in range(10):
        noiseAcc[i] = percept10(noise=1, lr =.005, n_hidden = 50, numEpochs = 30)
        
    return noNoiseAcc, noiseAcc

noNoiseAcc, noiseAcc = test1()

print ("********************")
print ("no noise: ")
print (noNoiseAcc)

no_avg = sum(noNoiseAcc) / float(len(noNoiseAcc))
print("avg: ")
print (no_avg)

print ("********************")
print ("noise: ") 
print (noiseAcc)

avg = sum(noiseAcc) / float(len(noiseAcc))
print("avg: ")
print (avg)



