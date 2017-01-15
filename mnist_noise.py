import math
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops

def percept10(mnist, noise, lr, n_hidden, numEpochs, wdev, bdev):

    # Parameters
    learning_rate = lr
    training_epochs = numEpochs
    batch_size = 200
    display_step = 1
    
    # Network Parameters
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
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden], stddev = wdev )),
        'h2': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = wdev )),
        'h3': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = wdev)),
        'h4': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = wdev)),
        'h5': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = wdev)),
        'h6': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = wdev)),
        'h7': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = wdev)),
        'h8': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = wdev)),
        'h9': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = wdev)),
        'h10': tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev = wdev)),
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], stddev = wdev)),
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden], stddev = bdev)),
        'b2': tf.Variable(tf.random_normal([n_hidden], stddev = bdev)),
        'b3': tf.Variable(tf.random_normal([n_hidden], stddev = bdev)),
        'b4': tf.Variable(tf.random_normal([n_hidden], stddev = bdev)),
        'b5': tf.Variable(tf.random_normal([n_hidden], stddev = bdev)),
        'b6': tf.Variable(tf.random_normal([n_hidden], stddev = bdev)),
        'b7': tf.Variable(tf.random_normal([n_hidden], stddev = bdev)),
        'b8': tf.Variable(tf.random_normal([n_hidden], stddev = bdev)),
        'b9': tf.Variable(tf.random_normal([n_hidden], stddev = bdev)),
        'b10': tf.Variable(tf.random_normal([n_hidden], stddev = bdev)),
        'out': tf.Variable(tf.random_normal([n_classes], stddev = bdev)),
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
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

#An experiment looking at three initilizations with and without noise
def testMain():

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    n = 2     #how many trials per experiment?
    eps = 2  #how many epochs per net?
    
    #simple init
    noise_init0 = [0]*n
    znoise_init0 = [0]*n
    
    #good init
    noise_init1 = [0]*n
    znoise_init1 = [0]*n
    
    #my init
    noise_initm = [0]*n
    znoise_initm = [0]*n
    #my init tests (re)

    print("my init tests")
    for i in range(n):
        znoise_initm[i] = percept10(mnist, noise=0, lr =.005, n_hidden = 50, numEpochs = eps, wdev = 1/math.sqrt(50), bdev = 1/math.sqrt(50))
        noise_initm[i] = percept10(mnist, noise=1, lr =.005, n_hidden = 50, numEpochs = eps, wdev =  1/math.sqrt(50), bdev = 1/math.sqrt(50))
    
    
    #simple init tests (.01 stdev for weights and biases)
    print("simple init tests")
    for i in range(n):
        znoise_init0[i] = percept10(mnist, noise=0, lr =.005, n_hidden = 50, numEpochs = eps, wdev = .01, bdev = .01)
        noise_init0[i] = percept10(mnist, noise=1, lr =.005, n_hidden = 50, numEpochs = eps, wdev = .01, bdev = .01)
        
    #good init (derived in He et. al) tests
    print("good init tests")
    for i in range(n):
        znoise_init1[i] = percept10(mnist, noise=0, lr =.005, n_hidden = 50, numEpochs = eps, wdev = math.sqrt(2/50), bdev = 0)
        noise_init1[i] = percept10(mnist, noise=1, lr =.005, n_hidden = 50, numEpochs = eps, wdev =  math.sqrt(2/50), bdev = 0)
        
    return noise_init0, znoise_init0, noise_init1, znoise_init1, noise_initm, znoise_initm


n_i0, zn_i0, n_i1, zn_i1, n_im, zn_im = testMain()

import pickle

# obj0, obj1, obj2 are created here...

# Saving the objects:
with open('mnist_results.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([n_i0, zn_i0, n_i1, zn_i1, n_im, zn_im], f)

# Getting back the objects:
with open('mnist_results.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    n_i0, zn_i0, n_i1, zn_i1, n_im, zn_im = pickle.load(f)


