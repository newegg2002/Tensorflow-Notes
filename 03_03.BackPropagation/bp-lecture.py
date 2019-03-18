#encoding: utf-8

#This file is the sample code of lecture

import tensorflow as tf
import numpy as np

BATCH_SIZE   = 8
INPUT_SIZE   = 32
seed         = 23455
TOTAL_STEPS  = 3000

#get the random input data set X with seed
rng = np.random.RandomState(seed)
X   = rng.rand(INPUT_SIZE, 2)
#get the expected output data set Y
Y = [[int(x0 + x1 <1)] for (x0, x1) in X]

print "X:\n", X
print "Y:\n", Y

#input data x and expected output y_ placeholder
x   = tf.placeholder(tf.float32, shape=(None, 2))
y_  = tf.placeholder(tf.float32, shape=(None, 1))

#parameters: w1 & w2
w1  = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2  = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

#forward propagation process: a = xw1, y = aw2
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#define loss and back propagation
loss       = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss) 
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss) 

#run the session:
with tf.Session() as bpsess:
	init_op = tf.global_variables_initializer()
	bpsess.run(init_op)

	print "Parameters before training:"
	print "Parameter w1 = \n", bpsess.run(w1)
	print "Parameter w2 = \n", bpsess.run(w2)
	print "\n"

	for i in range(TOTAL_STEPS):
		start = (i * BATCH_SIZE) % INPUT_SIZE
		end   = start + BATCH_SIZE
		bpsess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
		if i % 500 == 0:
			total_loss = bpsess.run(loss, feed_dict={x:X, y_:Y})
			print "After %d times training, loss on all data is %g" %(i, total_loss)

	print "Parameters finish training:"
	print "Parameter w1 = \n", bpsess.run(w1)
	print "Parameter w2 = \n", bpsess.run(w2)
