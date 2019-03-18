#encoding: utf-8

#This file is the sample code of lecture opt4_1.py

import tensorflow as tf
import numpy as np

BATCH_SIZE   = 8
INPUT_SIZE   = 32
SEED         = 23455
TOTAL_STEPS  = 20000 

#get the random input data set X with seed
rng = np.random.RandomState(SEED)
X   = rng.rand(INPUT_SIZE, 2)
#get the expected output data set Y
Y = [[x0+x1+(rng.rand()/10.0-0.05)] for (x0, x1) in X]

print "X:\n", X
print "Y:\n", Y

#input data x and expected output y_ placeholder
x   = tf.placeholder(tf.float32, shape=(None, 2))
y_  = tf.placeholder(tf.float32, shape=(None, 1))

#parameters: w1
w1  = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))

#forward propagation process: y = xw1
y = tf.matmul(x, w1)

#define loss and back propagation
#loss       = tf.reduce_mean(tf.square(y - y_))
#use custom loss function, as lecture sample code opt4_2.py
COST   = 9
PROFIT = 1
loss       = tf.reduce_sum(tf.where(tf.greater(y, y_), COST * (y - y_), PROFIT * (y_ - y)))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss) 
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss) 

#run the session:
with tf.Session() as milksess:
	init_op = tf.global_variables_initializer()
	milksess.run(init_op)

	print "Parameters before training:"
	print "Parameter w1 = \n", milksess.run(w1)
	print "\n"

	for i in range(TOTAL_STEPS):
		start = (i * BATCH_SIZE) % INPUT_SIZE
		end   = start + BATCH_SIZE
		milksess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
		if i % 500 == 0:
			total_loss = milksess.run(loss, feed_dict={x:X, y_:Y})
			print "After %d times training, loss on all data is %g" %(i, total_loss)
			print "w1 is\n", milksess.run(w1)

	print "Parameters finish training:"
	print "Parameter w1 = \n", milksess.run(w1)
