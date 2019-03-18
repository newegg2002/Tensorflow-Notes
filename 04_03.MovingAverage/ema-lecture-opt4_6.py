#coding: utf-8

#This file is the sample code of lecture opt4_6.py
# Using EMA(exponential moving average) to perform parameters(w and b) optimizing.

import tensorflow as tf

#define w1 as the parameter to optimize
w1 = tf.Variable(0, dtype = tf.float32)

#
global_step = tf.Variable(0, trainable = False)

EMA_DECAY = 0.1
ema = tf.train.ExponentialMovingAverage(EMA_DECAY, global_step)

#parameter of ema.apply() is the list of all trainable variables.
#ema_op = ema.apply([w1])
ema_op = ema.apply(tf.trainable_variables())

#run the seesion
with tf.Session() as emaSess:
	init_op = tf.global_variables_initializer()
	emaSess.run(init_op)

	print "Before running moving average:"
	print emaSess.run([w1, ema.average(w1)])

	print "W1 = 1:"
	emaSess.run(tf.assign(w1, 1))
	emaSess.run(ema_op)
	print emaSess.run([w1, ema.average(w1)])

	print "global_step = 100, W1 = 10:"
	emaSess.run(tf.assign(global_step, 100))
	emaSess.run(tf.assign(w1, 10))
	emaSess.run(ema_op)
	print emaSess.run([w1, ema.average(w1)])


	emaSess.run(ema_op)
	print emaSess.run([w1, ema.average(w1)])

	emaSess.run(ema_op)
	print emaSess.run([w1, ema.average(w1)])

	emaSess.run(ema_op)
	print emaSess.run([w1, ema.average(w1)])
	
	emaSess.run(ema_op)
	print emaSess.run([w1, ema.average(w1)])

	emaSess.run(ema_op)
	print emaSess.run([w1, ema.average(w1)])