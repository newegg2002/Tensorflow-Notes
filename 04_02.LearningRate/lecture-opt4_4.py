#coding: utf-8

#This file is the sample code of lecture opt4_4.py
#set loss function loss = (w + 1)^2, w = 5, find the best w value by backfoward propagation
#which mean the right w value for minimum loss (approaching 0)

import tensorflow as tf

TOTAL_STEPS = 40

w = tf.Variable(tf.constant(5, dtype=tf.float32))

loss = tf.square(w + 1)

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as minWSess:
	init_op = tf.global_variables_initializer()
	minWSess.run(init_op)
	for i in range(TOTAL_STEPS):
		minWSess.run(train_step)
		w_val = minWSess.run(w)
		loss_val = minWSess.run(loss)
		print "After %d steps: w is %f, loss is %f." %(i, w_val, loss_val)