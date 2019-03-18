#coding: utf-8

#This file is the sample code of lecture opt4_5.py
#based on lecture-opt4_4.py, use exponential decay learning rate.

import tensorflow as tf

TOTAL_STEPS = 40
LR_BASE     = 0.1
LR_DECAY    = 0.99
LR_STEP     = 1

global_step   = tf.Variable(0, trainable = False)
learning_rate = tf.train.exponential_decay(LR_BASE, global_step, LR_STEP, LR_DECAY, staircase = True)

w = tf.Variable(tf.constant(5, dtype=tf.float32))

loss = tf.square(w + 1)

#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as minWSess:
	init_op = tf.global_variables_initializer()
	minWSess.run(init_op)
	for i in range(TOTAL_STEPS):
		minWSess.run(train_step)
		lr_val = minWSess.run(learning_rate)
		gs_val = minWSess.run(global_step)
		w_val = minWSess.run(w)
		loss_val = minWSess.run(loss)
		print "After %d steps: global_step is %f, w is %f;" %(i, gs_val, w_val)
		print "learning_rate is %f, loss is %f." %(lr_val, loss_val)