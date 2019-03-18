#coding: utf-8

#This file is the sample code of lecture opt4_8_backward.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import generate_dataset as gends
import fw_propagation as fwp

BATCH_SIZE  = 30
TOTAL_STEPS = 40000


LR_BASE = 0.001
LR_DECAY = 0.999
REGULARIZER = 0.01

def bk_propagation():
	x = tf.placeholder(tf.float32, shape = (None, 2))
	y_ = tf.placeholder(tf.float32, shape = (None, 1))

	X, Y_, Y_c = gends.generateds()

	y = fwp.fw_propagation(x, REGULARIZER)

	global_step = tf.Variable(0, trainable = False)

	learning_rate = tf.train.exponential_decay(
		LR_BASE,
		global_step,
		gends.INPUT_SIZE / BATCH_SIZE,
		LR_DECAY,
		staircase = True)

	loss_mse = tf.reduce_mean(tf.square(y - y_))
	loss = loss_mse

	#ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
	#loss_cem = tf.reduce_mean(ce)
	#loss = loss_cem

	loss_total = loss + tf.add_n(tf.get_collection("losses"))

	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)
	#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_total, global_step = global_step)

	EMA_DECAY = 0.1
	ema = tf.train.ExponentialMovingAverage(EMA_DECAY, global_step)
	ema_op = ema.apply(tf.trainable_variables())

	with tf.control_dependencies([train_step, ema_op]):
		train_op = tf.no_op(name = "train")


	with tf.Session() as nnSess:
		init_op = tf.global_variables_initializer()
		nnSess.run(init_op)

		for i in range(TOTAL_STEPS):
			start = (i * BATCH_SIZE) % gends.INPUT_SIZE
			end = start + BATCH_SIZE
			nnSess.run(train_op, feed_dict = {x:X[start:end], y_:Y_[start:end]})

			if i % 2000 == 0:
				loss_v = nnSess.run(loss_total, feed_dict = {x:X, y_:Y_})
				print "After %d steps, loss is %f." %(i, loss_v)

		xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
		grid = np.c_[xx.ravel(), yy.ravel()]
		probs = nnSess.run(y, feed_dict = {x:grid})
		probs = probs.reshape(xx.shape)

	plt.scatter(X[:,0], X[:,1], c = np.squeeze(Y_c))
	plt.contour(xx, yy, probs, levels = [.5])
	plt.show()

if __name__ == "__main__":
	bk_propagation()