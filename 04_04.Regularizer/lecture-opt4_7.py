#coding: utf-8

#This file is the sample code of lecture opt4_7.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

SEED        = 2
BATCH_SIZE  = 30
INPUT_SIZE  = 300
TOTAL_STEPS = 40000

rdm = np.random.RandomState(SEED)
X = rdm.randn(INPUT_SIZE, 2)

Y_ = [int((x0 * x0 + x1 * x1) < 2) for (x0, x1) in X]
Y_c = ["c" if y else "k" for y in Y_]

#print "Before reshape:"
#print X
#print Y_
#print Y_c

#reshape X to n * 2 matrix, Y_ to n * 1 matrix. 
#-1 as row number means which would caculated per 2nd parameter.
X = np.vstack(X).reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)

#print "After reshape:"
print X
print Y_
print Y_c

Y_c_ = np.squeeze(Y_c)
print Y_c_

#X[:,0] is 0 column of X, X.shape is (300, 2)
plt.scatter(X[:,0], X[:,1], c = np.squeeze(Y_c))
plt.show()

def get_weight(shape, regularizer):
	w = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
	tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.constant(0.01, shape = shape))
	return b

x = tf.placeholder(tf.float32, shape = (None, 2))
y_ = tf.placeholder(tf.float32, shape = (None, 1))

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2) + b2

loss_mse = tf.reduce_mean(tf.square(y - y_))
loss_total = loss_mse + tf.add_n(tf.get_collection("losses"))

#train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

with tf.Session() as reguSess:
	init_op = tf.global_variables_initializer()
	reguSess.run(init_op)

	for i in range(TOTAL_STEPS):
		start = (i * BATCH_SIZE) % INPUT_SIZE
		end = start + BATCH_SIZE
		reguSess.run(train_step, feed_dict = {x:X[start:end], y_:Y_[start:end]})

		if i % 2000 == 0:
			loss_mse_val = reguSess.run(loss_mse, feed_dict = {x:X, y_:Y_})
			print "After %d steps, loss is %f." %(i, loss_mse_val)

	xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
	grid = np.c_[xx.ravel(), yy.ravel()]
	#print grid

	probs = reguSess.run(y, feed_dict = {x:grid})
	probs = probs.reshape(xx.shape)

	print "w1:\n", reguSess.run(w1)
	print "b1:\n", reguSess.run(b1)
	print "w2:\n", reguSess.run(w2)
	print "b2:\n", reguSess.run(b2)


plt.scatter(X[:,0], X[:,1], c = np.squeeze(Y_c))
plt.contour(xx, yy, probs, colors = "red", levels = 0.5)
plt.show()
