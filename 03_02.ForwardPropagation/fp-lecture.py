#encoding: utf-8

#This file is the sample code of lecture

import tensorflow as tf

#x as input, and w as parameters

#set x as constant
#x   = tf.constant([[0.7, 0.5]])

#give x placeholder first, we'll feed the real value at running
#x = tf.placeholder(tf.float32, shape=(1, 2))
x = tf.placeholder(tf.float32, shape=(None, 2))

w1  = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2  = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

#forward propagation process: a = xw1, y = aw2
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#run the session
with tf.Session() as fpsess:
	init_op = tf.global_variables_initializer()
	fpsess.run(init_op)
	print "Parameter w1 = ", fpsess.run(w1)
	print "Parameter w2 = ", fpsess.run(w2)
	#print "The result y = ", fpsess.run(y)
	#print "The result y = ", fpsess.run(y, feed_dict={x:[[0.7, 0.5]]})
	print "The result y = ", fpsess.run(y, feed_dict={x:[[0.7, 0.5], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]})

