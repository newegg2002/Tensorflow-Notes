#encoding: utf-8

#We'll implement triangle judgement with NN
#Given integer (x1, x2, x3) as 3 sides, to judge if triangle can be composed with them 

#####0: import & constant & data set ######
import tensorflow as tf
import numpy as np

BATCH_SIZE   = 8
INPUT_SIZE   = 32
seed         = 84520
TOTAL_STEPS  = 3000

#get the random input data set X with seed
rng = np.random.RandomState(seed)
#X   = rng.randint(0, 100, [INPUT_SIZE, 3])
X   = rng.rand(INPUT_SIZE, 3)
#get the expected output data set Y
Y = [[int((x0 + x1 > x2) and (x1 + x2 > x0) and (x2 + x0 > x1))] for (x0, x1, x2) in X]

print "X:\n", X
print "Y:\n", Y

#####1: Forward Propagation: Define x, w and y  ######
#input data x and expected output y_ placeholder
x   = tf.placeholder(tf.float32, shape=(None, 3))
y_  = tf.placeholder(tf.float32, shape=(None, 1))

#parameters: w1 / w2 / w3
w1  = tf.Variable(tf.random_normal([3, 5], stddev=1, seed=1))
w2  = tf.Variable(tf.random_normal([5, 3], stddev=1, seed=1))
w3  = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

#forward propagation process: a = xw1, b = aw2, y = bw3
a = tf.matmul(x,  w1)
b = tf.matmul(a,  w2)
y = tf.matmul(b,  w3)

#####2: Back Propagation: Define loss, train step ######
#define loss and back propagation
loss       = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss) 
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss) 


#####3: Run the session, training TOTAL_STEPS times ######
#run the session:
with tf.Session() as tj_sess:
	init_op = tf.global_variables_initializer()
	tj_sess.run(init_op)

	print "Parameters before training:"
	print "Parameter w1 = \n", tj_sess.run(w1)
	print "Parameter w2 = \n", tj_sess.run(w2)
	print "Parameter w3 = \n", tj_sess.run(w3)
	print "\n"

	for i in range(TOTAL_STEPS):
		start = (i * BATCH_SIZE) % INPUT_SIZE
		end   = start + BATCH_SIZE
		tj_sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
		if i % 500 == 0:
			total_loss = tj_sess.run(loss, feed_dict={x:X, y_:Y})
			print "After %d times training, loss on all data is %g" %(i, total_loss)

	print "Parameters finish training:"
	print "Parameter w1 = \n", tj_sess.run(w1)
	print "Parameter w2 = \n", tj_sess.run(w2)
	print "Parameter w3 = \n", tj_sess.run(w3)

	print "Test result: ", tj_sess.run(y, feed_dict={x:[[3.0, 4.0, 5.0]]})
	print "Test result: ", tj_sess.run(y, feed_dict={x:[[0.01674828, 0.43797014, 0.20689225]]})