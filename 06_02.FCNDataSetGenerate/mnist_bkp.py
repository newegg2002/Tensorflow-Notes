#coding: utf-8

#This file is the sample code of lecture mnist_backward.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import mnist_fwp as fwp
import mnist_gends as gends

BATCH_SIZE  = 200 
TOTAL_STEPS = 50000

#TOTAL_NUM_EXAMPLES = mnist.train.num_examples
TOTAL_NUM_EXAMPLES = 60000

LR_BASE = 0.1
LR_DECAY = 0.99
REGULARIZER = 0.0001

EMA_DECAY = 0.99

MODEL_NAME      = "mnist_mode"
DATA_SAVE_PATH  = "./data/"
MODEL_SAVE_PATH = "./model/"

def bk_propagation(mnist):
	x = tf.placeholder(tf.float32, shape = (None, fwp.INPUT_NODE))
	y_ = tf.placeholder(tf.float32, shape = (None, fwp.OUTPUT_NODE))

	#X, Y_, Y_c = gends.generateds()

	y = fwp.fw_propagation(x, REGULARIZER)

	global_step = tf.Variable(0, trainable = False)

	#loss_mse = tf.reduce_mean(tf.square(y - y_))
	#loss = loss_mse

	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
	loss_cem = tf.reduce_mean(ce)
	loss = loss_cem

	loss_total = loss + tf.add_n(tf.get_collection("losses"))

	learning_rate = tf.train.exponential_decay(
		LR_BASE,
		global_step,
		TOTAL_NUM_EXAMPLES / BATCH_SIZE,
		LR_DECAY,
		staircase = True)


	#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_total, global_step = global_step)

	ema = tf.train.ExponentialMovingAverage(EMA_DECAY, global_step)
	ema_op = ema.apply(tf.trainable_variables())

	with tf.control_dependencies([train_step, ema_op]):
		train_op = tf.no_op(name = "train")

	saver = tf.train.Saver()
	img_batch, label_batch = gends.get_tfRecords(BATCH_SIZE)

	with tf.Session() as fcSess:
		init_op = tf.global_variables_initializer()
		fcSess.run(init_op)

		#multi-threads start
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=fcSess, coord=coord)

		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(fcSess, ckpt.model_checkpoint_path)
			#xs, ys = mnist.train.next_batch(BATCH_SIZE)
			xs, ys = fcSess.run([img_batch, label_batch])
			_, loss_v, step = fcSess.run([train_op, loss, global_step], feed_dict = {x: xs, y_: ys})
			print "Restored session at %d steps, loss is %g." %(step, loss_v)
		else:
			step = 0
			loss_v = float("inf")

		for i in range(step, TOTAL_STEPS):
			if i % 1000 == 0:
				print "After %d steps, loss is %g." %(step, loss_v)
				saver.save(fcSess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)

			#xs, ys = mnist.train.next_batch(BATCH_SIZE)
			xs, ys = fcSess.run([img_batch, label_batch])
			_, loss_v, step = fcSess.run([train_op, loss, global_step], feed_dict = {x: xs, y_: ys})
		print "After %d steps, loss is %g." %(step, loss_v)
		saver.save(fcSess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)

		#multi-threads end
		coord.request_stop()
		coord.join(threads)


def main():
	#mnist = input_data.read_data_sets(DATA_SAVE_PATH, one_hot = True)
	#bk_propagation(mnist)
	bk_propagation(None)

if __name__ == "__main__":
	main()