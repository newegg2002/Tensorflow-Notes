#coding: utf-8

#This file is the sample code of lecture mnist_test.py

import tensorflow as tf
from   tensorflow.examples.tutorials.mnist import input_data

import time

import mnist_fwp as fwp
import mnist_bkp as bkp

TEST_INTVAL_IN_SECS = 15

def test(mnist):
	with tf.Graph().as_default() as g:
		x  = tf.placeholder(tf.float32, [None, fwp.INPUT_NODE])
		y_ = tf.placeholder(tf.float32, [None, fwp.OUTPUT_NODE])
		y  = fwp.fw_propagation(x, None)

		ema = tf.train.ExponentialMovingAverage(bkp.EMA_DECAY)
		ema_restore = ema.variables_to_restore()
		saver = tf.train.Saver(ema_restore)

		correct_predicition = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_predicition, tf.float32))

		while True:
			with tf.Session() as testSess:
				ckpt = tf.train.get_checkpoint_state(bkp.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(testSess, ckpt.model_checkpoint_path)
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					accuracy_score = testSess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
					print "After %s steps, test accuracy is %g." %(global_step, accuracy_score)
				else:
					print "No checkpoint data found."
					return

			time.sleep(TEST_INTVAL_IN_SECS)

def main():
	mnist = input_data.read_data_sets(bkp.DATA_SAVE_PATH, one_hot = True)
	test(mnist)


if __name__ == "__main__":
	main()