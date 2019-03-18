#coding: utf-8

#This file is the sample code of lecture mnist_test.py

import tensorflow as tf
from   tensorflow.examples.tutorials.mnist import input_data

import time

import mnist_fwp as fwp
import mnist_bkp as bkp
import mnist_gends as gends

TOTAL_TEST_EXAMPLES = 10000
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

		img_batch, label_batch = gends.get_tfRecords(TOTAL_TEST_EXAMPLES, isTrain = False)

		while True:
			with tf.Session() as testSess:
				ckpt = tf.train.get_checkpoint_state(bkp.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(testSess, ckpt.model_checkpoint_path)
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

					#multi-threads start
					coord = tf.train.Coordinator()
					threads = tf.train.start_queue_runners(sess=testSess, coord=coord)

					xs, y_s = testSess.run([img_batch, label_batch])

					#accuracy_score = testSess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
					accuracy_score = testSess.run(accuracy, feed_dict = {x: xs, y_: y_s})
					print "After %s steps, test accuracy is %g." %(global_step, accuracy_score)

					#multi-threads end
					coord.request_stop()
					coord.join(threads)
				else:
					print "No checkpoint data found."
					return

			time.sleep(TEST_INTVAL_IN_SECS)

def main():
	#mnist = input_data.read_data_sets(bkp.DATA_SAVE_PATH, one_hot = True)
	#test(mnist)
	test(None)


if __name__ == "__main__":
	main()