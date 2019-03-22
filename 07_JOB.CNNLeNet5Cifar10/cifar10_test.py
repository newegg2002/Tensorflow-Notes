#coding: utf-8

#This file is the sample code of lecture mnist_lenet5_test.py

import tensorflow as tf
import numpy as np

import time

import cifar10_fwp as fwp
import cifar10_bkp as bkp
import cifar10_gends as gends

TOTAL_TEST_EXAMPLES = 10000
TEST_INTVAL_IN_SECS = 20
CIFAR10_TEST_COUNT  = 2500

def test(mnist):
	with tf.Graph().as_default() as g:
		#x  = tf.placeholder(tf.float32, [None, fwp.INPUT_NODE])
		#同样，调整x的维度
		x  = tf.placeholder(tf.float32, [CIFAR10_TEST_COUNT, fwp.IMAGE_XY_RES, fwp.IMAGE_XY_RES, fwp.IMAGE_CHANS])
		y_ = tf.placeholder(tf.float32, [None, fwp.OUTPUT_NODE])
		y  = fwp.fw_propagation(x, False, None)

		ema = tf.train.ExponentialMovingAverage(bkp.EMA_DECAY)
		ema_restore = ema.variables_to_restore()
		saver = tf.train.Saver(ema_restore)

		correct_predicition = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_predicition, tf.float32))

		img_batch, label_batch = gends.get_tfRecords(CIFAR10_TEST_COUNT, isTrain = False)

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
					xs_reshaped = np.reshape(xs, [CIFAR10_TEST_COUNT, fwp.IMAGE_XY_RES, fwp.IMAGE_XY_RES, fwp.IMAGE_CHANS])

					accuracy_score = testSess.run(accuracy, feed_dict = {x: xs_reshaped, y_: y_s})
					print "After %s steps, test accuracy is %g." %(global_step, accuracy_score)

					#multi-threads end
					coord.request_stop()
					coord.join(threads)
				else:
					print "No checkpoint data found."
					return

			time.sleep(TEST_INTVAL_IN_SECS)

def main():
	test(None)


if __name__ == "__main__":
	main()