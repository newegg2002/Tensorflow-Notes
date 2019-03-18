#coding: utf-8

#This file is the sample code of lecture mnist_app.py

import tensorflow as tf
import numpy as np
from PIL import Image

import mnist_fwp as fwp
import mnist_bkp as bkp


IMG_X_RESOLUTION = 28
IMG_Y_RESOLUTION = 28
BLACK_THRESHOLD = 50
ALL_WHITE_VALUE = 255


def FCN_restore_model(imgarr_pped):
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, [None, fwp.INPUT_NODE])
		y = fwp.fw_propagation(x, None)
		resultV = tf.argmax(y, 1)

		var_avg = tf.train.ExponentialMovingAverage(bkp.EMA_DECAY)
		var_to_restore = var_avg.variables_to_restore()
		saver = tf.train.Saver(var_to_restore)

		with tf.Session() as fcnSess:
			ckpt = tf.train.get_checkpoint_state(bkp.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(fcnSess, ckpt.model_checkpoint_path)
				resultV = fcnSess.run(resultV, feed_dict = {x:imgarr_pped})
				return resultV
			else:
				print "No checkpoint data found."
				return -1

def pre_process_pic(testpic_path):
	img = Image.open(testpic_path)
	reImg = img.resize((IMG_X_RESOLUTION, IMG_Y_RESOLUTION), Image.ANTIALIAS)
	imgarr = np.array(reImg.convert('L'))
	
	for x in xrange(IMG_X_RESOLUTION):
		for y in xrange(IMG_Y_RESOLUTION):
			imgarr[x][y] = ALL_WHITE_VALUE - imgarr[x][y]
			if (imgarr[x][y] < BLACK_THRESHOLD):
				imgarr[x][y] = 0
			else:
				imgarr[x][y] = ALL_WHITE_VALUE

	nm_arr = imgarr.reshape(1, IMG_X_RESOLUTION * IMG_Y_RESOLUTION)
	nm_arr = nm_arr.astype(np.float32)
	img_pped = np.multiply(nm_arr, 1./255.)

	return img_pped

def application():
	testcnt = input("Please input the count of test items:")

	for i in range(testcnt):
		testpic = raw_input("Please input the name of test picture:")
		testpic = "./pic/" + testpic + ".png"
		testpicarr = pre_process_pic(testpic)
		resultV = FCN_restore_model(testpicarr)
		print "The prediction number is:", resultV


def main():
	application()

if __name__ == '__main__':
	main()